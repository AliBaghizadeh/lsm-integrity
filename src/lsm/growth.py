"""
Stage 8: cross-run growth-rate estimation, remaining-life projection, and the
"no growth" baseline.

Cross-survey physical-defect identity is ALREADY solved by
`truth.build_truth_registry` (defect position is fixed across a line's runs,
only severity grows) and `train._match_all_indications` (which matches every
run's indications against that SAME per-line registry) -- this module's only
new logic is sorting the already-aligned points by `run_id` and fitting a
rate over them, plus projecting that rate forward to a limit state.

Closed-form empirical-Bayes shrinkage throughout -- no PyMC/Stan, matching
this project's established practice (bootstrap CIs, split conformal,
isotonic calibration) of defensible, dependency-light statistics.

Run: python -m lsm forecast [--as-of DATE]
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from lsm import bundle as bundle_module
from lsm import dig_feedback, train
from lsm import predict as predict_module
from lsm.config import Config
from lsm.evaluate import bootstrap_ci, paired_bootstrap_ci
from lsm.hashing import data_sha256
from lsm.indications import attach_severity
from lsm.model_card import append_growth_section

GATE_GROWTH_MARGIN = (
    0.0  # baseline_mae - model_mae must exceed this at the CI lower bound
)


def build_growth_frame(matched_all: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per (matched_source_id, survey_id, run_id): `y_true`
    (severity_smys at that peak row), `distance_m` (the chainage MATCH
    RESIDUAL PLAN.md asks to record -- already produced by
    `match_dug_indications`, just carried through here), sorted by
    (matched_source_id, run_id). Defect-only.

    `generate.py` initialises severity_smys to NaN off-defect (see
    train.py::_build_severity_training_frame's own comment, found via Stage
    6) -- the same rare edge case (a detected peak lands just outside a
    defect's exact label window while still within the looser dig-matching
    tolerance) can produce a NaN `y_true` here too. Dropped with the same
    loud guard, a small, deliberate, self-contained duplication of that
    logic rather than importing train.py's private frame builder (which
    also attaches feature columns and a CV fold this module doesn't need).
    """
    columns = ["matched_source_id", "survey_id", "run_id", "y_true", "distance_m"]
    defects_only = matched_all[matched_all["matched_kind"] == "defect"].copy()
    if len(defects_only) == 0:
        return pd.DataFrame(columns=columns)

    truth = corpus[["survey_id", "chainage_m", "severity_smys"]].rename(
        columns={"chainage_m": "chainage_peak_m", "severity_smys": "y_true"}
    )
    with_truth = defects_only.merge(
        truth, on=["survey_id", "chainage_peak_m"], how="left"
    )
    run_ids = corpus[["survey_id", "run_id"]].drop_duplicates()
    with_run = with_truth.merge(run_ids, on="survey_id", how="left")

    n_nan = int(with_run["y_true"].isna().sum())
    if n_nan > 0:
        print(
            f"build_growth_frame: dropping {n_nan} matched indication(s) whose peak row "
            "has a NaN severity_smys (peak landed outside the true label window while "
            "still within the dig-matching tolerance) -- see train.py's comment."
        )
        with_run = with_run[with_run["y_true"].notna()]

    out = (
        with_run[columns]
        .sort_values(["matched_source_id", "run_id"])
        .reset_index(drop=True)
    )
    return out


def _ols_log_slope(run_ids: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Simple least-squares slope of log(y_true) ~ run_id, plus its residual-
    based variance (nan if fewer than 3 points -- 2 points determine a line
    exactly, with no residual left to estimate a variance from).
    """
    x = run_ids.astype(float)
    y = np.log(y_true.astype(float))
    x_mean, y_mean = x.mean(), y.mean()
    var_x = float(np.sum((x - x_mean) ** 2))
    if var_x == 0:
        return float("nan"), float("nan")
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / var_x)
    if len(x) > 2:
        resid = y - (y_mean + slope * (x - x_mean))
        slope_var = float(np.sum(resid**2) / (len(x) - 2) / var_x)
    else:
        slope_var = float("nan")
    return slope, slope_var


def fit_population_log_rate(growth_frame: pd.DataFrame) -> tuple[float, float]:
    """Inverse-variance-weighted pooling of every defect's own OLS log-linear
    slope (log(y_true) ~ run_id), for defects with >= 2 observations. A
    2-observation defect's own variance can't be estimated (see
    `_ols_log_slope`) and is weighted equally with the rest rather than
    dropped. Returns (pooled_rate, pooled_rate_variance); (0.0, inf) if no
    defect has >= 2 observations at all. On this corpus's synthetic law,
    pooled_rate should land near ln(1.15) ~= 0.1398 -- checked directly in
    tests/test_growth.py.
    """
    slopes, weights = [], []
    for _, group in growth_frame.groupby("matched_source_id"):
        if len(group) < 2:
            continue
        slope, slope_var = _ols_log_slope(
            group["run_id"].to_numpy(), group["y_true"].to_numpy()
        )
        if not np.isfinite(slope):
            continue
        weight = 1.0 / slope_var if np.isfinite(slope_var) and slope_var > 0 else 1.0
        slopes.append(slope)
        weights.append(weight)
    if not slopes:
        return 0.0, float("inf")
    slopes_arr, weights_arr = np.array(slopes), np.array(weights)
    pooled_rate = float(np.average(slopes_arr, weights=weights_arr))
    pooled_var = float(1.0 / weights_arr.sum())
    return pooled_rate, pooled_var


def fit_defect_rates(
    growth_frame: pd.DataFrame,
    population_rate: float,
    population_var: float,
    min_observations_for_own_rate: int,
) -> pd.DataFrame:
    """Per `matched_source_id`: n_obs, own_rate (nan below the minimum),
    own_var, shrunk_rate (precision-weighted convex combination of own_rate
    and population_rate), shrinkage_weight in [0, 1]. A defect below the
    minimum, or with only 2 observations (own_var not independently
    estimable), gets shrinkage_weight=0 (fully population-shrunk) or borrows
    the population's own variance respectively -- never a fabricated
    precision from a variance that was never actually measured.
    """
    rows = []
    pop_precision = (
        1.0 / population_var
        if np.isfinite(population_var) and population_var > 0
        else 0.0
    )
    for source_id, group in growth_frame.groupby("matched_source_id"):
        n_obs = len(group)
        if n_obs < min_observations_for_own_rate:
            rows.append(
                {
                    "matched_source_id": source_id,
                    "n_obs": n_obs,
                    "own_rate": float("nan"),
                    "own_var": float("nan"),
                    "shrunk_rate": population_rate,
                    "shrinkage_weight": 0.0,
                }
            )
            continue
        own_rate, own_var = _ols_log_slope(
            group["run_id"].to_numpy(), group["y_true"].to_numpy()
        )
        if not np.isfinite(own_rate):
            rows.append(
                {
                    "matched_source_id": source_id,
                    "n_obs": n_obs,
                    "own_rate": float("nan"),
                    "own_var": float("nan"),
                    "shrunk_rate": population_rate,
                    "shrinkage_weight": 0.0,
                }
            )
            continue
        if not np.isfinite(own_var):
            own_var = (
                population_var  # 2 observations: borrow the population's own variance
            )
        own_precision = 1.0 / own_var if np.isfinite(own_var) and own_var > 0 else 0.0
        total_precision = own_precision + pop_precision
        weight = own_precision / total_precision if total_precision > 0 else 0.0
        shrunk_rate = weight * own_rate + (1 - weight) * population_rate
        rows.append(
            {
                "matched_source_id": source_id,
                "n_obs": n_obs,
                "own_rate": own_rate,
                "own_var": own_var,
                "shrunk_rate": shrunk_rate,
                "shrinkage_weight": weight,
            }
        )
    return pd.DataFrame(rows)


def no_growth_baseline_predict(last_observed_severity: np.ndarray) -> np.ndarray:
    """The baseline PLAN.md requires first: next severity = last observed,
    unchanged. Compared against the model on ONE-STEP-AHEAD SEVERITY MAE at
    a held-out run, never on remaining-life directly (an unchanging severity
    implies infinite life, which has no MAE against a finite one).
    """
    return np.asarray(last_observed_severity, dtype=float)


def project_remaining_life(
    current_severity: tuple[float, float, float],
    shrunk_rate: float,
    limit_state_smys: float,
) -> tuple[float, float, float]:
    """Solves log(limit_state) = log(current) + rate * t for t (in
    run-intervals), applied to each of the current severity's (lo, med, hi)
    bounds independently. A LOWER current severity has more headroom under
    the same rate and therefore takes LONGER to reach the limit -- so
    `life_hi` (the longest remaining life) pairs with `current_lo`, and
    `life_lo` pairs with `current_hi`. Monotonicity, not a mislabel.

    STATED LIMITATION (also in the model card): this propagates the
    severity model's own current-value uncertainty only, not growth-rate
    ESTIMATION uncertainty itself -- a full predictive distribution would
    need both, but this project's established practice throughout is
    closed-form, defensible statistics over exotic uncertainty quantification.
    """
    current_lo, current_med, current_hi = current_severity

    def _life(current: float) -> float:
        if current <= 0 or current >= limit_state_smys:
            return 0.0
        if shrunk_rate <= 0:
            return float("inf")
        return float((np.log(limit_state_smys) - np.log(current)) / shrunk_rate)

    life_hi = _life(current_lo)
    life_med = _life(current_med)
    life_lo = _life(current_hi)
    return life_lo, life_med, life_hi


def evaluate_growth_gate(
    growth_frame: pd.DataFrame,
    train_run_ids: frozenset[int],
    test_run_id: int,
    min_observations_for_own_rate: int,
    boot_cfg,
    seed: int,
) -> dict:
    """PLAN.md's literal gate, the same split SHAPE as
    `scale_eval.run_temporal_holdout` (train_run_ids={0,1}, test_run_id=2)
    but living here permanently as production code, not a scale rehearsal:
    fit population + per-defect rates on `train_run_ids` only, project each
    matched defect's severity forward to `test_run_id` from its LAST
    training-window observation, compare one-step-ahead MAE (paired over
    defects) against the no-growth baseline.
    """
    train_frame = growth_frame[growth_frame["run_id"].isin(train_run_ids)]
    test_frame = growth_frame[growth_frame["run_id"] == test_run_id]

    population_rate, population_var = fit_population_log_rate(train_frame)
    defect_rates = fit_defect_rates(
        train_frame, population_rate, population_var, min_observations_for_own_rate
    )
    rate_by_defect = defect_rates.set_index("matched_source_id")["shrunk_rate"]

    model_errors, baseline_errors = [], []
    for source_id, test_group in test_frame.groupby("matched_source_id"):
        train_obs = train_frame[train_frame["matched_source_id"] == source_id]
        if len(train_obs) == 0 or source_id not in rate_by_defect.index:
            continue
        last_obs = train_obs.loc[train_obs["run_id"].idxmax()]
        shrunk_rate = rate_by_defect.loc[source_id]
        for _, test_row in test_group.iterrows():
            dt_runs = test_row["run_id"] - last_obs["run_id"]
            model_pred = last_obs["y_true"] * np.exp(shrunk_rate * dt_runs)
            baseline_pred = no_growth_baseline_predict(np.array([last_obs["y_true"]]))[
                0
            ]
            model_errors.append(abs(model_pred - test_row["y_true"]))
            baseline_errors.append(abs(baseline_pred - test_row["y_true"]))

    model_mae = bootstrap_ci(model_errors, boot_cfg.n_resamples, boot_cfg.level, seed)
    baseline_mae = bootstrap_ci(
        baseline_errors, boot_cfg.n_resamples, boot_cfg.level, seed + 1
    )
    gap = paired_bootstrap_ci(
        baseline_errors, model_errors, boot_cfg.n_resamples, boot_cfg.level, seed + 2
    )
    gate_passed = bool(
        gap[1] > GATE_GROWTH_MARGIN
    )  # CI lower bound, not the point estimate

    return {
        "population_log_rate": population_rate,
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "gap_baseline_minus_model": gap,
        "gate_passed": gate_passed,
        "n_defects_evaluated": len(model_errors),
    }


def _project_remaining_life_frame(
    growth_frame: pd.DataFrame,
    defect_rates: pd.DataFrame,
    matched_all: pd.DataFrame,
    corpus: pd.DataFrame,
    severity_bundle: dict | None,
    growth_cfg: dict,
) -> pd.DataFrame:
    """Production remaining-life: for every matched defect, the "current
    severity" anchor is the RELEASED severity bundle's (sev_pred, sev_lo,
    sev_hi) on its LATEST-run indication -- via `attach_severity`, the exact
    function `predict.py` already uses -- because in production there is no
    ground truth to project from, only the model's own calibrated interval.
    Empty if no severity model has been released yet (severity is estimated
    for every indication regardless of a growth model's existence, but
    remaining life needs a starting point to project from).
    """
    columns = [
        "matched_source_id",
        "n_obs",
        "shrinkage_weight",
        "shrunk_rate",
        "sev_pred",
        "sev_lo",
        "sev_hi",
        "life_lo",
        "life_med",
        "life_hi",
    ]
    if severity_bundle is None or len(growth_frame) == 0:
        return pd.DataFrame(columns=columns)

    defects_only = matched_all[matched_all["matched_kind"] == "defect"].copy()
    run_ids = corpus[["survey_id", "run_id"]].drop_duplicates()
    defects_only = defects_only.merge(run_ids, on="survey_id", how="left")
    latest_idx = defects_only.groupby("matched_source_id")["run_id"].idxmax()
    latest_indications = defects_only.loc[latest_idx].reset_index(drop=True)

    base_feature_cols = [c for c in severity_bundle["feature_cols"] if c != "extent_m"]
    with_severity = attach_severity(
        latest_indications,
        corpus,
        severity_bundle["model"],
        base_feature_cols,
        nominal_coverage=1.0,
    )

    rate_lookup = defect_rates.set_index("matched_source_id")
    rows = []
    for _, row in with_severity.iterrows():
        source_id = row["matched_source_id"]
        if source_id not in rate_lookup.index or pd.isna(row["sev_pred"]):
            continue
        rate_row = rate_lookup.loc[source_id]
        life_lo, life_med, life_hi = project_remaining_life(
            (row["sev_lo"], row["sev_pred"], row["sev_hi"]),
            rate_row["shrunk_rate"],
            growth_cfg["limit_state_smys"],
        )
        rows.append(
            {
                "matched_source_id": source_id,
                "n_obs": rate_row["n_obs"],
                "shrinkage_weight": rate_row["shrinkage_weight"],
                "shrunk_rate": rate_row["shrunk_rate"],
                "sev_pred": row["sev_pred"],
                "sev_lo": row["sev_lo"],
                "sev_hi": row["sev_hi"],
                "life_lo": life_lo,
                "life_med": life_med,
                "life_hi": life_hi,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def run_forecast(cfg: Config, conn, as_of: str | None = None) -> dict:
    """Stage 8 orchestrator, analogous to `train.run_train`. Reuses
    `train.py`'s already-private helpers directly via `train._foo(...)`,
    the exact pattern `scale_eval.py` already established for the same
    reason -- no signature or visibility changes to `train.py` are needed.

    `as_of` defaults to now() (matching `run_train`'s own default);
    threading an earlier ISO date exercises `load_feature_corpus`'s
    point-in-time plumbing for real, for the first time anywhere in this
    project (every other call site always passes "now").
    """
    now = dt.datetime.now(dt.UTC)
    as_of = as_of or now.isoformat()
    feature_version = cfg.base.features.version

    corpus = train.load_feature_corpus(
        cfg.env.storage.feature_dir, feature_version, as_of=as_of
    )
    if len(corpus) == 0:
        raise RuntimeError("empty feature corpus -- run `lsm features` first")

    survey_ids = sorted(corpus["survey_id"].unique())
    truth_geo = train._load_truth_and_geometry(conn, survey_ids)
    corpus = corpus.merge(
        truth_geo, on=["survey_id", "sample_idx"], how="left", validate="one_to_one"
    )
    feature_cols = train.feature_columns(cfg.base.features)

    line_ids = sorted(corpus["line_id"].unique())
    registries = []
    for line_id in line_ids:
        ref_survey_id = min(
            corpus.loc[corpus["line_id"] == line_id, "survey_id"].unique()
        )
        ref_rows = corpus[corpus["survey_id"] == ref_survey_id]
        registries.append(
            train.build_truth_registry(
                ref_rows, line_id, ref_rows["chainage_m"].to_numpy()
            )
        )
    registry = pd.concat(registries, ignore_index=True)

    pipeline_version, anomaly_version, severity_version, _classify_version = (
        predict_module.latest_pipeline_release(conn)
    )
    model_dir = Path(cfg.env.storage.model_dir)
    anomaly_bundle = bundle_module.load_bundle(
        model_dir / anomaly_version / "bundle.joblib",
        expected_feature_version=feature_version,
        expected_schema_version=cfg.base.schema_version,
    )
    corpus["_score"] = anomaly_bundle["model"].score(corpus)
    all_indications = train._cluster_all_indications(
        corpus, "_score", anomaly_bundle["threshold"]
    )
    matched_all = train._match_all_indications(all_indications, corpus, registry)
    growth_frame = build_growth_frame(matched_all, corpus)

    growth_cfg = cfg.base.model.growth
    boot_cfg = cfg.base.model.bootstrap
    gate = evaluate_growth_gate(
        growth_frame,
        frozenset(growth_cfg["train_run_ids"]),
        growth_cfg["test_run_id"],
        growth_cfg["min_observations_for_own_rate"],
        boot_cfg,
        cfg.seed,
    )

    population_rate, population_var = fit_population_log_rate(growth_frame)
    defect_rates = fit_defect_rates(
        growth_frame,
        population_rate,
        population_var,
        growth_cfg["min_observations_for_own_rate"],
    )

    severity_bundle = None
    if severity_version is not None:
        severity_bundle = bundle_module.load_bundle(
            model_dir / severity_version / "bundle.joblib",
            expected_feature_version=feature_version,
            expected_schema_version=cfg.base.schema_version,
        )
    remaining_life_frame = _project_remaining_life_frame(
        growth_frame,
        defect_rates,
        matched_all,
        corpus,
        severity_bundle,
        growth_cfg,
    )

    result = {
        "growth": {"mae": gate["model_mae"]},
        "growth_baseline": {"mae": gate["baseline_mae"]},
        "growth_gap_baseline_minus_model": gate["gap_baseline_minus_model"],
        "growth_gate_passed": gate["gate_passed"],
        "growth_population_log_rate": gate["population_log_rate"],
        "n_growth_samples": len(growth_frame),
        "n_defects_evaluated": gate["n_defects_evaluated"],
        "median_days_to_verification": dig_feedback.median_days_to_verification(conn),
    }

    _persist_growth(
        cfg,
        conn,
        now,
        pipeline_version,
        as_of,
        survey_ids,
        result,
        population_rate,
        population_var,
        defect_rates,
        remaining_life_frame,
        growth_cfg,
        feature_cols,
    )
    return result


def _persist_growth(
    cfg: Config,
    conn,
    now: dt.datetime,
    pipeline_version: str,
    as_of: str,
    survey_ids: list[str],
    result: dict,
    population_rate: float,
    population_var: float,
    defect_rates: pd.DataFrame,
    remaining_life_frame: pd.DataFrame,
    growth_cfg: dict,
    feature_cols: list[str],
) -> None:
    """Persists a `model_run` row (task='growth'), a bundle, a
    `pipeline_release.growth_version` UPDATE (never a new release row --
    growth is one more model type attached to the SAME already-released
    pipeline, per SKILL invariant #11: an indication comes from up to four
    models plus a feature version, promoted/rolled back together), a
    `growth_forecast.parquet` audit artifact, an MLflow run, and a model-card
    section.
    """
    git_sha = train._git_sha()
    content_hashes = [
        row[0]
        for row in conn.execute(
            f"SELECT content_sha256 FROM survey WHERE survey_id IN ({','.join('?' * len(survey_ids))})",
            survey_ids,
        ).fetchall()
    ]
    corpus_data_sha256 = data_sha256(content_hashes)

    mlflow.set_tracking_uri(cfg.env.mlflow["tracking_uri"])
    mlflow.set_experiment(cfg.base.mlflow.get("experiment", "lsm-integrity"))
    with mlflow.start_run(run_name="stage8-growth") as run:
        mlflow.log_params(
            {
                "config_sha256": cfg.config_sha256,
                "git_sha": git_sha,
                "data_sha256": corpus_data_sha256,
                "seed": cfg.seed,
                "as_of": as_of,
                "limit_state_smys": growth_cfg["limit_state_smys"],
            }
        )
        mlflow.log_metrics(
            {
                "population_log_rate": result["growth_population_log_rate"],
                "model_mae": result["growth"]["mae"][0],
                "baseline_mae": result["growth_baseline"]["mae"][0],
                "n_growth_samples": result["n_growth_samples"],
            }
        )

        date_tag = now.strftime("%Y.%m.%d")
        short_sha = train._short_sha(git_sha)
        growth_version = f"growth-eb-{date_tag}-{short_sha}"
        artifact_path = (
            Path(cfg.env.storage.model_dir) / growth_version / "bundle.joblib"
        )
        bundle_module.save_bundle(
            artifact_path,
            {
                "task": "growth",
                "model_kind": "empirical_bayes_log_linear",
                "feature_cols": feature_cols,
                "population_log_rate": population_rate,
                "population_log_rate_var": population_var,
                "defect_rates": defect_rates.to_dict(orient="records"),
                "limit_state_smys": growth_cfg["limit_state_smys"],
                "assumed_interval_years": growth_cfg["assumed_interval_years"],
                "feature_version": cfg.base.features.version,
                "schema_version": cfg.base.schema_version,
                "config_sha256": cfg.config_sha256,
                "git_sha": git_sha,
                "data_sha256": corpus_data_sha256,
                "truth_as_of": as_of,
            },
        )

        conn.execute(
            "INSERT OR REPLACE INTO model_run (model_version, task, mlflow_run_id, git_sha, "
            "config_sha256, data_sha256, feature_version, truth_as_of, trained_at, "
            "metrics_json, artifact_uri, final_test_uses) VALUES (?,?,?,?,?,?,?,?,?,?,?,0)",
            (
                growth_version,
                "growth",
                run.info.run_id,
                git_sha,
                cfg.config_sha256,
                corpus_data_sha256,
                cfg.base.features.version,
                as_of,
                now.isoformat(),
                json.dumps(
                    {
                        "population_log_rate": result["growth_population_log_rate"],
                        "mae": result["growth"]["mae"],
                        "baseline_mae": result["growth_baseline"]["mae"],
                        "gate_passed": result["growth_gate_passed"],
                    },
                    default=list,
                ),
                str(artifact_path),
            ),
        )
        conn.execute(
            "UPDATE pipeline_release SET growth_version=? WHERE pipeline_version=?",
            (growth_version, pipeline_version),
        )
        conn.commit()

        reports_dir = Path(cfg.env.storage.reports_dir) / pipeline_version
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "growth_forecast.parquet"
        remaining_life_frame.to_parquet(report_path, index=False)
        mlflow.log_artifact(str(report_path))

        model_card_path = (
            Path(cfg.env.storage.model_dir) / pipeline_version / "model_card.md"
        )
        if model_card_path.exists():
            append_growth_section(
                model_card_path,
                provenance={
                    "limit_state_smys": growth_cfg["limit_state_smys"],
                    "assumed_interval_years": growth_cfg["assumed_interval_years"],
                    "synthetic_growth_rate": cfg.base.data.growth,
                },
                result=result,
            )
            mlflow.log_artifact(str(model_card_path))
