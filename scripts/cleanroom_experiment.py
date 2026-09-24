"""The clean-room experiment: what would an IN-HOUSE controlled campaign buy?

The proposal this measures: rather than only ever seeing real buried pipeline
-- where a defect's anomaly arrives mixed with external interference and a
    girth weld every ~12 m -- the reference setup uses isolated test spools. Isolated pipe,
known defects, no interference, no construction features, chainage from a
tape measure rather than a dropping-out GPS. Does data like that actually
help the model that has to work in the FIELD?

Two experiments, deliberately separated, because only the second one answers
the question a budget holder is really asking:

  E1  CEILING (diagnostic).  Run the identical Stage 3/4/5 evaluation
      within each domain. Field-vs-clean-room side by side decomposes the
      current failure into "confounder-limited" (the clean room fixes it)
      versus "physics-limited" (the scalar-projection/noise floor, which no
      amount of clean data touches). On its own this proves nothing about
      the proposal -- removing the confounders from a confounder-limited
      problem HAS to make it easier. It is reported as a diagnostic, not as
      evidence for the campaign.

  E2  TRANSFER (the actual answer).  A 2x2: train on one domain's
      indications, evaluate on the other's held-out LINES.

                          test: clean-room     test: field
        train: clean-room    ceiling            <- THE CELL THAT DECIDES
        train: field         reverse transfer   current baseline

      Reporting only the top-left is the self-flattering version of this
      experiment. The top-right is what a lab campaign actually buys, since
      the deployment target is messy real pipe either way.

WHICH MODELS THIS TARGETS.  Stage 3 (detect) is IsolationForest -- unsupervised,
refit per fold. Labelled ground truth does not feed it, so E2 does not run a
detection cell; E1 reports detection for both domains purely as the ceiling
diagnostic. Stage 4 (severity) and Stage 5 (classify) are supervised and are
where a labelled in-house campaign can actually pay -- and Stage 5 is the gate
this project currently fails (SCC recall ~0). Those two carry E2.

WHAT THIS CANNOT SHOW, stated up front rather than left for a reader to find:
in a synthetic corpus the ground truth exists by construction, so what is
measured here is the value of REMOVING CONFOUNDERS and HAVING LABELS. The
largest real-world payoff of an in-house campaign -- correcting the physics
assumptions in the forward model itself -- cannot be demonstrated by a study
whose physics is the assumption under test.

SAFETY: like scripts/ablation_ladder.py, this NEVER touches `data/raw`,
`data/lsm.db` or any other production path -- every survey goes under an
isolated tmp directory and truth is read off the generator's own raw Parquet
columns, never via SQLite.

SCALE: CLEANROOM_N_LINES lines per domain (default 6 -> a 3-line train / 3-line
test split, 36 physical defects per side per domain) at the SAME per-line
length/defect/interference/run density as `config/base.yaml` production. Defect
DENSITY is held identical across the two domains on purpose: a real test
facility would pack defects far closer together than 6/km, and this project has
already measured that dense packing degrades background contrast on its own
(config/base.yaml's `n_lines` comment, 3.19x -> ~2.7-3.0x) -- letting the
clean-room corpus also be denser would confound "no interference" with "worse
detrend baseline". That density question is a separate experiment, not this one.

Usage:
    python scripts/cleanroom_experiment.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from lsm import train as train_module
from lsm.config import load_config
from lsm.evaluate import add_fold_column
from lsm.features import SurveyContext, compute_survey_features, feature_columns
from lsm.generate import generate_all
from lsm.models.classify import ClassifyModel, MajorityClassBaseline
from lsm.models.severity import GlobalMeanSeverityBaseline, SeverityModel
from lsm.registration import register_survey
from lsm.schemas import CLASSIFY_CLASSES
from lsm.truth import build_truth_registry

# Deliberate scale-down from production's 5 lines, same treatment as
# ablation_ladder.py's ABLATION_N_LINES -- but note this one must stay EVEN
# and >= 4, since E2 splits the lines in half into train/test halves.
N_LINES = int(os.environ.get("CLEANROOM_N_LINES", "6"))

DOMAINS = ("field", "cleanroom")


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def _isolated_config(domain: str):
    """A Config pointed at a throwaway tmp directory, NEVER `data/raw` or
    `data/lsm.db` -- see module docstring's SAFETY note.

    `cleanroom` differs from `field` in exactly three physical ways, all of
    them things a real isolated test spool genuinely lacks:
      * no external interference sources  (n_interference = 0)
      * no girth-weld train               (weld.enabled = False)
      * chainage from a tape measure, not GPS dead-reckoning (applied later,
        in `_build_corpus` -- it is a measurement choice, not a config field)
    Everything else -- rig, walk model, sensor imperfections, GPS dropout in
    the raw file, defect physics, severity ranges, growth -- is identical, so
    any difference measured here is attributable to those three and not to a
    quietly easier simulation.
    """
    cfg = load_config("dev")
    tmp = Path(tempfile.mkdtemp(prefix=f"lsm_cleanroom_{domain}_"))
    cfg.env.storage.raw_dir = str(tmp / "raw")
    cfg.env.storage.sqlite_path = str(
        tmp / "lsm.db"
    )  # unused (no ingest), set for hygiene only
    cfg.env.storage.feature_dir = str(tmp / "features")
    cfg.env.storage.model_dir = str(tmp / "models")
    cfg.env.storage.reports_dir = str(tmp / "reports")
    cfg.base.data.n_lines = N_LINES
    cfg.base.data.rig = "scalar"
    if domain == "cleanroom":
        cfg.base.data.n_interference = 0
        cfg.base.data.weld = cfg.base.data.weld.model_copy(update={"enabled": False})
    return cfg


def _truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Truth labels straight off the raw survey's own columns -- no SQLite
    ingest (see module docstring's SAFETY note).
    """
    return df[
        ["sample_idx", "defect", "defect_type", "interference", "severity_smys"]
    ].copy()


def _build_corpus(
    cfg, domain: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    """Generate one domain's corpus and compute the FULL fv=3 feature set.

    Chainage differs by domain, and deliberately so -- it is one of the three
    things being tested:
      * field     -- Stage B's real weld-comb + GPS dead-reckoning registration
                     (`register_survey`), exactly what `lsm features` runs in
                     production.
      * cleanroom -- `chainage_true_m` directly, i.e. a tape-measured test
                     spool. `dist_to_weld_m` is all-NaN because there ARE no
                     welds to measure a distance to; the column is kept (not
                     dropped) so both domains carry an identical 45-column
                     feature contract and a cross-domain model never sees a
                     different feature list than it was fitted on.

    Returns (corpus, registry, run_line_id, survey_ids).
    """
    results = generate_all(cfg.base.data, cfg.env.storage.raw_dir, seed=cfg.seed)
    frames = []
    for sr in results:
        df = pd.read_parquet(sr.path)
        if domain == "field":
            reg = register_survey(df, cfg.base.data)
            chainage_m, dist_to_weld_m = reg.chainage_m, reg.dist_to_weld_m
        else:
            chainage_m = df["chainage_true_m"].to_numpy(dtype=float)
            dist_to_weld_m = np.full(len(df), np.nan)
        ctx = SurveyContext(
            survey_id=sr.survey_id,
            line_id=sr.line_id,
            run_id=sr.run_id,
            surveyed_at=sr.surveyed_at,
            standoff_m=sr.standoff_m,
            array_spacing_m=cfg.base.data.array.spacing_m,
        )
        feat = compute_survey_features(
            df,
            ctx,
            cfg.base.features,
            chainage_m=chainage_m,
            dist_to_weld_m=dist_to_weld_m,
        )
        feat = feat.merge(
            _truth_columns(df), on="sample_idx", how="left", validate="one_to_one"
        )
        frames.append(feat)
    corpus = pd.concat(frames, ignore_index=True)

    registries = []
    for line_id in sorted(corpus["line_id"].unique()):
        ref_survey_id = min(
            corpus.loc[corpus["line_id"] == line_id, "survey_id"].unique()
        )
        ref_rows = corpus[corpus["survey_id"] == ref_survey_id]
        registries.append(
            build_truth_registry(ref_rows, line_id, ref_rows["chainage_m"].to_numpy())
        )
    registry = pd.concat(registries, ignore_index=True)

    run_line_id = (
        corpus.drop_duplicates("survey_id").set_index("survey_id")["line_id"].to_dict()
    )
    return corpus, registry, run_line_id, sorted(run_line_id)


# ---------------------------------------------------------------------------
# E2: cross-domain fits
# ---------------------------------------------------------------------------


def _split_lines() -> tuple[list[str], list[str]]:
    """First half of the lines train, second half test -- the SAME line ids in
    both domains, so every cell of the 2x2 is evaluated on genuinely held-out
    lines and the four cells are directly comparable to each other.
    """
    line_ids = [f"LINE{i:03d}" for i in range(N_LINES)]
    half = N_LINES // 2
    return line_ids[:half], line_ids[half:]


def _cross_domain_severity(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
    cfg,
    seed: int,
) -> tuple[dict, dict] | None:
    """Fit SeverityModel (+ its conformal margin) on `train_frame`, predict on
    `test_frame`. Same inner train/calibration split by DEFECT as
    `train._run_severity_cv` -- conformal's coverage guarantee needs the
    calibration set disjoint from what fit the quantile models, and that is
    just as true across domains as within one.

    Returns (model_metrics, baseline_metrics), or None if the frames are too
    small to fit at all.
    """
    if len(train_frame) < 4 or len(test_frame) == 0:
        return None
    sev_cfg = cfg.base.model.severity
    rng = np.random.default_rng(seed)

    defects = train_frame["matched_source_id"].unique()
    rng.shuffle(defects)
    n_calib = (
        max(1, round(len(defects) * train_module.CONFORMAL_CALIB_FRACTION))
        if len(defects) > 1
        else 0
    )
    calib_ids = set(defects[:n_calib])
    train = train_frame[~train_frame["matched_source_id"].isin(calib_ids)]
    calib = train_frame[train_frame["matched_source_id"].isin(calib_ids)]
    if len(train) == 0:
        train, calib = train_frame, train_frame.iloc[0:0]

    model = SeverityModel(
        feature_cols=feature_cols,
        quantiles=tuple(sev_cfg["quantiles"]),
        conformal_alpha=sev_cfg["conformal_alpha"],
        seed=seed,
        lgbm_cfg=cfg.base.model.lightgbm.model_dump(),
        min_child_samples=sev_cfg.get("min_child_samples", 3),
        n_estimators=sev_cfg.get("n_estimators", 50),
        num_leaves=sev_cfg.get("num_leaves", 7),
    ).fit(train, train["y_true"].to_numpy(), calib, calib["y_true"].to_numpy())
    med, lo, hi = model.predict(test_frame)

    baseline = GlobalMeanSeverityBaseline(
        conformal_alpha=sev_cfg["conformal_alpha"]
    ).fit(train["y_true"].to_numpy(), calib["y_true"].to_numpy())
    bmed, blo, bhi = baseline.predict(len(test_frame))

    ids = test_frame["matched_source_id"].to_numpy()
    y_true = test_frame["y_true"].to_numpy()
    oof_model = pd.DataFrame(
        {"defect_id": ids, "y_true": y_true, "y_pred": med, "lo": lo, "hi": hi}
    )
    oof_base = pd.DataFrame(
        {"defect_id": ids, "y_true": y_true, "y_pred": bmed, "lo": blo, "hi": bhi}
    )
    return (
        train_module._severity_bootstrap_metrics(oof_model, cfg, seed=seed + 20),
        train_module._severity_bootstrap_metrics(oof_base, cfg, seed=seed + 30),
    )


def _cross_domain_classify(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
    cfg,
    seed: int,
) -> tuple[dict, dict] | None:
    """Fit ClassifyModel (+ isotonic calibration) on `train_frame`, predict on
    `test_frame`. Class-stratified defect-grouped calibration split, same as
    `train._run_classify_cv`.

    Note the asymmetry this experiment exists to expose: a clean-room training
    frame contains NO `interference` class at all (there is no interference on
    an isolated spool), so a clean-room-trained classifier structurally cannot
    predict it. That is not a bug to work around -- it is one of the concrete
    costs of the proposal, and it shows up honestly as interference recall and
    precision when such a model is evaluated on field data.
    """
    if (
        len(train_frame) < 8
        or len(test_frame) == 0
        or train_frame["defect_type"].nunique() < 2
    ):
        return None
    classify_cfg = cfg.base.model.classify
    lgbm_cfg = {
        **cfg.base.model.lightgbm.model_dump(),
        "n_estimators": classify_cfg.get("n_estimators", 50),
        "num_leaves": classify_cfg.get("num_leaves", 7),
    }
    rng = np.random.default_rng(seed)

    train_ids, calib_ids = train_module._stratified_defect_calib_split(
        train_frame,
        "defect_type",
        "matched_source_id",
        train_module.CLASSIFY_CALIB_FRACTION,
        rng,
    )
    train = train_frame[train_frame["matched_source_id"].isin(train_ids)]
    calib = train_frame[train_frame["matched_source_id"].isin(calib_ids)]
    if len(train) == 0:
        train, calib = train_frame, train_frame.iloc[0:0]

    model = ClassifyModel(
        feature_cols=feature_cols,
        classes=CLASSIFY_CLASSES,
        seed=seed,
        lgbm_cfg=lgbm_cfg,
        class_weight=classify_cfg.get("class_weight", "balanced"),
        min_child_samples=classify_cfg.get("min_child_samples", 3),
    ).fit(
        train, train["defect_type"].to_numpy(), calib, calib["defect_type"].to_numpy()
    )
    proba = model.predict_proba(test_frame)
    pred_type, pred_conf = model.predict(test_frame)

    baseline = MajorityClassBaseline(classes=CLASSIFY_CLASSES).fit(
        train["defect_type"].to_numpy()
    )
    base_proba = baseline.predict_proba(len(test_frame))
    base_pred_type, base_pred_conf = baseline.predict(len(test_frame))

    def _frame(pt, pc, pr) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "defect_id": test_frame["matched_source_id"].to_numpy(),
                "true_class": test_frame["defect_type"].to_numpy(),
                "pred_type": np.asarray(pt),
                "pred_conf": np.asarray(pc),
            }
        )
        for c in CLASSIFY_CLASSES:
            out[c] = pr[c].to_numpy()
        return out

    return (
        train_module._classify_bootstrap_metrics(
            _frame(pred_type, pred_conf, proba), cfg, seed=seed + 40
        ),
        train_module._classify_bootstrap_metrics(
            _frame(base_pred_type, base_pred_conf, base_proba), cfg, seed=seed + 50
        ),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_ci(triple) -> str:
    if triple is None or not np.isfinite(triple[0]):
        return "n/a"
    point, lo, hi = triple
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    if N_LINES < 4 or N_LINES % 2 != 0:
        raise SystemExit(
            f"CLEANROOM_N_LINES must be even and >= 4 (got {N_LINES}) -- E2 splits lines in half."
        )

    train_lines, test_lines = _split_lines()
    lines: list[str] = []
    lines.append(
        f"Clean-room experiment -- {N_LINES} lines/domain at production per-line density."
    )
    lines.append(
        f"E2 line split: train {train_lines} / test {test_lines} (same ids in both domains)."
    )
    print("\n".join(lines) + "\n")

    feature_cols: list[str] = []
    domain_data: dict[str, dict] = {}

    # -- E1: within-domain ceiling ------------------------------------------
    for domain in DOMAINS:
        cfg = _isolated_config(domain)
        # Identical in both domains (same FeaturesConfig) -- deliberately so:
        # a cross-domain model must never see a different feature list than it
        # was fitted on. See `_build_corpus` on why dist_to_weld_m is kept.
        feature_cols = feature_columns(cfg.base.features)
        print(
            f"[{domain}] generating + featurising {N_LINES} lines x {cfg.base.data.n_runs} runs..."
        )
        t_build = time.perf_counter()
        corpus, registry, run_line_id, survey_ids = _build_corpus(cfg, domain)
        print(f"[{domain}] built in {time.perf_counter() - t_build:.0f} s")
        print(
            f"[{domain}] corpus: {len(corpus):,} rows, {len(survey_ids)} surveys, "
            f"{len(registry)} truth sources "
            f"({int((registry['kind'] == 'defect').sum())} defect, "
            f"{int((registry['kind'] == 'interference').sum())} interference)"
        )

        split_cfg = cfg.base.model.split
        corpus = add_fold_column(
            corpus,
            "line_id",
            "chainage_m",
            block_m=split_cfg.fallback_block_m,
            n_folds=split_cfg.n_folds,
        )
        print(f"[{domain}] E1: Stage 3/4/5 within-domain evaluation...")
        t_eval = time.perf_counter()
        corpus, result, severity_frame, classify_frame, _cal = (
            train_module._evaluate_corpus(
                corpus,
                feature_cols,
                cfg,
                seed=cfg.seed + 700,
                registry=registry,
                run_line_id=run_line_id,
                survey_ids=survey_ids,
            )
        )
        # line_id for the E2 split -- carried through attach_indication_features
        # from the corpus, but mapped defensively rather than assumed.
        for frame in (severity_frame, classify_frame):
            if len(frame):
                frame["line_id"] = frame["survey_id"].map(run_line_id)
        domain_data[domain] = {
            "cfg": cfg,
            "result": result,
            "severity_frame": severity_frame,
            "classify_frame": classify_frame,
            "n_rows": len(corpus),
        }
        print(
            f"[{domain}] E1 done in {time.perf_counter() - t_eval:.0f} s -- "
            f"IF recall {_fmt_ci(result['isolation_forest']['recall_at_budget'])}"
        )

    # -- E1 report ----------------------------------------------------------
    out: list[str] = [
        "",
        "=" * 100,
        "E1 -- CEILING (within-domain, diagnostic only: NOT evidence for the campaign)",
        "=" * 100,
    ]
    for domain in DOMAINS:
        r = domain_data[domain]["result"]
        out.append(
            f"\n[{domain}]  {domain_data[domain]['n_rows']:,} rows, "
            f"{r['n_defects']} physical defects, {r['n_surveys']} surveys"
        )
        out.append("  Stage 3 detect")
        out.append(
            f"    MAD  recall @ budget      {_fmt_ci(r['mad']['recall_at_budget'])}"
        )
        out.append(
            f"    IF   recall @ budget      {_fmt_ci(r['isolation_forest']['recall_at_budget'])}"
        )
        out.append(
            f"    recall gap (IF - MAD)     {_fmt_ci(r['recall_gap_if_minus_mad'])}"
            f"   gate(>=0.15 @ CI lo): {'PASS' if r['gate_passed'] else 'FAIL'}"
        )
        out.append(
            f"    IF   false-dig rate       {_fmt_ci(r['isolation_forest']['false_dig_rate'])}"
        )
        out.append(
            f"    IF   localisation (cm)    {_fmt_ci(r['isolation_forest']['localisation_error_cm'])}"
        )
        if "severity" in r:
            out.append("  Stage 4 severity")
            out.append(
                f"    coverage                  {_fmt_ci(r['severity']['coverage'])}"
                f"   (baseline {_fmt_ci(r['severity_baseline']['coverage'])})"
                f"   gate[0.87,0.93]: {'PASS' if r.get('severity_gate_passed') else 'FAIL'}"
            )
            out.append(
                f"    MAE                       {_fmt_ci(r['severity']['mae'])}"
                f"   (baseline {_fmt_ci(r['severity_baseline']['mae'])})"
            )
        else:
            out.append(
                "  Stage 4 severity            not evaluated (too few matched indications)"
            )
        if "classify" in r:
            out.append("  Stage 5 classify (per-class recall)")
            for c in CLASSIFY_CLASSES:
                out.append(
                    f"    {c:<14}            {_fmt_ci(r['classify']['per_class_recall'].get(c))}"
                )
            out.append(
                f"    interference precision    {_fmt_ci(r['classify']['interference_precision'])}"
            )
        else:
            out.append(
                "  Stage 5 classify            not evaluated (too few matched indications)"
            )

    out.append(
        "\nNOTE: the two domains have DIFFERENT physical defect universes (different"
    )
    out.append(
        "corpora, different RNG trajectories), so these are independent CIs -- read them"
    )
    out.append(
        "via overlap, never as a paired delta. Same convention as ablation_ladder.py's"
    )
    out.append("arm 5 -> 6 comparison.")

    # -- E2: the transfer matrix --------------------------------------------
    out += [
        "",
        "=" * 100,
        "E2 -- TRANSFER (train on one domain, test on the other's held-out lines)",
        "=" * 100,
    ]

    cfg_ref = domain_data["field"]["cfg"]
    sev_feature_cols = [*feature_cols, "extent_m"]

    sev_cells: dict[tuple[str, str], object] = {}
    cls_cells: dict[tuple[str, str], object] = {}
    counts: list[str] = []
    for src in DOMAINS:
        for tgt in DOMAINS:
            sev_src = domain_data[src]["severity_frame"]
            sev_tgt = domain_data[tgt]["severity_frame"]
            cls_src = domain_data[src]["classify_frame"]
            cls_tgt = domain_data[tgt]["classify_frame"]
            sev_tr = (
                sev_src[sev_src["line_id"].isin(train_lines)]
                if len(sev_src)
                else sev_src
            )
            sev_te = (
                sev_tgt[sev_tgt["line_id"].isin(test_lines)]
                if len(sev_tgt)
                else sev_tgt
            )
            cls_tr = (
                cls_src[cls_src["line_id"].isin(train_lines)]
                if len(cls_src)
                else cls_src
            )
            cls_te = (
                cls_tgt[cls_tgt["line_id"].isin(test_lines)]
                if len(cls_tgt)
                else cls_tgt
            )

            # Distinct physical sources, not just indication rows -- the
            # bootstrap resamples by source, so THAT is what sets CI width.
            def _n_src(frame: pd.DataFrame) -> int:
                return int(frame["matched_source_id"].nunique()) if len(frame) else 0

            counts.append(
                f"  train {src:<9} -> test {tgt:<9}  "
                f"severity {len(sev_tr):>4}/{_n_src(sev_tr):>3}src train, {len(sev_te):>4}/{_n_src(sev_te):>3}src test | "
                f"classify {len(cls_tr):>4}/{_n_src(cls_tr):>3}src, {len(cls_te):>4}/{_n_src(cls_te):>3}src"
            )
            print(f"E2: train {src} -> test {tgt} ...")
            sev_cells[(src, tgt)] = _cross_domain_severity(
                sev_tr, sev_te, sev_feature_cols, cfg_ref, seed=cfg_ref.seed + 800
            )
            cls_cells[(src, tgt)] = _cross_domain_classify(
                cls_tr, cls_te, sev_feature_cols, cfg_ref, seed=cfg_ref.seed + 900
            )

    out.append(
        f"\nIndication counts per cell (train lines {train_lines} / test lines {test_lines}):"
    )
    out += counts
    out.append(
        "\nCAVEAT on the test sets: an indication only exists because the DETECTOR flagged"
    )
    out.append(
        "it, and that detector was cross-validated within its own domain across all lines"
    )
    out.append(
        "-- so a test-line indication was produced by a model that saw other folds of the"
    )
    out.append(
        "same domain. That is common-mode across every cell sharing a test domain (the two"
    )
    out.append(
        "`-> field` rows are scored on the identical indication set, likewise the two"
    )
    out.append(
        "`-> cleanroom` rows), so it cannot favour one SOURCE domain over the other, which"
    )
    out.append(
        "is the comparison being made. It does mean an absolute number here is not a"
    )
    out.append("deployment estimate.")

    out.append(
        "\n-- Stage 5 classify: per-class recall on the TEST domain's held-out lines --"
    )
    header = f"  {'train -> test':<28}" + "".join(f"{c:>22}" for c in CLASSIFY_CLASSES)
    out.append(header)
    for src in DOMAINS:
        for tgt in DOMAINS:
            cell = cls_cells[(src, tgt)]
            label = f"{src} -> {tgt}"
            if cell is None:
                out.append(f"  {label:<28}{'not fittable (too few indications)':>22}")
                continue
            metrics, _base = cell
            row = f"  {label:<28}"
            row += "".join(
                f"{_fmt_ci(metrics['per_class_recall'].get(c)):>22}"
                for c in CLASSIFY_CLASSES
            )
            out.append(row)

    out.append(f"\n  {'train -> test':<28}{'interference precision':>26}{'brier':>26}")
    for src in DOMAINS:
        for tgt in DOMAINS:
            cell = cls_cells[(src, tgt)]
            label = f"{src} -> {tgt}"
            if cell is None:
                out.append(f"  {label:<28}{'n/a':>26}{'n/a':>26}")
                continue
            metrics, _base = cell
            out.append(
                f"  {label:<28}{_fmt_ci(metrics['interference_precision']):>26}"
                f"{_fmt_ci(metrics['brier']):>26}"
            )

    out.append(
        "\n-- Stage 4 severity: conformal coverage / MAE on the TEST domain's held-out lines --"
    )
    # interval_width is reported ALONGSIDE coverage deliberately: coverage on
    # its own is buyable by simply widening the interval, so a cell with high
    # coverage AND worse MAE is ambiguous (better-calibrated, or just vaguer?)
    # until the width is on the page next to it.
    out.append(
        f"  {'train -> test':<28}{'coverage':>26}{'interval width':>26}{'MAE':>26}{'baseline MAE':>26}"
    )
    for src in DOMAINS:
        for tgt in DOMAINS:
            cell = sev_cells[(src, tgt)]
            label = f"{src} -> {tgt}"
            if cell is None:
                out.append(f"  {label:<28}" + f"{'n/a':>26}" * 4)
                continue
            metrics, base = cell
            out.append(
                f"  {label:<28}{_fmt_ci(metrics['coverage']):>26}"
                f"{_fmt_ci(metrics['interval_width']):>26}"
                f"{_fmt_ci(metrics['mae']):>26}{_fmt_ci(base['mae']):>26}"
            )

    out.append("")
    out.append(
        "HOW TO READ E2: the decisive comparison is the two rows ending in `-> field`."
    )
    out.append(
        "`cleanroom -> field` is what an in-house campaign actually buys; `field -> field`"
    )
    out.append(
        "is today's status quo on the same held-out lines. If the clean-room-trained model"
    )
    out.append(
        "does not at least match the field-trained one ON FIELD DATA, the campaign does not"
    )
    out.append("pay for the model, whatever the `-> cleanroom` column says.")

    report = "\n".join(out)
    print(report)

    out_path = PROJECT_ROOT / "reports" / "cleanroom-transfer.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# The clean-room experiment: does in-house controlled data help the field model?\n\n"
        f"Measured at `CLEANROOM_N_LINES={N_LINES}` lines per domain, production per-line density "
        "(see `scripts/cleanroom_experiment.py`'s module docstring for the scale, safety and "
        "what-this-cannot-show notes).\n\n"
        f"```\n{report}\n```\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
