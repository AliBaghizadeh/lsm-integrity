"""Typer CLI -- the implementation surface. The Dagster asset graph
(dagster_defs.py) is the orchestration interface for anything beyond a single
manual invocation; both call the same functions in generate/ingest/validate.py.
"""

from __future__ import annotations

from pathlib import Path

import typer

from lsm.config import load_config
from lsm.db import connect

app = typer.Typer(add_completion=False, help="LSM integrity pipeline CLI")


@app.command()
def generate(env: str = "dev") -> None:
    """Generate synthetic surveys -> data/raw/*.parquet."""
    from lsm.generate import generate_all

    cfg = load_config(env)
    raw_dir = Path(cfg.env.storage.raw_dir)
    results = generate_all(cfg.base.data, raw_dir, cfg.seed)
    typer.echo(f"Wrote {len(results)} surveys to {raw_dir}")
    for r in results:
        typer.echo(
            f"  {r.survey_id}: {r.n_samples} rows, content_sha256={r.content_sha256[:12]}..."
        )


@app.command()
def ingest(env: str = "dev") -> None:
    """Raw parquet -> SQLite: register -> validate -> load readings if clean.
    Exits non-zero if any survey hard-fails (and is quarantined)."""
    from lsm.generate import load_survey_result
    from lsm.ingest import IngestConflictError
    from lsm.pipeline import run_survey_pipeline

    cfg = load_config(env)
    conn = connect(cfg.env.storage.sqlite_path)
    raw_dir = Path(cfg.env.storage.raw_dir)

    n_clean = n_quarantined = n_noop = n_conflict = 0
    for path in sorted(raw_dir.glob("line_id=*/run_id=*/survey.parquet")):
        sr = load_survey_result(path, cfg.base.data.step_m, cfg.base.data.depth_m)
        try:
            status, report = run_survey_pipeline(conn, sr, cfg)
        except IngestConflictError as exc:
            n_conflict += 1
            typer.echo(f"CONFLICT: {exc}", err=True)
            continue
        summary = report.summary()
        typer.echo(
            f"{sr.survey_id}: register={status} pass={summary['pass']} "
            f"warn={summary['warn']} fail={summary['fail']}"
        )
        if status == "noop":
            n_noop += 1
        if report.has_fail:
            n_quarantined += 1
        else:
            n_clean += 1
    typer.echo(
        f"Ingested: {n_clean} clean, {n_quarantined} quarantined, "
        f"{n_noop} no-op, {n_conflict} conflicts"
    )
    if n_conflict or n_quarantined:
        raise typer.Exit(code=1)


@app.command()
def validate(env: str = "dev") -> None:
    """Re-run the DQ gates on every registered survey. Exits non-zero on any hard fail."""
    import pandas as pd

    from lsm.validate import validate_raw_survey

    cfg = load_config(env)
    conn = connect(cfg.env.storage.sqlite_path)
    quarantine_dir = Path(cfg.env.storage.quarantine_dir)

    rows = conn.execute(
        "SELECT survey_id, line_id, step_m, chainage_start_m, chainage_end_m, "
        "content_sha256, source_uri FROM survey"
    ).fetchall()
    any_fail = False
    for survey_id, line_id, step_m, c_start, c_end, content_hash, source_uri in rows:
        df = pd.read_parquet(source_uri)
        report = validate_raw_survey(
            conn,
            survey_id,
            line_id,
            step_m,
            c_start,
            c_end,
            content_hash,
            df,
            cfg.base.validate,
            quarantine_dir=quarantine_dir,
            source_path=Path(source_uri),
        )
        summary = report.summary()
        typer.echo(
            f"{survey_id}: pass={summary['pass']} warn={summary['warn']} fail={summary['fail']}"
        )
        if report.has_fail:
            any_fail = True
            for r in report.results:
                if r.status == "fail":
                    typer.echo(
                        f"  FAIL {r.check_name}: n_affected={r.n_affected} {r.detail}"
                    )
    if any_fail:
        raise typer.Exit(code=1)


@app.command()
def features(
    env: str = "dev",
    force: bool = typer.Option(False, help="recompute even on a cache hit"),
) -> None:
    """Background removal + window/shape features -> feature store.

    Cache-keyed on content_sha256 + feature_version, so a re-run is free and a
    features.py change (with its version bump) invalidates exactly the right
    partitions. Quarantined surveys are skipped, not failed: they were already
    reported by `validate`, and repeating the failure here adds noise, not signal.
    """
    from lsm.pipeline import SurveyNotFeaturisableError, run_feature_pipeline

    cfg = load_config(env)
    conn = connect(cfg.env.storage.sqlite_path)
    rows = conn.execute(
        "SELECT survey_id, status FROM survey ORDER BY survey_id"
    ).fetchall()
    if not rows:
        typer.echo("No registered surveys. Run `lsm ingest` first.", err=True)
        raise typer.Exit(code=1)

    n_computed = n_hit = n_skipped = 0
    for survey_id, status in rows:
        try:
            outcome, dir_path = run_feature_pipeline(conn, survey_id, cfg, force=force)
        except SurveyNotFeaturisableError as exc:
            n_skipped += 1
            typer.echo(f"{survey_id}: skipped ({exc})")
            continue
        n_computed += outcome == "computed"
        n_hit += outcome == "hit"
        typer.echo(f"{survey_id}: {outcome} -> {dir_path}")
    typer.echo(
        f"Features fv={cfg.base.features.version}: {n_computed} computed, "
        f"{n_hit} cache hits, {n_skipped} skipped"
    )


@app.command()
def train(env: str = "dev") -> None:
    """[Stage 3+4] MAD baseline vs IsolationForest (detection), then per-
    indication severity (LightGBM quantile + split conformal) on whatever the
    detector actually flags. Grouped CV throughout, MLflow-tracked, real
    `bundle.py` artifacts, model_run + pipeline_release rows, a real
    `indication` table, and an auto-generated model card.

    Classification and growth models are Stage 5+, not built here. Exits
    non-zero if the Stage 3 detection gate fails (severity's own gate is
    reported but does not fail the command -- there may legitimately be too
    few matched indications to evaluate it at all yet).
    """
    from lsm.db import connect
    from lsm.train import run_train

    cfg = load_config(env)
    conn = connect(cfg.env.storage.sqlite_path)
    result = run_train(cfg, conn)
    if not result["gate_passed"]:
        raise typer.Exit(code=1)


@app.command()
def predict(
    env: str = "dev", survey_id: str = typer.Argument(..., help="e.g. LINE000_R0")
) -> None:
    """[Stage 3+4] Score one already-featurised survey against the latest
    released pipeline -> `indication` rows + `indications.geojson`.

    Severity fields (`sev_pred`/`sev_lo`/`sev_hi`/`interval_nominal`) are
    filled in if a severity model has been released; NULL otherwise, not an
    error -- Stage 4 severity is conditional on Stage 3 detection.
    """
    from lsm.db import connect
    from lsm.predict import predict_survey

    cfg = load_config(env)
    conn = connect(cfg.env.storage.sqlite_path)
    indications = predict_survey(conn, survey_id, cfg)
    typer.echo(f"{survey_id}: wrote {len(indications)} indications")


@app.command()
def forecast(
    env: str = "dev",
    as_of: str = typer.Option(
        None,
        help="ISO date -- reconstruct what was knowable on this date; defaults to now",
    ),
) -> None:
    """[Stage 8] Per-defect growth -> remaining life.

    Fits a partially-pooled log-linear growth rate per physical defect
    (shrunk toward the population rate), projects each toward the
    configured limit state, and gates on beating a "no growth" baseline at
    a held-out run. Requires a released pipeline (`lsm train` first).
    """
    from lsm.db import connect
    from lsm.growth import run_forecast

    cfg = load_config(env)
    conn = connect(cfg.env.storage.sqlite_path)
    result = run_forecast(cfg, conn, as_of=as_of)
    typer.echo(
        f"population log-growth-rate: {result['growth_population_log_rate']:.4f} "
        f"(n={result['n_defects_evaluated']} defects evaluated)"
    )
    typer.echo(
        f"gate (beats no-growth baseline): {'PASSED' if result['growth_gate_passed'] else 'DID NOT PASS'}"
    )
    if not result["growth_gate_passed"]:
        raise typer.Exit(code=1)


@app.command()
def monitor(
    env: str = "dev",
    survey_id: str = typer.Argument(..., help="an already-`lsm predict`-ed survey"),
) -> None:
    """[Stage 8] Drift: PSI/KS vs bundle reference, indications/km, regime shift."""
    from lsm.db import connect
    from lsm.monitor import monitor_report_to_text, monitor_survey

    cfg = load_config(env)
    conn = connect(cfg.env.storage.sqlite_path)
    report = monitor_survey(conn, survey_id, cfg)
    typer.echo(monitor_report_to_text(report))
    if report["status"] == "block":
        raise typer.Exit(code=1)


@app.command()
def verify(
    env: str = "dev",
    indication_id: str = typer.Argument(
        ..., help="the indication being verified by an excavation"
    ),
    severity_smys: float = typer.Argument(
        ..., help="as-found severity (%SMYS) at the dig"
    ),
    defect_type: str = typer.Option(
        None, help="verified defect type, if reclassified at the dig"
    ),
) -> None:
    """[Stage 8] Record an excavation's ground truth -- writes `source=
    'excavation'` truth rows, closing the dig-feedback loop
    (validation-and-trust.md Layer 5). Not part of PLAN.md's original named
    command list, but the loop needs an entry point to close.
    """
    from lsm.db import connect
    from lsm.dig_feedback import record_excavation

    cfg = load_config(env)
    conn = connect(cfg.env.storage.sqlite_path)
    defect_id = record_excavation(
        conn, indication_id, severity_smys, verified_defect_type=defect_type
    )
    typer.echo(f"{indication_id}: verified as {defect_id}")


@app.command()
def serve() -> None:
    """[Stage 4.5] Launch the polished demo app (APP_MODE=demo|live, pre-baked
    scenarios). A thin `streamlit run` wrapper -- it blocks until Ctrl-C, same
    as running Streamlit directly. HF Spaces deploy is a separate, not-yet-built
    step (needs a cloud-account decision, see PLAN.md).

    Stage 3's own thin app/streamlit_app.py still exists, unchanged, and needs
    no CLI wrapper: run it directly with `streamlit run app/streamlit_app.py`.
    """
    import subprocess

    app_path = Path(__file__).resolve().parents[2] / "app" / "demo_app.py"
    result = subprocess.run(["streamlit", "run", str(app_path)], check=False)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


if __name__ == "__main__":
    app()
