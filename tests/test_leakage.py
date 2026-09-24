"""
Required CI check (SKILL invariant #1 / PLAN.md Stage 3): a physical chainage
block must never be split across train and test, and the SAME block re-observed
by a different survey run of the same line must land in the identical fold --
otherwise a model can "detect" a defect only because it memorised that exact
row from a different run of the same stretch, in train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lsm.evaluate import (
    add_fold_column,
    add_fold_column_by_line,
    assign_fold,
    assign_fold_by_line,
    assign_group,
)


def test_same_group_key_always_gets_the_same_fold():
    keys = [f"LINE000::{i}" for i in range(50)]
    folds = {k: assign_fold(k, n_folds=5) for k in keys}
    for k in keys:
        assert assign_fold(k, n_folds=5) == folds[k]


def test_group_key_ignores_run_id_so_the_same_block_matches_across_runs():
    """The whole point: (line_id, block) must NOT depend on run_id, or the same
    physical defect seen in run 0 and run 1 could land in different folds.
    """
    block_run0 = assign_group("LINE000", chainage_m=150.0, block_m=100.0)
    block_run1 = assign_group("LINE000", chainage_m=150.0, block_m=100.0)
    assert block_run0 == block_run1


def test_no_group_crosses_folds_across_three_synthetic_survey_runs():
    """Build a 2000 m / 3-run synthetic corpus (matching the real generator's
    shape) and assert every (line_id, block) group is wholly contained in one
    fold, for every run.
    """
    step_m = 0.5
    length_m = 2000.0
    n = int(length_m / step_m)
    chainage = np.arange(n) * step_m

    rows = []
    for run_id in range(3):
        for c in chainage:
            rows.append({"line_id": "LINE000", "run_id": run_id, "chainage_m": c})
    df = pd.DataFrame(rows)

    out = add_fold_column(df, "line_id", "chainage_m", block_m=100.0, n_folds=5)

    # Every group_key must map to exactly one fold, regardless of run_id.
    fold_per_group = out.groupby("group_key")["fold"].nunique()
    assert (fold_per_group == 1).all(), (
        "a (line_id, block) group was split across folds"
    )

    # The same block across different runs must carry the same group_key (and
    # therefore the same fold) -- check this explicitly, not just derived.
    pivot = out.drop_duplicates(["run_id", "group_key"]).pivot(
        index="group_key", columns="run_id", values="fold"
    )
    for _, row in pivot.iterrows():
        assert row.nunique() == 1, (
            "the same block landed in different folds across runs"
        )

    # Sanity: with 20 blocks and 5 folds, more than one fold should actually be used.
    assert out["fold"].nunique() > 1


def test_folds_are_reasonably_balanced():
    """Not a leakage check per se, but a canary: if hashing degenerated to
    putting everything in one fold, CV would silently train on 100% and
    evaluate on nothing.
    """
    keys = [f"LINE000::{i}" for i in range(200)]
    folds = [assign_fold(k, n_folds=5) for k in keys]
    counts = pd.Series(folds).value_counts()
    assert counts.min() > 0
    assert counts.max() / counts.min() < 3.0


# Stage 6: whole-line holdout ("the real generalisation test", PLAN.md) -- an
# entire physical line lands in one fold, deliberately NOT wired into the
# default Stage 3-5 path (see evaluate.py's docstrings) -- only exercised by
# scale_eval.run_whole_line_cv.


def test_assign_fold_by_line_is_stable():
    lines = [f"LINE{i:03d}" for i in range(50)]
    folds = {line: assign_fold_by_line(line, n_folds=5) for line in lines}
    for line in lines:
        assert assign_fold_by_line(line, n_folds=5) == folds[line]


def test_add_fold_column_by_line_ignores_chainage_and_run_id():
    """The whole point of whole-line grouping: varying chainage_m/run_id for
    the SAME line must never change its fold -- unlike the block scheme,
    which explicitly groups by chainage.
    """
    rows = []
    for run_id in range(3):
        for chainage_m in (0.0, 500.0, 1999.5):
            rows.append(
                {"line_id": "LINE000", "run_id": run_id, "chainage_m": chainage_m}
            )
    df = pd.DataFrame(rows)

    out = add_fold_column_by_line(df, "line_id", n_folds=5)
    assert out["fold"].nunique() == 1


def test_whole_line_never_split_across_folds():
    line_ids = [f"LINE{i:03d}" for i in range(10)]
    rows = []
    for line_id in line_ids:
        for run_id in range(3):
            for chainage_m in (0.0, 100.0, 200.0, 300.0):
                rows.append(
                    {"line_id": line_id, "run_id": run_id, "chainage_m": chainage_m}
                )
    df = pd.DataFrame(rows)

    out = add_fold_column_by_line(df, "line_id", n_folds=5)
    assert (out.groupby("line_id")["fold"].nunique() == 1).all()


def test_whole_line_folds_are_reasonably_balanced():
    lines = [f"LINE{i:03d}" for i in range(40)]
    folds = [assign_fold_by_line(line, n_folds=5) for line in lines]
    counts = pd.Series(folds).value_counts()
    assert counts.min() > 0
    assert counts.max() / counts.min() < 3.0
