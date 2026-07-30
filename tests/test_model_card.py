"""Stage 4: model_card.py -- auto-generated markdown model card."""

from __future__ import annotations

from lsm.model_card import write_model_card

TRIPLE = (0.7, 0.5, 0.9)

BASE_RESULT = {
    "mad": {
        "recall_at_budget": TRIPLE, "false_dig_rate": TRIPLE,
        "interference_dig_fraction": TRIPLE, "localisation_error_m": TRIPLE, "pr_auc": 0.5,
    },
    "isolation_forest": {
        "recall_at_budget": TRIPLE, "false_dig_rate": TRIPLE,
        "interference_dig_fraction": TRIPLE, "localisation_error_m": TRIPLE, "pr_auc": 0.5,
    },
    "recall_gap_if_minus_mad": TRIPLE,
    "interference_gap_mad_minus_if": TRIPLE,
    "gate_passed": False,
    "interference_attributable": False,
    "n_defects": 12,
    "n_surveys": 3,
}

PROVENANCE = {
    "pipeline_version": "2026.07.29-abc123",
    "config_sha256": "cfgabc",
    "git_sha": "abc123",
    "data_sha256": "dataabc",
    "feature_version": 1,
    "schema_version": 2,
    "n_defects": 12,
    "n_surveys": 3,
}


def test_write_model_card_without_severity(tmp_path):
    path = tmp_path / "model_card.md"
    write_model_card(path, PROVENANCE, BASE_RESULT)

    text = path.read_text(encoding="utf-8")
    assert "synthetic demonstrator" in text
    assert "2026.07.29-abc123" in text
    assert "DID NOT PASS" in text
    assert "## Severity" not in text


def test_write_model_card_with_severity(tmp_path):
    result = dict(BASE_RESULT)
    result["severity"] = {"coverage": TRIPLE, "mae": TRIPLE, "interval_width": TRIPLE}
    result["severity_baseline"] = {"coverage": TRIPLE, "mae": TRIPLE, "interval_width": TRIPLE}
    result["severity_gate_passed"] = True
    result["n_severity_samples"] = 18

    path = tmp_path / "model_card.md"
    write_model_card(path, PROVENANCE, result)

    text = path.read_text(encoding="utf-8")
    assert "## Severity" in text
    assert "n matched, severity-labelled indications: 18" in text
    assert "gate (coverage in [0.87, 0.93]): PASSED" in text
