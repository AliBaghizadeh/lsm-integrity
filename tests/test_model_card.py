"""Stage 4: model_card.py -- auto-generated markdown model card."""

from __future__ import annotations

from lsm.model_card import write_model_card

TRIPLE = (0.7, 0.5, 0.9)

BASE_RESULT = {
    "mad": {
        "recall_at_budget": TRIPLE,
        "false_dig_rate": TRIPLE,
        "interference_dig_fraction": TRIPLE,
        "localisation_error_m": TRIPLE,
        "localisation_error_cm": (70.0, 50.0, 90.0),
        "pr_auc": 0.5,
    },
    "isolation_forest": {
        "recall_at_budget": TRIPLE,
        "false_dig_rate": TRIPLE,
        "interference_dig_fraction": TRIPLE,
        "localisation_error_m": TRIPLE,
        "localisation_error_cm": (70.0, 50.0, 90.0),
        "pr_auc": 0.5,
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
    result["severity_baseline"] = {
        "coverage": TRIPLE,
        "mae": TRIPLE,
        "interval_width": TRIPLE,
    }
    result["severity_gate_passed"] = True
    result["n_severity_samples"] = 18

    path = tmp_path / "model_card.md"
    write_model_card(path, PROVENANCE, result)

    text = path.read_text(encoding="utf-8")
    assert "## Severity" in text
    assert "n matched, severity-labelled indications: 18" in text
    assert "gate (coverage in [0.87, 0.93]): PASSED" in text


PER_CLASS_RECALL = {
    "scc": (0.95, 0.90, 1.0),
    "weld": TRIPLE,
    "dent": TRIPLE,
    "corrosion": TRIPLE,
    "interference": TRIPLE,
}


def test_write_model_card_with_classify(tmp_path):
    result = dict(BASE_RESULT)
    result["classify"] = {
        "per_class_recall": PER_CLASS_RECALL,
        "interference_precision": TRIPLE,
        "brier": TRIPLE,
    }
    result["classify_baseline"] = {
        "per_class_recall": PER_CLASS_RECALL,
        "interference_precision": TRIPLE,
        "brier": TRIPLE,
    }
    result["classify_recall_gate_passed"] = True
    result["classify_shap_check"] = {
        "top_features": ["r_mag_nt", "width_m"],
        "leaked_denylist_features": [],
        "passed": True,
    }
    result["classify_gate_passed"] = True
    result["n_classify_samples"] = 24

    provenance = dict(PROVENANCE)
    provenance["consequence_proxy"] = {
        "scc": 1.0,
        "corrosion": 0.6,
        "dent": 0.5,
        "weld": 0.4,
        "interference": 0.0,
    }

    path = tmp_path / "model_card.md"
    write_model_card(path, provenance, result)

    text = path.read_text(encoding="utf-8")
    assert "## Classification" in text
    assert "scc **(protected, gate below)**" in text
    assert "n matched, classifiable indications: 24" in text
    assert "recall gate (SCC recall >= 0.90 at the CI lower bound): PASSED" in text
    assert "physics-consistency gate" in text and "PASSED" in text
    assert "overall Stage 5 gate: PASSED" in text
    assert "STATED ENGINEERING-JUDGMENT ranking" in text
    assert "- scc: 1.0" in text


def test_write_model_card_with_classify_gate_failure_shows_leaked_features(tmp_path):
    result = dict(BASE_RESULT)
    result["classify"] = {
        "per_class_recall": {"scc": (0.5, 0.2, 0.8), "weld": TRIPLE},
        "interference_precision": TRIPLE,
        "brier": TRIPLE,
    }
    result["classify_baseline"] = {
        "per_class_recall": {"scc": (0.5, 0.2, 0.8), "weld": TRIPLE},
        "interference_precision": TRIPLE,
        "brier": TRIPLE,
    }
    result["classify_recall_gate_passed"] = False
    result["classify_shap_check"] = {
        "top_features": ["chainage_m", "r_mag_nt"],
        "leaked_denylist_features": ["chainage_m"],
        "passed": False,
    }
    result["classify_gate_passed"] = False
    result["n_classify_samples"] = 10

    path = tmp_path / "model_card.md"
    write_model_card(path, PROVENANCE, result)

    text = path.read_text(encoding="utf-8")
    assert (
        "recall gate (SCC recall >= 0.90 at the CI lower bound): DID NOT PASS" in text
    )
    assert "physics-consistency gate" in text and "DID NOT PASS" in text
    assert "leaked features: ['chainage_m']" in text
    assert "overall Stage 5 gate: DID NOT PASS" in text
