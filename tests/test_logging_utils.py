from __future__ import annotations

import json

from lsm.logging_utils import get_logger


def test_log_output_is_valid_json_with_correlation_fields(capsys):
    log = get_logger("lsm.test")
    log.info("something happened", extra={"survey_id": "LINE000_R0", "status": "accepted"})

    captured = capsys.readouterr()
    line = captured.out.strip().splitlines()[-1]
    payload = json.loads(line)  # raises if not valid JSON

    assert payload["msg"] == "something happened"
    assert payload["survey_id"] == "LINE000_R0"
    assert payload["status"] == "accepted"
    assert payload["level"] == "INFO"
    assert "ts" in payload


def test_get_logger_does_not_duplicate_handlers():
    log1 = get_logger("lsm.dup_test")
    log2 = get_logger("lsm.dup_test")
    assert log1 is log2
    assert len(log1.handlers) == 1
