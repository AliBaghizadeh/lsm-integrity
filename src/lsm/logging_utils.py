"""Structured JSON logging, correlated by survey_id / run_id / pipeline_version.

An alert with no destination is not monitoring; a log with no correlation key is
not debuggable. Every log call in this project goes through get_logger() and
passes correlation fields via `extra`.
"""

from __future__ import annotations

import json
import logging
import sys
import time

# Correlation keys are named rather than discovered: these are the fields every
# query and dashboard joins on, so they must appear under a stable name even if a
# caller stops passing one.
CORRELATION_KEYS = (
    "survey_id",
    "run_id",
    "pipeline_version",
    "feature_version",
    "check_name",
    "status",
)

# Everything logging puts on a LogRecord itself. Anything outside this set arrived
# via `extra=` and is payload we want -- listing the standard names is what lets
# the formatter pass arbitrary metrics (rows/sec, durations, counts) through
# without each one needing to be added here first.
_STANDARD_RECORD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in CORRELATION_KEYS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
