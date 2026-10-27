"""Central configuration: paths, detection thresholds, validation settings and model pricing.

Every threshold can be overridden from the environment (.env) to try variants without editing
code; the defaults below are the values used for the evaluation. See .env.example.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = ROOT_DIR / "reports"


def _env(name: str, default, cast=str):
    """Read an environment variable with a typed default (read at instantiation time)."""
    return field(default_factory=lambda: cast(os.environ[name]) if os.environ.get(name, "").strip() else default)


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "si", "sí"}


DEFAULT_MODEL = os.environ.get("GOOGLE_MODEL_ID", "gemini-3.1-flash-lite")

# Client-side requests-per-minute cap per model. The Gemini free tier allows 15 RPM;
# set LLM_MAX_RPM=0 on a paid key to remove the cap.
LLM_MAX_RPM = int(os.environ.get("LLM_MAX_RPM", "14"))

# USD per 1M tokens, standard paid tier. Thinking tokens are billed at the output price.
# Source: https://ai.google.dev/gemini-api/docs/pricing (checked October 2026).
PRICING_USD_PER_MTOK = {
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    # Introductory price until 2026-12-31; $1.50 / $7.50 from 2027-01-01.
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
}

# Extracted counterparties below this self-reported confidence are treated as unknown.
ENTITY_MIN_CONFIDENCE = float(os.environ.get("ENTITY_MIN_CONFIDENCE", "0.7"))


@dataclass(frozen=True)
class DetectionConfig:
    # Amount spike: debit > multiplier * rolling mean of the user's previous debits.
    rolling_window: str = _env("ROLLING_WINDOW", "30D")
    rolling_multiplier: float = _env("ROLLING_MULTIPLIER", 1.5, float)
    # A spike becomes an alert if the same user had another spike in this lookback.
    escalation_lookback: str = _env("ESCALATION_LOOKBACK", "7D")

    # Money mule (notebooks/data_exploration.ipynb, section 17): a credit followed by an
    # outgoing transfer of 85%-100% of its amount within 72 hours.
    mule_window_hours: int = _env("MULE_WINDOW_HOURS", 72, int)
    mule_ratio_min: float = _env("MULE_RATIO_MIN", 0.85, float)
    mule_ratio_max: float = _env("MULE_RATIO_MAX", 1.0, float)

    # Weekly overspend: weekly debit total > multiplier * mean of the previous N weeks.
    # With weekly_only_new_merchants, only the week's debits at merchants the user never used
    # before are flagged (the usual grocery store in an expensive week is not suspicious).
    weekly_baseline_weeks: int = _env("WEEKLY_BASELINE_WEEKS", 4, int)
    weekly_min_history_weeks: int = _env("WEEKLY_MIN_HISTORY_WEEKS", 2, int)
    weekly_multiplier: float = _env("WEEKLY_MULTIPLIER", 1.5, float)
    weekly_only_new_merchants: bool = _env("WEEKLY_ONLY_NEW_MERCHANTS", True, _bool)


@dataclass(frozen=True)
class ValidationConfig:
    # Below this confidence the LLM verdict is not trusted: the transaction goes to human review.
    min_confidence: float = _env("VALIDATION_MIN_CONFIDENCE", 0.6, float)
    baseline_days: int = _env("VALIDATION_BASELINE_DAYS", 90, int)
    recent_history_days: int = _env("VALIDATION_RECENT_DAYS", 30, int)
    max_recent_transactions: int = _env("VALIDATION_MAX_RECENT_TRANSACTIONS", 25, int)
    max_workers: int = _env("LLM_MAX_WORKERS", 8, int)
