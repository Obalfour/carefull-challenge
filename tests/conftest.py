import json

import pandas as pd
import pytest

from carefull.config import DATA_DIR, DetectionConfig
from carefull.llm import LLMProvider, Usage

# Explicit values, so the tests do not depend on whatever thresholds are set in .env.
DEFAULT_RULES = DetectionConfig(
    rolling_window="30D", rolling_multiplier=1.5, escalation_lookback="7D",
    mule_window_hours=72, mule_ratio_min=0.85, mule_ratio_max=1.0,
    weekly_baseline_weeks=4, weekly_min_history_weeks=2, weekly_multiplier=1.5,
    weekly_only_new_merchants=True,
)


class ScriptedProvider(LLMProvider):
    """Returns pre-written responses in order; counts calls instead of hitting an API."""

    def __init__(self, responses, **kwargs):
        super().__init__("scripted-model", **kwargs)
        self.responses = list(responses)
        self.prompts = []

    def _call(self, prompt, system, schema_model, temperature, max_tokens):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, Usage(input_tokens=100, output_tokens=20)


@pytest.fixture
def scripted():
    return ScriptedProvider


def txn(tid, user, date, amount, kind, desc, category, channel="POS"):
    return {"transaction_id": tid, "user_id": user, "date": pd.Timestamp(date), "amount": amount,
            "type": kind, "raw_description": desc, "category": category, "channel": channel}


@pytest.fixture
def make_txns():
    def _make(rows):
        return pd.DataFrame(rows).sort_values(["user_id", "date"]).reset_index(drop=True)
    return _make


@pytest.fixture(scope="session")
def dataset():
    from carefull.data import load_labels, load_profiles, load_transactions
    return load_transactions(DATA_DIR), load_profiles(DATA_DIR), load_labels(DATA_DIR)


def extraction_json(name, entity_type="person", confidence=0.95):
    return json.dumps({"counterparty_name": name, "entity_type": entity_type, "confidence": confidence})
