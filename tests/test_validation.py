import re

import pytest

from carefull.config import ValidationConfig
from carefull.detection import run_detection
from carefull.enrichment import enrich_transactions
from carefull.llm import LLMProvider, Usage
from carefull.validation import Reasoning, RiskAssessment, build_prompt, decide

from .conftest import DEFAULT_RULES, extraction_json

CFG = ValidationConfig(min_confidence=0.6, baseline_days=90, recent_history_days=30,
                       max_recent_transactions=25, max_workers=1)


def assessment(risk, confidence):
    return RiskAssessment(
        reasoning=Reasoning(evidence_for_risk=[], evidence_against_risk=[], summary="s"),
        pattern="none", risk_level=risk, confidence=confidence, recommended_action="no_action")


@pytest.mark.parametrize("risk, expected", [("high", "alert"), ("medium", "suspicious"), ("low", "benign")])
def test_confident_verdicts_map_to_labels(risk, expected):
    assert decide("alert", assessment(risk, 0.9), CFG) == (expected, False, "llm")


def test_low_confidence_goes_to_human_review():
    assert decide("alert", assessment("low", 0.4), CFG) == ("suspicious", True, "low_confidence")


def test_llm_failure_keeps_the_rule_label():
    assert decide("alert", None, CFG) == ("alert", True, "rules_fallback_llm_error")


def test_rule_benign_never_reaches_the_llm():
    assert decide("benign", None, CFG) == ("benign", False, "rules")


class RegexExtractor(LLMProvider):
    """Stands in for the extraction model: strips the payment rail and keeps the name."""

    def _call(self, prompt, system, schema_model, temperature, max_tokens):
        desc = prompt.split(": ", 1)[1]
        name = re.sub(r"^(ZELLE|WIRE)( (PAYMENT|PMT|TRANSFER))?( TO)? ", "", desc).title()
        return extraction_json(name), Usage()


@pytest.fixture(scope="module")
def margaret(dataset):
    """Margaret's transactions, enriched without API calls."""
    txns, profiles, _ = dataset
    user = txns[txns["user_id"] == "USR-001"]
    enriched = enrich_transactions(user, RegexExtractor("regex"), max_workers=1)
    detected = run_detection(enriched, DEFAULT_RULES)
    profile = profiles.set_index("user_id", drop=False).loc["USR-001"].to_dict()
    return detected, profile


def test_prompt_lists_every_transfer_to_the_same_recipient(margaret):
    detected, profile = margaret
    row = detected[detected["transaction_id"] == "TXN-783823"].iloc[0]  # $2,000 to Kevin Richards
    prompt = build_prompt(row, detected, profile, CFG, DEFAULT_RULES)
    section = prompt.split("<transfers_to_same_recipient>")[1].split("</transfers_to_same_recipient>")[0]
    assert section.count("Kevin Richards (person)") == 7
    assert "$200.00" in section and "$2,000.00" in section
    assert "Lives alone since husband passed" in prompt


def test_prompt_does_not_leak_later_transactions(margaret):
    detected, profile = margaret
    row = detected[detected["transaction_id"] == "TXN-783823"].iloc[0]
    prompt = build_prompt(row, detected, profile, CFG, DEFAULT_RULES)
    assert "WIRE TRANSFER TO KEVIN RICHARDS" not in prompt  # Oct 22, a week later
