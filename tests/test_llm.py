import pytest

from carefull.enrichment import CounterpartyExtraction
from carefull.llm import LLMError, QuotaExhaustedError, ResponseCache, TransientLLMError

from .conftest import extraction_json


def test_valid_answer_is_parsed_and_recorded(scripted):
    llm = scripted([extraction_json("Kevin R")])
    result = llm.generate_structured("Transaction description: ZELLE TO KEVIN R", CounterpartyExtraction,
                                     purpose="extraction")
    assert result.counterparty_name == "Kevin R"
    assert len(llm.records) == 1 and llm.records[0].ok and llm.records[0].input_tokens == 100


def test_schema_error_is_sent_back_to_the_model(scripted):
    llm = scripted(['{"counterparty_name": "Kevin R"}', extraction_json("Kevin R")])
    result = llm.generate_structured("Transaction description: ZELLE TO KEVIN R", CounterpartyExtraction)
    assert result.entity_type == "person"
    assert "did not match the required JSON schema" in llm.prompts[1]
    assert llm.records[0].attempts == 2


def test_markdown_fences_are_tolerated(scripted):
    llm = scripted([f"```json\n{extraction_json('Kevin R')}\n```"])
    assert llm.generate_structured("x", CounterpartyExtraction).counterparty_name == "Kevin R"


def test_transient_errors_are_retried(scripted, monkeypatch):
    monkeypatch.setattr("carefull.llm.time.sleep", lambda s: None)
    llm = scripted([TransientLLMError("HTTP 503"), extraction_json("Kevin R")])
    assert llm.generate_structured("x", CounterpartyExtraction).counterparty_name == "Kevin R"


def test_gives_up_after_max_retries(scripted):
    llm = scripted(["not json"] * 2, max_retries=2)
    with pytest.raises(LLMError):
        llm.generate_structured("x", CounterpartyExtraction)
    assert not llm.records[-1].ok


def test_exhausted_quota_is_not_retried(scripted):
    llm = scripted([QuotaExhaustedError("daily quota")])
    with pytest.raises(QuotaExhaustedError):
        llm.generate_structured("x", CounterpartyExtraction)
    assert llm.responses == []


def test_cache_avoids_a_second_call(scripted, tmp_path):
    cache = ResponseCache(tmp_path / "cache.json")
    first = scripted([extraction_json("Kevin R")], cache=cache)
    first.generate_structured("x", CounterpartyExtraction)

    second = scripted([], cache=ResponseCache(tmp_path / "cache.json"))
    assert second.generate_structured("x", CounterpartyExtraction).counterparty_name == "Kevin R"
    assert second.prompts == [] and second.records[0].cached
