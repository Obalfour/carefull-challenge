import pytest

from carefull.enrichment import needs_entity_extraction, parse_description, resolve_alias, same_person


@pytest.mark.parametrize("desc, country, international", [
    ("PACHA IBIZA ESP", "Spain", True),
    ("MANDARIN ORIENTL BKK TH", "Thailand", True),
    ("HARRODS LTD LONDON GBR", "United Kingdom", True),
    ("KROGER #1234 COLUMBUS OH", "United States", False),
    ("MARRIOTT NAPA VALLEY CA", "United States", False),
    ("BRITISH AIRWAYS PNR XKJD92", None, False),
])
def test_parse_description_reads_trailing_location(desc, country, international):
    parsed = parse_description(desc)
    assert parsed["country"] == country
    assert parsed["is_international"] is international


def test_parse_description_drops_store_numbers():
    assert parse_description("KROGER #1234 COLUMBUS OH")["merchant"] == "Kroger Columbus"


def test_p2p_keywords():
    assert needs_entity_extraction("ZELLE PMT TO KEVIN R")
    assert needs_entity_extraction("WIRE DEPOSIT FROM SUMMIT TRADE GROUP")
    assert not needs_entity_extraction("BOB EVANS #312 COLUMBUS OH")


def test_spellings_of_the_same_person_resolve_to_the_longest():
    names = ["Kevin R", "Kevin Rich", "Kevin Richards"]
    assert {resolve_alias(n, names) for n in names} == {"Kevin Richards"}
    names = ["Dmitri Volkov", "D Volkov", "Dmitri V"]
    assert {resolve_alias(n, names) for n in names} == {"Dmitri Volkov"}


def test_relatives_with_the_same_surname_stay_separate():
    assert not same_person("Marco Martinez", "Sofia Martinez")
    assert not same_person("Isabella Martinez", "Marco Martinez")


def test_single_word_names_never_match():
    assert not same_person("Kevin", "Kevin Richards")
