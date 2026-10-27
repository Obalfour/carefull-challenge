"""Transaction enrichment, adapted from notebooks/data_enrichment.ipynb: description parsing,
LLM counterparty extraction, per-user alias resolution and recipient history.
"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from .config import ENTITY_MIN_CONFIDENCE
from .llm import LLMProvider, QuotaExhaustedError

log = logging.getLogger(__name__)

# Keyword pre-filter: only these descriptions (plus transfers/deposits) go to the LLM.
P2P_KEYWORDS = ["ZELLE", "VENMO", "WIRE TRANSFER", "WIRE"]
COUNTERPARTY_CATEGORIES = {"transfer", "deposit"}

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
    "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
}
# Card networks append ISO alpha-2 or alpha-3 country codes. Codes that collide with US states
# (CA, DE, IN, CO...) are left out on purpose.
COUNTRY_CODES = {
    "ES": "Spain", "ESP": "Spain",
    "TH": "Thailand", "THA": "Thailand",
    "AE": "United Arab Emirates", "ARE": "United Arab Emirates",
    "GB": "United Kingdom", "GBR": "United Kingdom", "UK": "United Kingdom",
    "FR": "France", "FRA": "France",
    "IT": "Italy", "ITA": "Italy",
    "DEU": "Germany",
    "MX": "Mexico", "MEX": "Mexico",
    "JP": "Japan", "JPN": "Japan",
    "NG": "Nigeria", "NGA": "Nigeria",
    "CAN": "Canada",
}
_STORE_NUMBER = re.compile(r"\s*#\s*\d+")


def parse_description(desc: str) -> dict:
    """Normalize a raw description and read the trailing location code, if any."""
    tokens = " ".join(str(desc).upper().split()).split()
    country, region = None, None
    if len(tokens) > 1 and tokens[-1] in COUNTRY_CODES:
        country = COUNTRY_CODES[tokens.pop()]
    elif len(tokens) > 1 and tokens[-1] in US_STATES:
        country, region = "United States", tokens.pop()
    merchant = _STORE_NUMBER.sub("", " ".join(tokens)).strip().title()
    return {
        "merchant": merchant,
        "region": region,
        "country": country,
        "is_international": country is not None and country != "United States",
    }


def needs_entity_extraction(desc: str) -> bool:
    desc = str(desc).upper()
    return any(k in desc for k in P2P_KEYWORDS)



class CounterpartyExtraction(BaseModel):
    counterparty_name: str | None = Field(
        description="Normalized counterparty name in Title Case, or null if it cannot be identified")
    entity_type: Literal["person", "business", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)


EXTRACTION_SYSTEM_PROMPT = """
You are an entity extraction system for bank transaction descriptions.

Your task is to extract and normalize the counterparty of a transfer or deposit: the person or
entity that receives the money (outgoing) or sends it (incoming), and say whether it is a person.

Rules:
1. Extract only the counterparty name. Drop payment rails and verbs (ZELLE, VENMO, WIRE, ACH,
   PAYMENT, PMT, TO, FROM, TRANSFER, DEPOSIT, CREDIT).
2. Normalize the name to title case.
3. Preserve abbreviations when the full name is not available. Do not invent or complete names.
4. entity_type is "person" only for an individual. Companies, LLCs, stores, institutions and
   agencies are "business". If you cannot tell, use "unknown".
5. If the counterparty cannot be confidently identified, return null with a low confidence.
6. Return valid JSON only, without markdown.

Examples:

Input: ZELLE PAYMENT TO KEVIN R
Output: {"counterparty_name": "Kevin R", "entity_type": "person", "confidence": 0.95}

Input: VENMO PAYMENT TO MARCO MARTINEZ
Output: {"counterparty_name": "Marco Martinez", "entity_type": "person", "confidence": 0.99}

Input: ACH CREDIT NORTHWIND LOGISTICS INC
Output: {"counterparty_name": "Northwind Logistics Inc", "entity_type": "business", "confidence": 0.97}

Input: ZELLE TO WILSON PLUMBING
Output: {"counterparty_name": "Wilson Plumbing", "entity_type": "business", "confidence": 0.85}

Input: WIRE TRANSFER REF 88213
Output: {"counterparty_name": null, "entity_type": "unknown", "confidence": 0.1}
"""


def extract_counterparties(descriptions, llm: LLMProvider, max_workers: int = 8) -> dict:
    """One LLM call per unique description (repeated descriptions are free)."""
    unique = sorted(set(descriptions))

    def _one(desc):
        try:
            result = llm.generate_structured(
                f"Transaction description: {desc}", CounterpartyExtraction,
                system=EXTRACTION_SYSTEM_PROMPT, purpose="extraction",
            )
            return desc, result
        except QuotaExhaustedError:
            raise
        except Exception as err:
            log.warning("Extraction failed for %r: %s", desc, err)
            return desc, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(_one, unique))



def same_person(name1, name2, threshold=85) -> bool:
    """True if two names probably refer to the same person."""
    if pd.isna(name1) or pd.isna(name2):
        return False
    n1, n2 = name1.strip().lower(), name2.strip().lower()
    if n1 == n2:
        return True
    p1, p2 = n1.split(), n2.split()
    if len(p1) < 2 or len(p2) < 2:
        return False

    def compatible(a, b):
        return a == b or a.startswith(b) or b.startswith(a) or a[0] == b[0]

    first_match = compatible(p1[0], p2[0])
    last_match = compatible(p1[-1], p2[-1])
    return fuzz.token_sort_ratio(n1, n2) >= threshold or (first_match and last_match)


def resolve_alias(name, user_names, threshold=85):
    """Return the longest (most descriptive) spelling among the compatible aliases."""
    similar = [other for other in user_names if not pd.isna(other) and same_person(name, other, threshold)]
    if not similar:
        return name
    return max(set(similar + [name]), key=len)


def _canonical_counterparties(df: pd.DataFrame) -> pd.Series:
    canonical = pd.Series(None, index=df.index, dtype="object")
    confident = df["counterparty_name"].notna() & (df["counterparty_confidence"] >= ENTITY_MIN_CONFIDENCE)

    business = confident & (df["counterparty_type"] == "business")
    canonical[business] = df.loc[business, "counterparty_name"]

    person = confident & (df["counterparty_type"] == "person")
    for _, idx in df[person].groupby("user_id").groups.items():
        names = df.loc[idx, "counterparty_name"].unique().tolist()
        canonical[idx] = df.loc[idx, "counterparty_name"].map(lambda n: resolve_alias(n, names))
    return canonical


def _add_recipient_history(df: pd.DataFrame) -> pd.DataFrame:
    df["recipient_prior_transfers"] = pd.NA
    df["recipient_first_seen"] = pd.NaT
    df["recipient_prev_amount"] = float("nan")
    df["recipient_total_prior"] = float("nan")

    mask = df["is_p2p_transfer"] & df["canonical_counterparty"].notna()
    t = df[mask].sort_values("date")
    grp = t.groupby(["user_id", "canonical_counterparty"])
    df.loc[t.index, "recipient_prior_transfers"] = grp.cumcount()
    df.loc[t.index, "recipient_first_seen"] = grp["date"].transform("min")
    df.loc[t.index, "recipient_prev_amount"] = grp["amount"].shift()
    df.loc[t.index, "recipient_total_prior"] = grp["amount"].cumsum() - t["amount"]
    return df


def add_parsed_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based enrichment only (no LLM). Enough to run the detection rules."""
    parsed = pd.DataFrame(df["raw_description"].map(parse_description).tolist(), index=df.index)
    df = pd.concat([df, parsed], axis=1)
    has_keyword = df["raw_description"].map(needs_entity_extraction)
    df["is_p2p_transfer"] = (df["type"] == "debit") & (df["category"] == "transfer") & has_keyword
    return df


def enrich_transactions(df: pd.DataFrame, llm: LLMProvider, max_workers: int = 8) -> pd.DataFrame:
    df = add_parsed_fields(df)
    needs_extraction = df["raw_description"].map(needs_entity_extraction) | df["category"].isin(COUNTERPARTY_CATEGORIES)

    log.info("Extracting counterparties for %d transactions (%d unique descriptions)",
             needs_extraction.sum(), df.loc[needs_extraction, "raw_description"].nunique())
    extractions = extract_counterparties(df.loc[needs_extraction, "raw_description"], llm, max_workers)

    def field(desc, attr):
        result = extractions.get(desc)
        return getattr(result, attr) if result is not None else None

    df["counterparty_name"] = None
    df["counterparty_type"] = None
    df["counterparty_confidence"] = float("nan")
    rows = df.index[needs_extraction]
    df.loc[rows, "counterparty_name"] = df.loc[rows, "raw_description"].map(lambda d: field(d, "counterparty_name"))
    df.loc[rows, "counterparty_type"] = df.loc[rows, "raw_description"].map(lambda d: field(d, "entity_type"))
    df.loc[rows, "counterparty_confidence"] = df.loc[rows, "raw_description"].map(lambda d: field(d, "confidence"))

    df["canonical_counterparty"] = _canonical_counterparties(df)
    return _add_recipient_history(df)
