"""LLM risk assessment of rule-flagged transactions, and the final decision.

The prompt only lists transactions up to the one under review. Two exceptions come from the
rules themselves: the other leg of a money-mule pair, and the weekly-overspend total, which is
computed over the whole calendar week.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from .config import DetectionConfig, ValidationConfig
from .llm import LLMProvider, QuotaExhaustedError

log = logging.getLogger(__name__)


class Reasoning(BaseModel):
    evidence_for_risk: list[str] = Field(description="Concrete facts that point to fraud, scam or exploitation")
    evidence_against_risk: list[str] = Field(
        description="Facts from the profile or history that make the activity plausible for this user")
    summary: str = Field(description="One or two sentences a caregiver could understand")


class RiskAssessment(BaseModel):
    # Reasoning comes first so the model lays out the evidence before committing to a verdict.
    reasoning: Reasoning
    pattern: Literal["escalating_transfers", "money_mule", "out_of_profile_spending", "other", "none"]
    risk_level: Literal["high", "medium", "low"]
    confidence: float = Field(ge=0.0, le=1.0, description="How sure you are of risk_level given the evidence")
    recommended_action: Literal["alert_trusted_contact", "monitor", "no_action"]


SYSTEM_PROMPT = """
You are a fraud analyst at Carefull, a financial safety service for older adults. Rule-based
detectors flagged a bank transaction. Decide whether it is genuinely concerning for THIS user.

Why it matters: a missed case can cost a vulnerable person their savings; a false alarm sent to
their adult children or caregivers erodes trust and feels patronizing. Judge each case on the
user's own profile and history, not on generic notions of "large" or "foreign".

Patterns you are looking for:
- escalating_transfers: payments to a person who is new in the user's history, with amounts
  growing over time (romance scams, undue influence, exploitation).
- money_mule: unfamiliar deposits, often from businesses the user has no relationship with,
  quickly forwarded (most of the amount) to another party.
- out_of_profile_spending: purchases inconsistent with the user's location, lifestyle and spending
  history (e.g. international luxury or nightlife on an account with only local, modest spending),
  suggesting a compromised card or account takeover.

How to judge:
- The rules are deliberately sensitive and fire on many routine transactions. A rule firing is
  not evidence by itself; look for facts that match a pattern.
- Evaluate the transaction under review. A routine purchase (usual store, usual amount) is low
  risk even if it happens in an unusual week or alongside suspicious transactions.
- If the transaction is one step of a multi-transaction pattern, it shares the pattern's risk.
- Payments to family members or long-standing recipients named in the biography or history,
  recurring bills and spending the biography explicitly anticipates (travel, generosity) lower
  the risk.

Risk levels:
- high: matches a pattern and nothing in the profile or history plausibly explains it. A trusted
  contact should be alerted.
- medium: unusual and partially consistent with a pattern, but not conclusive yet (e.g. a first
  payment to a new person). Monitor.
- low: consistent with this user's profile and history.

Confidence: 0.9+ when the evidence is clear-cut; 0.6-0.8 when it leans one way; below 0.6 when key
context is missing or the evidence conflicts. Do not inflate confidence; low-confidence cases are
sent to a human reviewer.

Answer with JSON only.
"""

_RULE_TEXT = {
    "money_mule": "money-mule pass-through",
    "amount_spike": "amount spike vs. rolling mean",
    "weekly_overspend": "weekly spending above the user's recent weekly average",
}


def _section(tag: str, body: str) -> str:
    return f"<{tag}>\n{body.strip()}\n</{tag}>"


def _fmt_line(r) -> str:
    extra = ""
    if pd.notna(r.get("canonical_counterparty")):
        extra += f" | counterparty: {r['canonical_counterparty']} ({r['counterparty_type']})"
    elif pd.notna(r.get("counterparty_type")):
        extra += f" | counterparty: unidentified ({r['counterparty_type']})"
    if r.get("is_international"):
        extra += f" | country: {r['country']}"
    return (f"{r['date']:%Y-%m-%d %H:%M} | {r['type']} | ${r['amount']:,.2f} | {r['raw_description']}"
            f" | {r['category']} | {r['channel']}{extra}")


def _fmt_profile(profile: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in profile.items())


def _fmt_transaction(r) -> str:
    lines = [
        f"transaction_id: {r['transaction_id']}",
        f"date: {r['date']:%Y-%m-%d %H:%M} ({r['date']:%A})",
        f"type: {r['type']}  amount: ${r['amount']:,.2f}",
        f"raw_description: {r['raw_description']}",
        f"category: {r['category']}  channel: {r['channel']}",
        f"parsed merchant: {r['merchant']}  country: {r['country'] if pd.notna(r['country']) else 'unknown'}"
        f"  international: {bool(r['is_international'])}",
    ]
    if pd.notna(r.get("counterparty_type")):
        lines.append(
            f"counterparty: {r['counterparty_name']} -> canonical: {r['canonical_counterparty']}"
            f"  type: {r['counterparty_type']}  extraction confidence: {r['counterparty_confidence']:.2f}")
    return "\n".join(lines)


def _days(window: str) -> int:
    return pd.Timedelta(window).days


def _fmt_triggers(r, det_cfg: DetectionConfig) -> str:
    lines = []
    if r["mule_flag"]:
        lines.append(
            f"- {_RULE_TEXT['money_mule']}: this is the {r['mule_role']} leg of a pair with "
            f"{r['mule_partner_id']}; {r['mule_ratio']:.0%} of the credit left the account "
            f"{r['mule_hours']:.1f} hours after it arrived.")
    if r["spike_label"]:
        line = (f"- {_RULE_TEXT['amount_spike']}: ${r['amount']:,.2f} is {r['spike_ratio']:.1f}x the mean of the "
                f"user's debits in the previous {_days(det_cfg.rolling_window)} days (${r['rolling_mean']:,.2f}).")
        if r["spike_label"] == "alert":
            line += f" Another spike happened in the previous {_days(det_cfg.escalation_lookback)} days."
        lines.append(line)
    if r["weekly_flag"]:
        lines.append(
            f"- {_RULE_TEXT['weekly_overspend']}: ${r['week_spend']:,.2f} spent in the week starting "
            f"{r['week_start']:%Y-%m-%d} vs. an average of ${r['week_baseline']:,.2f} per week over the "
            f"previous {det_cfg.weekly_baseline_weeks} weeks ({r['week_spend'] / r['week_baseline']:.1f}x)."
            + (" This is the user's first transaction at this merchant." if r["new_merchant"] else ""))
    return "\n".join(lines)


def _fmt_baseline(hist: pd.DataFrame) -> str:
    if hist.empty:
        return "No prior history available."
    lines = [f"Period: {hist['date'].min():%Y-%m-%d} to {hist['date'].max():%Y-%m-%d}"]

    debits = hist[hist["type"] == "debit"]
    if not debits.empty:
        lines.append("Spending by category (count, total, median, max):")
        by_cat = debits.groupby("category")["amount"].agg(["count", "sum", "median", "max"])
        for cat, s in by_cat.sort_values("sum", ascending=False).iterrows():
            lines.append(f"  {cat}: {s['count']:.0f} tx, ${s['sum']:,.2f}, median ${s['median']:,.2f}, "
                         f"max ${s['max']:,.2f}")
        top_merchants = debits["merchant"].value_counts().head(8)
        lines.append("Most frequent merchants: " + ", ".join(f"{m} ({n})" for m, n in top_merchants.items()))

    credits = hist[hist["type"] == "credit"]
    if not credits.empty:
        lines.append("Credits:")
        for desc, s in credits.groupby("raw_description")["amount"].agg(["count", "mean"]).iterrows():
            lines.append(f"  {desc}: {s['count']:.0f} x avg ${s['mean']:,.2f}")

    countries = hist.loc[hist["is_international"].astype(bool), "country"].value_counts()
    lines.append("International transactions: " + (
        ", ".join(f"{c} ({n})" for c, n in countries.items()) if not countries.empty else "none"))

    recipients = hist[hist["is_p2p_transfer"] & hist["canonical_counterparty"].notna()]
    if not recipients.empty:
        lines.append("Transfer recipients (count, total):")
        for name, s in recipients.groupby("canonical_counterparty")["amount"].agg(["count", "sum"]).iterrows():
            lines.append(f"  {name}: {s['count']:.0f} transfers, ${s['sum']:,.2f}")
    return "\n".join(lines)


def build_prompt(r, user_txns: pd.DataFrame, profile: dict, cfg: ValidationConfig,
                 det_cfg: DetectionConfig) -> str:
    now = r["date"]
    prior = user_txns[user_txns["date"] < now]
    recent_start = now - pd.Timedelta(days=cfg.recent_history_days)
    # The baseline stops where the recent window starts, so an ongoing incident does not
    # contaminate what the model sees as "established behavior".
    baseline = prior[(prior["date"] >= now - pd.Timedelta(days=cfg.baseline_days)) & (prior["date"] < recent_start)]
    recent = prior[prior["date"] >= recent_start].tail(cfg.max_recent_transactions)

    sections = [
        _section("user_profile", _fmt_profile(profile)),
        _section("transaction_under_review", _fmt_transaction(r)),
        _section("rule_triggers", _fmt_triggers(r, det_cfg)),
        _section(f"established_behavior_{cfg.baseline_days}_to_{cfg.recent_history_days}_days_before",
                 _fmt_baseline(baseline)),
        _section(f"transactions_previous_{cfg.recent_history_days}_days",
                 "\n".join(_fmt_line(x) for _, x in recent.iterrows()) or "None."),
    ]

    patterns = r["candidate_patterns"].split(";")
    if "escalating_transfers" in patterns or r["is_p2p_transfer"]:
        if pd.notna(r["canonical_counterparty"]):
            same = user_txns[(user_txns["canonical_counterparty"] == r["canonical_counterparty"])
                             & user_txns["is_p2p_transfer"] & (user_txns["date"] <= now)]
            first_seen = same["date"].min()
            body = (f"All transfers to {r['canonical_counterparty']} up to now ({len(same)}; first seen "
                    f"{first_seen:%Y-%m-%d}, user history starts {user_txns['date'].min():%Y-%m-%d}):\n")
            body += "\n".join(_fmt_line(x) for _, x in same.iterrows())
        else:
            body = "The recipient could not be identified as a person from the description."
        sections.append(_section("transfers_to_same_recipient", body))

    if r["mule_flag"]:
        partner = user_txns[user_txns["transaction_id"] == r["mule_partner_id"]]
        sections.append(_section("paired_transaction", "\n".join(_fmt_line(x) for _, x in partner.iterrows())))

    if r["weekly_flag"]:
        week = user_txns[(user_txns["week_start"] == r["week_start"]) & (user_txns["type"] == "debit")
                         & (user_txns["date"] <= now)]
        sections.append(_section("debits_this_week_so_far", "\n".join(_fmt_line(x) for _, x in week.iterrows())))

    return "\n\n".join(sections) + "\n\nAssess the transaction under review."


def validate_flagged(df: pd.DataFrame, profiles: pd.DataFrame, llm: LLMProvider,
                     cfg: ValidationConfig, det_cfg: DetectionConfig) -> dict[str, RiskAssessment | None]:
    """Run the risk assessment on every transaction with a non-benign rule label."""
    profile_by_user = profiles.set_index("user_id", drop=False).to_dict("index")
    txns_by_user = {uid: g for uid, g in df.groupby("user_id")}
    flagged = df[df["rule_label"] != "benign"]
    log.info("Validating %d flagged transactions with %s", len(flagged), llm.model_id)

    def _one(item):
        _, r = item
        prompt = build_prompt(r, txns_by_user[r["user_id"]], profile_by_user[r["user_id"]], cfg, det_cfg)
        try:
            return r["transaction_id"], llm.generate_structured(
                prompt, RiskAssessment, system=SYSTEM_PROMPT, purpose="validation")
        except QuotaExhaustedError:
            # A partial run would fall back to the rule label everywhere and skew the evaluation.
            raise
        except Exception as err:
            log.warning("Validation failed for %s: %s", r["transaction_id"], err)
            return r["transaction_id"], None

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        return dict(pool.map(_one, flagged.iterrows()))


RISK_TO_LABEL = {"high": "alert", "medium": "suspicious", "low": "benign"}


def decide(rule_label: str, assessment: RiskAssessment | None, cfg: ValidationConfig) -> tuple[str, bool, str]:
    """Final label, whether a human should review it, and where the decision came from."""
    if rule_label == "benign":
        return "benign", False, "rules"
    if assessment is None:
        # Fail safe: if the LLM is unavailable, keep the rule verdict and ask for review.
        return rule_label, True, "rules_fallback_llm_error"
    if assessment.confidence < cfg.min_confidence:
        # Uncertain verdicts are never silently dropped: they go to a human as suspicious.
        return "suspicious", True, "low_confidence"
    return RISK_TO_LABEL[assessment.risk_level], False, "llm"


def apply_decisions(df: pd.DataFrame, assessments: dict, cfg: ValidationConfig) -> pd.DataFrame:
    df = df.copy()
    cols = {k: [] for k in ["llm_risk_level", "llm_confidence", "llm_pattern", "llm_action", "llm_summary",
                            "llm_evidence_for", "llm_evidence_against", "final_label", "needs_review",
                            "decision_source"]}
    for _, r in df.iterrows():
        a = assessments.get(r["transaction_id"])
        label, review, source = decide(r["rule_label"], a, cfg)
        cols["llm_risk_level"].append(a.risk_level if a else None)
        cols["llm_confidence"].append(a.confidence if a else None)
        cols["llm_pattern"].append(a.pattern if a else None)
        cols["llm_action"].append(a.recommended_action if a else None)
        cols["llm_summary"].append(a.reasoning.summary if a else None)
        cols["llm_evidence_for"].append(" | ".join(a.reasoning.evidence_for_risk) if a else None)
        cols["llm_evidence_against"].append(" | ".join(a.reasoning.evidence_against_risk) if a else None)
        cols["final_label"].append(label)
        cols["needs_review"].append(review)
        cols["decision_source"].append(source)
    for k, v in cols.items():
        df[k] = v
    return df
