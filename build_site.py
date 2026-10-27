"""Export a pipeline run to the static results explorer in site/ (published with GitHub Pages).

    python build_site.py
    python build_site.py --model gemini-3.1-flash-lite

Writes site/data.js; open site/index.html in a browser to check it locally.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from carefull.config import DATA_DIR, OUTPUT_DIR, ROOT_DIR
from carefull.data import load_labels, load_profiles
from carefull.pipeline import run_dir_for

SITE_DIR = ROOT_DIR / "site"
MONTHLY_VOLUME = 50_000


def _clean(value):
    """JSON-safe scalar: NaN/NaT become None, numpy types become Python types."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _round(value, digits=2):
    value = _clean(value)
    return round(value, digits) if isinstance(value, float) else value


def transaction_record(r) -> dict:
    split = lambda s: [x for x in str(s).split(" | ") if x] if _clean(s) else []
    return {
        "id": r["transaction_id"],
        "date": r["date"].strftime("%Y-%m-%dT%H:%M"),
        "amount": _round(r["amount"]),
        "type": r["type"],
        "description": r["raw_description"],
        "category": r["category"],
        "channel": r["channel"],
        "country": _clean(r["country"]) if r["is_international"] else None,
        "counterparty": _clean(r["canonical_counterparty"]),
        "counterpartyType": _clean(r["counterparty_type"]),
        "rules": {
            "mule": bool(r["mule_flag"]) and {
                "role": r["mule_role"], "partner": r["mule_partner_id"],
                "ratio": _round(r["mule_ratio"], 3), "hours": _round(r["mule_hours"], 1),
            },
            "spike": bool(_clean(r["spike_label"])) and {
                "label": r["spike_label"], "ratio": _round(r["spike_ratio"], 1),
                "rollingMean": _round(r["rolling_mean"]),
            },
            "weekly": bool(r["weekly_flag"]) and {
                "weekStart": r["week_start"].strftime("%Y-%m-%d"), "spend": _round(r["week_spend"]),
                "baseline": _round(r["week_baseline"]),
            },
        },
        "ruleLabel": r["rule_label"],
        "llm": _clean(r["llm_risk_level"]) and {
            "risk": r["llm_risk_level"], "confidence": _round(r["llm_confidence"]),
            "pattern": r["llm_pattern"], "action": r["llm_action"], "summary": r["llm_summary"],
            "evidenceFor": split(r["llm_evidence_for"]), "evidenceAgainst": split(r["llm_evidence_against"]),
        },
        "finalLabel": r["final_label"],
        "needsReview": bool(r["needs_review"]),
        "truth": r["ground_truth_label"],
        "truthType": _clean(r["alert_type"]),
    }


def build(model: str, output_dir: Path) -> dict:
    run_dir = run_dir_for(output_dir, model)
    pred = pd.read_csv(run_dir / "predictions.csv", parse_dates=["date", "week_start"])
    pred = pred.merge(load_labels(DATA_DIR), on="transaction_id", how="left", validate="one_to_one")
    calls = pd.read_csv(run_dir / "llm_calls.csv")
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(run_dir / "user_summary.csv").set_index("user_id")

    users = []
    for _, p in load_profiles(DATA_DIR).iterrows():
        txns = pred[pred["user_id"] == p["user_id"]].sort_values("date")
        s = summary.loc[p["user_id"]]
        users.append({
            "id": p["user_id"], "name": p["name"], "age": int(p["age"]), "occupation": p["occupation"],
            "income": int(p["monthly_income"]), "location": p["location"], "bio": p["bio"],
            "alerted": bool(s["alert_user"]), "nAlert": int(s["n_alert"]), "nSuspicious": int(s["n_suspicious"]),
            "patterns": [x for x in str(_clean(s["patterns"]) or "").split(";") if x],
            "amountAtRisk": _round(s["amount_at_risk"]), "topReason": _clean(s["top_reason"]) or "",
            "truthAlerted": bool((txns["ground_truth_label"] == "alert").any()),
            "transactions": [transaction_record(r) for _, r in txns.iterrows()],
        })

    positive = pred["ground_truth_label"] != "benign"
    final_pos = pred["final_label"] != "benign"
    rule_pos = pred["rule_label"] != "benign"
    return {
        "model": model,
        "runDate": meta["finished_at"][:10],
        "detection": meta["detection_config"],
        "stats": {
            "transactions": len(pred),
            "flaggedByRules": int(rule_pos.sum()),
            "confirmed": int(final_pos.sum()),
            "positives": int(positive.sum()),
            "caught": int((positive & final_pos).sum()),
            "falseAlarms": int((~positive & final_pos).sum()),
            "ruleFalsePositives": int((~positive & rule_pos).sum()),
            "usersAlerted": sum(u["alerted"] for u in users),
            "usersCorrect": sum(u["alerted"] == u["truthAlerted"] for u in users),
            "users": len(users),
            "monthlyCost": round(calls["cost_usd"].sum() / len(pred) * MONTHLY_VOLUME, 2),
            "monthlyVolume": MONTHLY_VOLUME,
        },
        "users": users,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    data = build(args.model, args.output_dir)
    SITE_DIR.mkdir(exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (SITE_DIR / "data.js").write_text(f"window.RUN = {payload};\n", encoding="utf-8")
    print(f"site/data.js written: {data['stats']['transactions']} transactions, {len(data['users'])} users")


if __name__ == "__main__":
    main()
