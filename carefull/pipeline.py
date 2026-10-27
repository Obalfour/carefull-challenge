"""Pipeline orchestration: load -> enrich -> detect -> validate -> decide -> save."""
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import DetectionConfig, ValidationConfig
from .data import load_profiles, load_transactions
from .detection import run_detection
from .enrichment import enrich_transactions
from .llm import ResponseCache, get_provider
from .validation import apply_decisions, validate_flagged

log = logging.getLogger(__name__)

PREDICTION_COLUMNS = [
    "transaction_id", "user_id", "date", "amount", "type", "raw_description", "category", "channel",
    "merchant", "country", "is_international", "is_p2p_transfer",
    "counterparty_name", "counterparty_type", "counterparty_confidence", "canonical_counterparty",
    "recipient_prior_transfers", "recipient_first_seen",
    "mule_flag", "mule_role", "mule_partner_id", "mule_ratio", "mule_hours",
    "rolling_mean", "spike_ratio", "spike_label", "week_start", "week_spend", "week_baseline", "weekly_flag",
    "triggered_rules", "candidate_patterns", "rule_label",
    "llm_risk_level", "llm_confidence", "llm_pattern", "llm_action", "llm_summary",
    "llm_evidence_for", "llm_evidence_against", "final_label", "needs_review", "decision_source",
]


def run_dir_for(output_dir: Path, model: str) -> Path:
    return output_dir / "runs" / model.replace("/", "_")


def summarize_users(pred: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Which users should be alerted, and why."""
    rows = []
    for user_id, g in pred.groupby("user_id"):
        alerts = g[g["final_label"] == "alert"]
        top = alerts.sort_values("llm_confidence", ascending=False).head(1)
        rows.append({
            "user_id": user_id,
            "name": profiles.set_index("user_id").loc[user_id, "name"],
            "alert_user": not alerts.empty,
            "n_alert": len(alerts),
            "n_suspicious": int((g["final_label"] == "suspicious").sum()),
            "n_needs_review": int(g["needs_review"].sum()),
            "patterns": ";".join(sorted(set(alerts["llm_pattern"].dropna()))),
            "amount_at_risk": round(alerts["amount"].sum(), 2),
            "top_reason": top["llm_summary"].iloc[0] if not top.empty else "",
        })
    return pd.DataFrame(rows)


def run_pipeline(model: str, data_dir: Path, output_dir: Path,
                 det_cfg: DetectionConfig | None = None,
                 val_cfg: ValidationConfig | None = None,
                 use_cache: bool = True) -> Path:
    det_cfg = det_cfg or DetectionConfig()
    val_cfg = val_cfg or ValidationConfig()
    log.info("Detection config: %s", asdict(det_cfg))
    start = time.perf_counter()
    txns = load_transactions(data_dir)
    profiles = load_profiles(data_dir)

    # One cache file per model, so runs of different models can execute in parallel.
    cache = ResponseCache(output_dir / "cache" / f"{model.replace('/', '_')}.json") if use_cache else None
    llm = get_provider(model, cache)
    try:
        enriched = enrich_transactions(txns, llm, val_cfg.max_workers)
        detected = run_detection(enriched, det_cfg)
        log.info("Rules flagged %d of %d transactions", (detected["rule_label"] != "benign").sum(), len(detected))
        assessments = validate_flagged(detected, profiles, llm, val_cfg, det_cfg)
        result = apply_decisions(detected, assessments, val_cfg)
    finally:
        if cache:
            cache.save()

    run_dir = run_dir_for(output_dir, model)
    run_dir.mkdir(parents=True, exist_ok=True)
    result[PREDICTION_COLUMNS].to_csv(run_dir / "predictions.csv", index=False)
    calls = pd.DataFrame([vars(r) for r in llm.records])
    calls.to_csv(run_dir / "llm_calls.csv", index=False)
    summarize_users(result, profiles).to_csv(run_dir / "user_summary.csv", index=False)

    meta = {
        "model": model,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "wall_time_s": round(time.perf_counter() - start, 1),
        "n_transactions": len(result),
        "n_flagged_by_rules": int((result["rule_label"] != "benign").sum()),
        "n_llm_calls": len(calls),
        "n_llm_failures": int((~calls["ok"]).sum()) if not calls.empty else 0,
        "detection_config": asdict(det_cfg),
        "validation_config": asdict(val_cfg),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Run saved to %s (%.1fs)", run_dir, meta["wall_time_s"])
    return run_dir
