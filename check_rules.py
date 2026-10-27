"""Run only the detection rules (no LLM, no API quota) and compare them with the ground truth.

Use it to try thresholds: change the values in .env (see .env.example) and re-run.

    python check_rules.py
    python check_rules.py --show-fp      # also list the false positives
"""
import argparse
from dataclasses import asdict

import pandas as pd

from carefull.config import DATA_DIR, DetectionConfig
from carefull.data import load_labels, load_transactions
from carefull.detection import run_detection
from carefull.enrichment import add_parsed_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show-fp", action="store_true", help="List the false positives")
    args = parser.parse_args()

    cfg = DetectionConfig()
    print("Detection config (from .env or defaults):")
    for k, v in asdict(cfg).items():
        print(f"  {k} = {v}")

    df = run_detection(add_parsed_fields(load_transactions(DATA_DIR)), cfg)
    df = df.merge(load_labels(DATA_DIR), on="transaction_id")
    positive = df["ground_truth_label"] != "benign"

    rules = {
        "money_mule": df["mule_flag"],
        "amount_spike": df["spike_label"].notna(),
        "  -> suspicious": df["spike_label"].eq("suspicious"),
        "  -> alert (escalated)": df["spike_label"].eq("alert"),
        "weekly_overspend": df["weekly_flag"],
        "any rule": df["rule_label"] != "benign",
    }
    rows = []
    for name, mask in rules.items():
        mask = mask.fillna(False).astype(bool)
        rows.append({"rule": name, "flagged": int(mask.sum()), "true_pos": int((mask & positive).sum()),
                     "false_pos": int((mask & ~positive).sum()),
                     "only_this_rule": int((mask & (df["triggered_rules"].str.count(";") == 0)).sum())
                     if not name.startswith(" ") and name != "any rule" else ""})
    print(f"\n{len(df)} transactions, {int(positive.sum())} ground-truth positives (suspicious or alert)\n")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nRule label vs. ground truth:")
    print(pd.crosstab(df["rule_label"], df["ground_truth_label"], margins=True))

    missed = df[positive & (df["rule_label"] == "benign")]
    print(f"\nPositives missed by the rules: {len(missed)}")
    if not missed.empty:
        print(missed[["transaction_id", "user_id", "date", "amount", "raw_description", "alert_type"]].to_string(index=False))

    if args.show_fp:
        fp = df[~positive & (df["rule_label"] != "benign")]
        print(f"\nFalse positives: {len(fp)}")
        print(fp[["transaction_id", "user_id", "date", "amount", "raw_description", "triggered_rules"]].to_string(index=False))


if __name__ == "__main__":
    main()
