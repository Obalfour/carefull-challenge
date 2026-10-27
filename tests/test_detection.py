from dataclasses import replace

from carefull.detection import run_detection
from carefull.enrichment import add_parsed_fields

from .conftest import DEFAULT_RULES, txn


def detect(df, cfg=DEFAULT_RULES):
    return run_detection(add_parsed_fields(df), cfg).set_index("transaction_id")


def groceries(user, start_day, n, amount=60.0):
    return [txn(f"G{user}{i}", user, f"2024-07-{start_day + i:02d} 10:00", amount, "debit",
                "PUBLIX SUPER MKT #1089 TAMPA FL", "groceries") for i in range(n)]


def test_money_mule_flags_both_legs(make_txns):
    df = make_txns(groceries("U1", 1, 5) + [
        txn("IN", "U1", "2024-07-10 10:00", 4000, "credit", "ACH DEPOSIT MERIDIAN HOLDINGS LLC", "deposit", "ACH"),
        txn("OUT", "U1", "2024-07-10 18:00", 3800, "debit", "ZELLE TO DMITRI VOLKOV", "transfer", "Zelle"),
    ])
    out = detect(df)
    assert out.loc["IN", "mule_flag"] and out.loc["OUT", "mule_flag"]
    assert out.loc["IN", "mule_role"] == "inflow" and out.loc["IN", "mule_partner_id"] == "OUT"
    assert out.loc["IN", "rule_label"] == "alert"


def test_money_mule_ignores_partial_or_late_outflows(make_txns):
    df = make_txns([
        txn("IN1", "U1", "2024-07-10 10:00", 4000, "credit", "ACH DEPOSIT ACME LLC", "deposit", "ACH"),
        txn("OUT1", "U1", "2024-07-10 18:00", 2000, "debit", "ZELLE TO JOHN DOE", "transfer", "Zelle"),
        txn("IN2", "U1", "2024-07-20 10:00", 4000, "credit", "ACH DEPOSIT ACME LLC", "deposit", "ACH"),
        txn("OUT2", "U1", "2024-07-24 10:00", 3900, "debit", "ZELLE TO JOHN DOE", "transfer", "Zelle"),
    ])
    out = detect(df)
    assert not out["mule_flag"].any()


def test_amount_spike_escalates_when_another_spike_is_recent(make_txns):
    df = make_txns(groceries("U1", 1, 6) + [
        txn("T1", "U1", "2024-07-10 12:00", 300, "debit", "ZELLE TO KEVIN R", "transfer", "Zelle"),
        txn("T2", "U1", "2024-07-15 12:00", 500, "debit", "ZELLE TO KEVIN RICHARDS", "transfer", "Zelle"),
    ])
    out = detect(df)
    assert out.loc["T1", "spike_label"] == "suspicious"
    assert out.loc["T2", "spike_label"] == "alert"
    assert out.loc["T1", "candidate_patterns"].startswith("escalating_transfers")


def test_amount_spike_needs_history(make_txns):
    df = make_txns([txn("T1", "U1", "2024-07-01 12:00", 5000, "debit", "BIG PURCHASE", "shopping")])
    assert detect(df).loc["T1", "rule_label"] == "benign"


def test_weekly_overspend_only_flags_new_merchants(make_txns):
    rows = [txn(f"W{w}", "U1", f"2024-07-{1 + 7 * w:02d} 10:00", 100, "debit",
                "PUBLIX SUPER MKT #1089 TAMPA FL", "groceries") for w in range(4)]
    rows += [
        txn("USUAL", "U1", "2024-07-29 10:00", 100, "debit", "PUBLIX SUPER MKT #1089 TAMPA FL", "groceries"),
        txn("NEW", "U1", "2024-07-30 10:00", 900, "debit", "USHUAIA BEACH HTL IBIZA ES", "lodging"),
    ]
    out = detect(make_txns(rows))
    assert out.loc["NEW", "weekly_flag"]
    assert not out.loc["USUAL", "weekly_flag"]

    every_debit = detect(make_txns(rows), replace(DEFAULT_RULES, weekly_only_new_merchants=False))
    assert every_debit.loc["USUAL", "weekly_flag"]


def test_rules_catch_every_labeled_case_in_the_dataset(dataset):
    """Regression guard for the thresholds documented in the README (70 flags, 32/32 positives)."""
    txns, _, labels = dataset
    out = run_detection(add_parsed_fields(txns), DEFAULT_RULES).merge(labels, on="transaction_id")
    positive = out["ground_truth_label"] != "benign"
    flagged = out["rule_label"] != "benign"
    assert (positive & ~flagged).sum() == 0
    assert flagged.sum() == 70
