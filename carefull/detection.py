"""Rule-based pattern detection. High recall on purpose: the LLM validation step adds precision.

Rules
-----
money_mule      A credit followed within 72h by an outgoing transfer of 85%-100% of its amount
                (notebooks/data_exploration.ipynb, section 17). Both legs -> alert.
amount_spike    A debit above 1.5x the rolling mean of the user's previous debits -> suspicious.
                If the user had another spike in the previous 7 days -> alert.
                On an outgoing person-to-person transfer this is the "escalating transfers" pattern.
weekly_overspend  The user's weekly debit total is above 1.5x their average weekly spend over the
                previous 4 weeks -> the week's debits at merchants the user never used before
                are alerts.
"""
import pandas as pd

from .config import DetectionConfig

SEVERITY = {"benign": 0, "suspicious": 1, "alert": 2}


def max_label(*labels) -> str:
    return max((label for label in labels if label), key=SEVERITY.get, default="benign")


def detect_money_mule(df: pd.DataFrame, cfg: DetectionConfig) -> pd.DataFrame:
    """Pair each credit with the first outgoing transfer of the same user within the window."""
    credits = (
        df[df["type"] == "credit"][["transaction_id", "user_id", "date", "amount"]]
        .rename(columns={"transaction_id": "in_id", "date": "in_date", "amount": "in_amount"})
        .sort_values("in_date")
    )
    outflows = (
        df[(df["type"] == "debit") & (df["category"] == "transfer")][["transaction_id", "user_id", "date", "amount"]]
        .rename(columns={"transaction_id": "out_id", "date": "out_date", "amount": "out_amount"})
        .sort_values("out_date")
    )
    # merge_asof consumes the right key, so keep a copy of the outflow date.
    outflows["out_key"] = outflows["out_date"]
    pairs = pd.merge_asof(
        credits, outflows, left_on="in_date", right_on="out_key", by="user_id",
        direction="forward", tolerance=pd.Timedelta(hours=cfg.mule_window_hours),
    ).dropna(subset=["out_id"])

    pairs["ratio"] = pairs["out_amount"] / pairs["in_amount"]
    pairs["hours"] = (pairs["out_date"] - pairs["in_date"]).dt.total_seconds() / 3600
    return pairs[pairs["ratio"].between(cfg.mule_ratio_min, cfg.mule_ratio_max)]


def _annotate_mule(df: pd.DataFrame, pairs: pd.DataFrame) -> None:
    df["mule_flag"] = False
    df["mule_role"] = None
    df["mule_partner_id"] = None
    df["mule_ratio"] = float("nan")
    df["mule_hours"] = float("nan")
    pos = {tid: i for i, tid in zip(df.index, df["transaction_id"])}
    for _, p in pairs.iterrows():
        for tid, partner, role in ((p["in_id"], p["out_id"], "inflow"), (p["out_id"], p["in_id"], "outflow")):
            i = pos[tid]
            df.loc[i, ["mule_flag", "mule_role", "mule_partner_id", "mule_ratio", "mule_hours"]] = (
                True, role, partner, p["ratio"], p["hours"])


def detect_amount_spikes(df: pd.DataFrame, cfg: DetectionConfig) -> None:
    """Rolling mean of previous debits (current one excluded) and 7-day escalation."""
    df["rolling_mean"] = float("nan")
    df["spike_ratio"] = float("nan")
    df["spike_label"] = None
    lookback = pd.Timedelta(cfg.escalation_lookback)

    for _, g in df[df["type"] == "debit"].groupby("user_id"):
        g = g.sort_values("date")
        rolling = g.set_index("date")["amount"].rolling(cfg.rolling_window, closed="left").mean().to_numpy()
        df.loc[g.index, "rolling_mean"] = rolling
        df.loc[g.index, "spike_ratio"] = g["amount"].to_numpy() / rolling

        spike_dates = []
        for idx, amount, date, mean in zip(g.index, g["amount"], g["date"], rolling):
            if not (mean > 0 and amount > cfg.rolling_multiplier * mean):
                continue
            recent = any(date - prev <= lookback for prev in spike_dates)
            df.loc[idx, "spike_label"] = "alert" if recent else "suspicious"
            spike_dates.append(date)


def detect_weekly_overspend(df: pd.DataFrame, cfg: DetectionConfig) -> None:
    """Compare each week's debit total with the mean of the previous weeks (zero-spend weeks count)."""
    debits = df[df["type"] == "debit"]
    week_start = debits["date"].dt.to_period("W-SUN").dt.start_time
    weekly = debits.groupby([debits["user_id"], week_start.rename("week_start")])["amount"].sum()

    frames = []
    for user_id, s in weekly.groupby(level="user_id"):
        s = s.droplevel("user_id")
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="W-MON"), fill_value=0.0)
        baseline = s.rolling(cfg.weekly_baseline_weeks, min_periods=cfg.weekly_min_history_weeks).mean().shift(1)
        frames.append(pd.DataFrame({
            "user_id": user_id, "week_start": s.index,
            "week_spend": s.to_numpy(), "week_baseline": baseline.to_numpy(),
        }))
    weeks = pd.concat(frames, ignore_index=True)
    weeks["weekly_flag"] = weeks["week_spend"] > cfg.weekly_multiplier * weeks["week_baseline"]

    df["week_start"] = df["date"].dt.to_period("W-SUN").dt.start_time
    merged = df[["user_id", "week_start"]].merge(weeks, on=["user_id", "week_start"], how="left")
    merged.index = df.index
    df["week_spend"] = merged["week_spend"]
    df["week_baseline"] = merged["week_baseline"]
    # Rows are sorted by user and date, so cumcount == 0 is the first visit to that merchant.
    df["new_merchant"] = df.groupby(["user_id", "merchant"]).cumcount().eq(0)
    flag = (df["type"] == "debit") & merged["weekly_flag"].fillna(False).astype(bool)
    if cfg.weekly_only_new_merchants:
        flag &= df["new_merchant"]
    df["weekly_flag"] = flag


def run_detection(df: pd.DataFrame, cfg: DetectionConfig) -> pd.DataFrame:
    df = df.copy()
    _annotate_mule(df, detect_money_mule(df, cfg))
    detect_amount_spikes(df, cfg)
    detect_weekly_overspend(df, cfg)

    rules, patterns, labels = [], [], []
    for _, r in df.iterrows():
        row_rules, row_patterns = [], []
        if r["mule_flag"]:
            row_rules.append("money_mule")
            row_patterns.append("money_mule")
        if r["spike_label"]:
            row_rules.append(f"amount_spike:{r['spike_label']}")
            row_patterns.append("escalating_transfers" if r["is_p2p_transfer"] else "amount_spike")
        if r["weekly_flag"]:
            row_rules.append("weekly_overspend")
            row_patterns.append("out_of_profile_spending")
        rules.append(";".join(row_rules))
        patterns.append(";".join(row_patterns))
        labels.append(max_label(
            "alert" if r["mule_flag"] else None,
            r["spike_label"],
            "alert" if r["weekly_flag"] else None,
        ))
    df["triggered_rules"] = rules
    df["candidate_patterns"] = patterns
    df["rule_label"] = labels
    return df
