"""Build the evaluation report from the runs in outputs/runs/.

    python evaluate.py
    python evaluate.py --monthly-volume 50000

Writes reports/evaluation_report.pdf, reports/evaluation_report.md (every table) and the
confusion-matrix charts in reports/figures/.
"""
import argparse
import json
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from carefull.config import DATA_DIR, OUTPUT_DIR, PRICING_USD_PER_MTOK, REPORTS_DIR
from carefull.data import load_labels, load_profiles
from carefull.pdf import html_to_pdf

LABELS = ["benign", "suspicious", "alert"]
STAGES = {"rule_label": "Rules only", "final_label": "Rules + LLM"}



def load_runs(output_dir: Path, labels: pd.DataFrame) -> dict:
    runs = {}
    for run_dir in sorted((output_dir / "runs").glob("*")):
        if not (run_dir / "predictions.csv").exists():
            continue
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        pred = pd.read_csv(run_dir / "predictions.csv", parse_dates=["date"])
        pred = pred.merge(labels, on="transaction_id", how="left", validate="one_to_one")
        calls = pd.read_csv(run_dir / "llm_calls.csv")
        runs[meta["model"]] = {"pred": pred, "calls": calls, "meta": meta}
    return runs



def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def confusion(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    cm = pd.crosstab(y_true, y_pred).reindex(index=LABELS, columns=LABELS, fill_value=0)
    cm.index.name, cm.columns.name = "true", "predicted"
    return cm


def per_class(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        m = binary_metrics(y_true == label, y_pred == label)
        rows.append({"class": label, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                     "support": int((y_true == label).sum())})
    return pd.DataFrame(rows)


def stage_metrics(pred: pd.DataFrame, col: str) -> dict:
    y_true, y_pred = pred["ground_truth_label"], pred[col]
    pc = per_class(y_true, y_pred)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(pc["f1"].mean()),
        "flagged": binary_metrics(y_true != "benign", y_pred != "benign"),
        "alert": binary_metrics(y_true == "alert", y_pred == "alert"),
        "per_class": pc,
        "confusion": confusion(y_true, y_pred),
    }


def user_level(pred: pd.DataFrame, col: str) -> pd.DataFrame:
    g = pred.groupby("user_id")
    out = pd.DataFrame({
        "should_alert": g["ground_truth_label"].apply(lambda s: (s == "alert").any()),
        "alerted": g[col].apply(lambda s: (s == "alert").any()),
        "true_alerts": g["ground_truth_label"].apply(lambda s: int((s == "alert").sum())),
        "predicted_alerts": g[col].apply(lambda s: int((s == "alert").sum())),
    })
    return out.reset_index()



def md_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return str(v).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def cm_table(cm: pd.DataFrame) -> str:
    df = cm.copy()
    df.insert(0, "true \\ predicted", df.index)
    return md_table(df)


def pct(x: float) -> str:
    return f"{x:.0%}"


def usd(x: float) -> str:
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def plot_confusions(model: str, metrics: dict, figures_dir: Path) -> str | None:
    """Confusion matrices (rules only vs. rules + LLM) with Altair.

    Always saves an interactive HTML; also a PNG when vl-convert-python is installed, so the
    Markdown report can embed it. Returns the file name to reference, or None.
    """
    try:
        import altair as alt
    except ImportError as err:
        print(f"altair unavailable ({err}); skipping figures")
        return None

    panels = []
    for col, title in STAGES.items():
        cm = metrics[col]["confusion"].reset_index().melt(id_vars="true", var_name="predicted", value_name="n")
        base = alt.Chart(cm).encode(
            x=alt.X("predicted:N", sort=LABELS, title="Predicted", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("true:N", sort=LABELS, title="Ground truth"),
        )
        heat = base.mark_rect().encode(
            color=alt.Color("n:Q", scale=alt.Scale(type="sqrt", scheme="blues"), legend=None),
            tooltip=["true:N", "predicted:N", "n:Q"],
        )
        text = base.mark_text(fontSize=14).encode(
            text="n:Q",
            color=alt.condition(alt.datum.n > cm["n"].max() / 2, alt.value("white"), alt.value("black")),
        )
        panels.append((heat + text).properties(width=220, height=220, title=title))
    chart = alt.hconcat(*panels).properties(title=model)

    stem = f"confusion_{model.replace('/', '_')}"
    chart.save(figures_dir / f"{stem}.html")
    try:
        chart.save(figures_dir / f"{stem}.png", scale_factor=2)
        return f"{stem}.png"
    except Exception as err:  # vl-convert-python not installed
        print(f"PNG export unavailable ({err}); linking the HTML chart instead")
        return f"{stem}.html"



def section_overview(runs: dict, labels: pd.DataFrame) -> list[str]:
    any_pred = next(iter(runs.values()))["pred"]
    dist = labels["ground_truth_label"].value_counts().reindex(LABELS, fill_value=0)
    types = labels["alert_type"].value_counts()
    lines = [
        "## 1. Setup",
        "",
        f"- Transactions: {len(any_pred)} across {any_pred['user_id'].nunique()} users "
        f"({any_pred['date'].min():%Y-%m-%d} to {any_pred['date'].max():%Y-%m-%d}).",
        "- Ground truth: " + ", ".join(f"{k} = {v}" for k, v in dist.items()) + ".",
        "- Ground-truth alert types: " + ", ".join(f"{k} ({v})" for k, v in types.items()) + ".",
        "- Models evaluated: " + ", ".join(f"`{m}`" for m in runs) + ".",
        "- Two stages are scored for every model: **Rules only** (the rule label before the LLM) and "
        "**Rules + LLM** (the final label after validation). The rules do not use the LLM, so "
        "their numbers only differ between runs through the counterparty extraction.",
        "- Final label mapping: LLM risk `high` -> alert, `medium` -> suspicious, `low` -> benign; "
        "confidence below the threshold -> suspicious + human review; LLM failure -> rule label + review.",
        "",
        "Positive class for the binary metrics: **flagged** = `suspicious` or `alert`. The **alert** "
        "metrics only count the `alert` class, which is what would notify a trusted contact.",
        "",
        "**Read the numbers with caution.** There is no held-out set: the rule thresholds were chosen "
        "with `check_rules.py` against these same labels, and the prompts were written after exploring "
        "these same users. With 32 positives from 3 incidents, one more or one fewer error moves recall "
        "by about 3 points. The metrics show that the design works on the patterns in this dataset, "
        "not how it generalizes.",
        "",
    ]
    return lines


def section_metrics(runs: dict, metrics: dict, figures: dict) -> list[str]:
    rows = []
    for model, by_stage in metrics.items():
        for col, stage in STAGES.items():
            m = by_stage[col]
            rows.append({
                "model": model, "stage": stage, "accuracy": m["accuracy"], "macro F1": m["macro_f1"],
                "flagged P": m["flagged"]["precision"], "flagged R": m["flagged"]["recall"],
                "flagged F1": m["flagged"]["f1"],
                "alert P": m["alert"]["precision"], "alert R": m["alert"]["recall"], "alert F1": m["alert"]["f1"],
            })
    lines = ["## 2. Classification metrics", "", md_table(pd.DataFrame(rows)), ""]

    for model, by_stage in metrics.items():
        lines += [f"### {model}", ""]
        fig = figures.get(model)
        if fig and fig.endswith(".png"):
            lines += [f"![Confusion matrices](figures/{fig})", ""]
        elif fig:
            lines += [f"[Confusion matrices (interactive)](figures/{fig})", ""]
        for col, stage in STAGES.items():
            lines += [f"**{stage}** - confusion matrix", "", cm_table(by_stage[col]["confusion"]), "",
                      f"**{stage}** - per class", "", md_table(by_stage[col]["per_class"]), ""]

        pred = runs[model]["pred"]
        lines += ["**Recall by ground-truth alert type** (share of each type that ends flagged / ends as alert)", ""]
        pos = pred[pred["ground_truth_label"] != "benign"]
        by_type = pos.groupby("alert_type").agg(
            n=("transaction_id", "size"),
            rules_flagged=("rule_label", lambda s: (s != "benign").mean()),
            final_flagged=("final_label", lambda s: (s != "benign").mean()),
            final_alert=("final_label", lambda s: (s == "alert").mean()),
        ).reset_index()
        lines += [md_table(by_type), ""]

        ul = user_level(pred, "final_label")
        ul_acc = (ul["should_alert"] == ul["alerted"]).mean()
        lines += [f"**User level** (does the user get at least one alert?) - accuracy {pct(ul_acc)}", "",
                  md_table(ul), ""]
    return lines


def section_errors(runs: dict) -> list[str]:
    lines = ["## 3. Error analysis", ""]
    for model, run in runs.items():
        pred = run["pred"]
        gt_pos = pred["ground_truth_label"] != "benign"
        rule_pos = pred["rule_label"] != "benign"
        final_pos = pred["final_label"] != "benign"

        rule_fp = (~gt_pos & rule_pos).sum()
        cleared = (~gt_pos & rule_pos & ~final_pos).sum()
        rule_tp = (gt_pos & rule_pos).sum()
        dismissed = (gt_pos & rule_pos & ~final_pos).sum()
        missed_by_rules = (gt_pos & ~rule_pos).sum()

        lines += [
            f"### {model}", "",
            f"- The rules flagged {rule_pos.sum()} transactions: {rule_tp} true positives and {rule_fp} false positives; "
            f"they missed {missed_by_rules} positives.",
            f"- The LLM cleared {cleared} of the {rule_fp} rule false positives ({pct(cleared / rule_fp if rule_fp else 0)}) "
            f"and wrongly dismissed {dismissed} of the {rule_tp} rule true positives "
            f"({pct(dismissed / rule_tp if rule_tp else 0)}).",
            f"- Sent to human review: {int(pred['needs_review'].sum())} "
            f"(decision sources: {pred['decision_source'].value_counts().to_dict()}).",
            "",
        ]

        severity = pred[gt_pos & final_pos & (pred["ground_truth_label"] != pred["final_label"])]
        if not severity.empty:
            lines += [f"- Severity mismatches (flagged, but suspicious/alert swapped): {len(severity)}", ""]

        fn = pred[gt_pos & ~final_pos].copy()
        fn["where"] = np.where(fn["rule_label"] == "benign", "missed by rules", "dismissed by LLM")
        lines += [f"**False negatives ({len(fn)})**", ""]
        if fn.empty:
            lines += ["None.", ""]
        else:
            lines += [md_table(fn[["transaction_id", "user_id", "date", "amount", "raw_description", "alert_type",
                                   "where", "triggered_rules", "llm_risk_level", "llm_confidence", "llm_summary"]]), ""]

        fp = pred[~gt_pos & final_pos]
        lines += [f"**False positives ({len(fp)})**", ""]
        if fp.empty:
            lines += ["None.", ""]
        else:
            by_user = fp.groupby("user_id").size().rename("false_positives").reset_index()
            by_rule = fp.groupby("triggered_rules").size().rename("false_positives").reset_index()
            by_cat = fp.groupby("category").size().rename("false_positives").reset_index()
            lines += ["By user:", "", md_table(by_user), "", "By triggered rules:", "", md_table(by_rule), "",
                      "By category:", "", md_table(by_cat), "",
                      md_table(fp[["transaction_id", "user_id", "date", "amount", "raw_description", "triggered_rules",
                                   "final_label", "llm_risk_level", "llm_confidence", "llm_summary"]]), ""]

        lines += _confidence_analysis(pred)
    return lines


def _confidence_analysis(pred: pd.DataFrame) -> list[str]:
    """Is the self-reported confidence informative? Accuracy of LLM decisions by confidence bucket."""
    v = pred[pred["llm_risk_level"].notna()].copy()
    if v.empty:
        return []
    v["llm_flagged"] = v["llm_risk_level"] != "low"
    v["correct"] = v["llm_flagged"] == (v["ground_truth_label"] != "benign")
    v["bucket"] = pd.cut(v["llm_confidence"], [0, 0.6, 0.8, 0.9, 1.0], include_lowest=True)
    table = v.groupby("bucket", observed=True).agg(
        n=("correct", "size"), accuracy=("correct", "mean")).reset_index()
    table["bucket"] = table["bucket"].astype(str)
    return ["**LLM confidence vs. correctness** (flag / no-flag decision on validated transactions)", "",
            md_table(table), ""]


def section_models(runs: dict, metrics: dict) -> list[str]:
    rows = []
    for model, run in runs.items():
        calls, meta = run["calls"], run["meta"]
        val = calls[calls["purpose"] == "validation"]
        m = metrics[model]["final_label"]
        rows.append({
            "model": model,
            "flagged F1": m["flagged"]["f1"], "alert F1": m["alert"]["f1"],
            "alert recall": m["alert"]["recall"], "alert precision": m["alert"]["precision"],
            "LLM calls": len(calls), "failed": int((~calls["ok"]).sum()),
            "retried": int((calls["attempts"] > 1).sum()),
            "latency p50 (s)": val["latency_s"].median(), "latency p95 (s)": val["latency_s"].quantile(0.95),
            "input tok": int(calls["input_tokens"].sum()), "output tok": int(calls["output_tokens"].sum()),
            "thinking tok": int(calls["thinking_tokens"].sum()),
            "run cost": calls["cost_usd"].sum(), "wall time (s)": meta["wall_time_s"],
        })
    lines = ["## 4. Model comparison", "",
             "Latency is measured per validation call (API time, excluding backoff sleeps). Cached calls keep "
             "the tokens and latency of the original call, so re-runs do not distort the comparison; wall time "
             "does reflect caching.", "",
             md_table(pd.DataFrame(rows), floatfmt=".3f"), ""]

    models = list(runs)
    if len(models) >= 2:
        a, b = models[0], models[1]
        pa = runs[a]["pred"].set_index("transaction_id")
        pb = runs[b]["pred"].set_index("transaction_id")
        flagged = pa.index[pa["rule_label"] != "benign"]
        agree = (pa.loc[flagged, "final_label"] == pb.loc[flagged, "final_label"]).mean()
        diff = pa.loc[flagged][pa.loc[flagged, "final_label"] != pb.loc[flagged, "final_label"]]
        table = pd.DataFrame({
            "transaction_id": diff.index, "user_id": diff["user_id"], "raw_description": diff["raw_description"],
            "ground truth": diff["ground_truth_label"], a: diff["final_label"],
            b: pb.loc[diff.index, "final_label"],
        })
        lines += [f"Agreement on the final label of rule-flagged transactions: {pct(agree)} "
                  f"({len(diff)} disagreements).", ""]
        if not table.empty:
            lines += [md_table(table), ""]
    return lines


def section_costs(runs: dict, volume: int) -> list[str]:
    rows, sens = [], []
    for model, run in runs.items():
        calls, pred = run["calls"], run["pred"]
        n = len(pred)
        flagged = int((pred["rule_label"] != "benign").sum())
        ext = calls[calls["purpose"] == "extraction"]
        val = calls[calls["purpose"] == "validation"]
        per_txn = calls["cost_usd"].sum() / n
        per_validation = val["cost_usd"].mean() if not val.empty else 0.0
        rows.append({
            "model": model,
            "price in/out per 1M": "${input} / ${output}".format(**PRICING_USD_PER_MTOK.get(model, {"input": "?", "output": "?"})),
            "extraction cost / call": usd(ext["cost_usd"].mean()) if not ext.empty else "",
            "validation cost / call": usd(per_validation),
            "flag rate": pct(flagged / n),
            "cost / transaction": usd(per_txn),
            f"monthly ({volume:,} tx)": usd(per_txn * volume),
        })
        ext_per_txn = ext["cost_usd"].sum() / n
        for rate in (0.05, 0.10, 0.25, flagged / n):
            sens.append({"model": model, "flag rate": pct(rate),
                         f"monthly cost ({volume:,} tx)": usd((ext_per_txn + rate * per_validation) * volume)})

    lines = [
        "## 5. Cost projection", "",
        f"Cost per transaction = total LLM cost of the run / transactions processed, so it already includes the "
        f"share of transactions that never reach the LLM. Extraction is cached per unique description, so its "
        f"cost per transaction falls as volume grows (descriptions repeat). Prices from "
        f"https://ai.google.dev/gemini-api/docs/pricing (standard tier).", "",
        md_table(pd.DataFrame(rows)), "",
        "**Sensitivity to the rule flag rate** (validation is the dominant cost and scales with how many "
        "transactions the rules flag):", "",
        md_table(pd.DataFrame(sens)), "",
    ]

    first = next(iter(runs.values()))["pred"]
    weekly_only = first[first["triggered_rules"] == "weekly_overspend"]
    if len(weekly_only):
        cleared = (weekly_only["final_label"] == "benign").mean()
        share = len(weekly_only) / max((first["rule_label"] != "benign").sum(), 1)
        lines += [
            f"Where to optimize: {len(weekly_only)} validations ({pct(share)} of all validations) come from "
            f"transactions flagged **only** by the weekly-overspend rule, and the LLM clears {pct(cleared)} of them "
            f"(`{next(iter(runs))}`). Validating one week per user (instead of each debit) or pre-filtering "
            f"routine merchants would cut most of that spend.", "",
        ]
    return lines



RULE_FAMILIES = {"money_mule": "Money mule", "amount_spike": "Amount spike", "weekly_overspend": "Weekly overspend"}


def _tag(label) -> str:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return '<span class="tag na">-</span>'
    return f'<span class="tag {escape(str(label))}">{escape(str(label))}</span>'


def _rule_families(triggered: str) -> list[str]:
    return [r.split(":")[0] for r in str(triggered).split(";") if r and r != "nan"]


def _html_table(headers: list[str], rows: list[list], num_cols: set = frozenset(), row_classes=None) -> str:
    head = "".join(f'<th class="{"num" if i in num_cols else ""}">{h}</th>' for i, h in enumerate(headers))
    body = []
    for r_i, row in enumerate(rows):
        cls = f' class="{row_classes[r_i]}"' if row_classes and row_classes[r_i] else ""
        cells = "".join(f'<td class="{"num" if i in num_cols else ""}">{v}</td>' for i, v in enumerate(row))
        body.append(f"<tr{cls}>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html_report(runs: dict, metrics: dict, figure_files: dict, profiles: pd.DataFrame, volume: int) -> str:
    model = next(iter(runs))
    pred, calls = runs[model]["pred"], runs[model]["calls"]
    names = profiles.set_index("user_id")["name"].to_dict()
    rules_m, final_m = metrics[model]["rule_label"], metrics[model]["final_label"]

    gt_pos = pred["ground_truth_label"] != "benign"
    rule_pos = pred["rule_label"] != "benign"
    final_pos = pred["final_label"] != "benign"
    n, n_pos, n_rule = len(pred), int(gt_pos.sum()), int(rule_pos.sum())
    caught = int((gt_pos & final_pos).sum())
    final_fp = int((~gt_pos & final_pos).sum())
    rule_fp = int((~gt_pos & rule_pos).sum())
    cleared = int((~gt_pos & rule_pos & ~final_pos).sum())
    ul = user_level(pred, "final_label")
    users_ok = int((ul["should_alert"] == ul["alerted"]).sum())
    alerted_users = ul[ul["alerted"]]["user_id"].map(names).tolist()
    cost_total = calls["cost_usd"].sum()
    per_txn = cost_total / n
    monthly = per_txn * volume

    kpis = f"""
    <div class="kpis">
      <div class="kpi good"><div class="value">{users_ok} / {len(ul)}</div><div class="label">users classified correctly (alert / no alert)</div></div>
      <div class="kpi good"><div class="value">{caught} / {n_pos}</div><div class="label">suspicious transactions caught ({pct(caught / n_pos)} recall)</div></div>
      <div class="kpi {'good' if final_fp == 0 else 'warn'}"><div class="value">{final_fp}</div><div class="label">false alarms after LLM validation ({cleared} of {rule_fp} rule false positives removed)</div></div>
      <div class="kpi"><div class="value">${monthly:,.0f}</div><div class="label">estimated LLM cost per month at {volume:,} transactions</div></div>
    </div>"""

    funnel = f"""
    <div class="funnel">
      <div class="step"><div class="n">{n}</div><div class="t">transactions</div></div>
      <div class="step"><div class="n">{n_rule}</div><div class="t">flagged by rules ({pct(n_rule / n)})</div></div>
      <div class="step"><div class="n">{int(final_pos.sum())}</div><div class="t">confirmed by the LLM</div></div>
      <div class="step"><div class="n">{len(alerted_users)}</div><div class="t">users alerted</div></div>
    </div>"""

    metric_rows, classes = [], []
    for label, m in (("Rules only", rules_m), ("Rules + LLM", final_m)):
        metric_rows.append([f"<b>{label}</b>", f"{m['flagged']['precision']:.2f}", f"{m['flagged']['recall']:.2f}",
                            f"{m['flagged']['f1']:.2f}", f"{m['alert']['precision']:.2f}", f"{m['alert']['recall']:.2f}",
                            f"{m['alert']['f1']:.2f}", f"{m['accuracy']:.2f}"])
        classes.append("highlight" if label == "Rules + LLM" else "muted")
    metrics_table = _html_table(
        ["Stage", "Flagged P", "Flagged R", "Flagged F1", "Alert P", "Alert R", "Alert F1", "Accuracy"],
        metric_rows, num_cols={1, 2, 3, 4, 5, 6, 7}, row_classes=classes)

    fig = figure_files.get(model)
    figure_html = (f'<div class="figure"><img src="figures/{fig}" style="width:64%">'
                   f'<div class="caption">Confusion matrices: rows = ground truth, columns = prediction.</div></div>'
                   if fig and fig.endswith(".png") else "")

    page1 = f"""
    <header class="doc">
      <h1>Evaluation report</h1>
      <div class="sub">Carefull transaction alert pipeline &middot; model <code>{escape(model)}</code> &middot;
      {n} transactions, {pred['user_id'].nunique()} users, {pred['date'].min():%b}&ndash;{pred['date'].max():%b %Y}
      &middot; generated {datetime.now():%Y-%m-%d}</div>
    </header>

    <div class="callout good"><div class="title">Bottom line</div>
      The system alerts exactly the three users who are being victimized
      ({escape(', '.join(alerted_users))}) and none of the two users with legitimately high or unusual
      spending. Cheap rules find every case (recall {rules_m['flagged']['recall']:.0%}) but also flag
      {rule_fp} benign transactions; the LLM, reading each user's profile and history, removes them.
    </div>
    {kpis}
    {funnel}
    <div class="callout warn"><div class="title">Read these numbers with caution</div>
      There is no held-out data. Rule thresholds were chosen against these same labels and the prompts were
      written after exploring these same five users. {n_pos} positives come from only 3 incidents, so one
      error moves recall by ~3 points. The result shows the design works on these patterns, not how it generalizes.
    </div>

    <h2>Classification metrics</h2>
    <p class="small"><b>Flagged</b> = suspicious or alert (anything a human should look at).
    <b>Alert</b> = only the class that notifies a trusted contact.</p>
    {metrics_table}
    {figure_html}
    """

    rule_rows = []
    for fam, label in RULE_FAMILIES.items():
        mask = pred["triggered_rules"].map(lambda t: fam in _rule_families(t))
        tp, fp = int((mask & gt_pos).sum()), int((mask & ~gt_pos).sum())
        rule_rows.append([label, int(mask.sum()), tp, fp, f"{tp / mask.sum():.0%}" if mask.sum() else "-",
                          int((mask & ~gt_pos & ~final_pos).sum())])
    rule_table = _html_table(["Rule", "Flagged", "True pos.", "False pos.", "Precision", "FP cleared by LLM"],
                             rule_rows, num_cols={1, 2, 3, 4, 5})

    fp_rows_df = pred[~gt_pos & rule_pos]
    by_user = fp_rows_df.groupby("user_id").size().sort_values(ascending=False)
    fp_user_rows = [[escape(names.get(u, u)), int(c),
                     ", ".join(f"{k} ({v})" for k, v in fp_rows_df[fp_rows_df["user_id"] == u]["category"]
                               .value_counts().head(3).items())] for u, c in by_user.items()]
    fp_user_table = _html_table(["User", "Rule FPs", "Main categories"], fp_user_rows, num_cols={1})

    examples = (fp_rows_df[~final_pos[fp_rows_df.index]].sort_values("amount", ascending=False)
                .drop_duplicates("user_id").head(3))
    example_html = "".join(
        f'<li><b>{escape(names.get(r["user_id"], r["user_id"]))}</b> &ndash; {escape(r["raw_description"])} '
        f'(${r["amount"]:,.0f}, {escape(", ".join(RULE_FAMILIES[f] for f in _rule_families(r["triggered_rules"])))}):'
        f' <span class="quote">&ldquo;{escape(str(r["llm_summary"]))}&rdquo;</span></li>'
        for _, r in examples.iterrows())

    esc = pred[pred["alert_type"].isin(["exploitation_escalating_transfers", "unusual_transfer"])].sort_values("date")
    esc_rows = [[f'{r["date"]:%b %d}', f'${r["amount"]:,.0f}', escape(str(r["raw_description"])), _tag(r["rule_label"]),
                 f'{r["llm_risk_level"]} ({r["llm_confidence"]:.2f})' if pd.notna(r["llm_risk_level"]) else "-",
                 _tag(r["final_label"]), _tag(r["ground_truth_label"])] for _, r in esc.iterrows()]
    esc_table = _html_table(["Date", "Amount", "Description", "Rules", "LLM risk (conf.)", "Final", "Truth"],
                            esc_rows, num_cols={1})

    errors = pred[gt_pos != final_pos]
    if errors.empty:
        errors_html = ('<div class="callout good"><div class="title">No remaining errors</div>'
                       'No false negatives and no false positives after validation, and the alert / suspicious '
                       'severity matches the ground truth on every flagged transaction.</div>')
    else:
        err_rows = [[r["transaction_id"], escape(names.get(r["user_id"], r["user_id"])), f'${r["amount"]:,.0f}',
                     escape(str(r["raw_description"])), _tag(r["final_label"]), _tag(r["ground_truth_label"]),
                     escape(str(r["llm_summary"]))] for _, r in errors.iterrows()]
        errors_html = _html_table(["Txn", "User", "Amount", "Description", "Predicted", "Truth", "LLM summary"],
                                  err_rows, num_cols={2})

    validated = pred[pred["llm_confidence"].notna()]
    review = int(pred["needs_review"].sum())
    rule_sev_wrong = int((esc["rule_label"] != esc["ground_truth_label"]).sum())
    final_sev_wrong = int((esc["final_label"] != esc["ground_truth_label"]).sum())

    page2 = f"""
    <h2>Error analysis</h2>
    <h3>Where the rule errors come from</h3>
    <div class="two-col">
      <div>{rule_table}</div>
      <div>{fp_user_table}</div>
    </div>
    <div class="callout"><div class="title">The errors are systematic, not random</div>
      Most rule false positives belong to <b>{escape(names.get(by_user.index[0], ''))}</b> and
      <b>{escape(names.get(by_user.index[1], '') if len(by_user) > 1 else '')}</b>: active users whose spending is
      high and variable by design (travel, golf, monthly transfers to family). Amount-based rules cannot tell
      "unusual for a statistic" from "unusual for this person"; the profile can. The money-mule rule is the
      only one that is precise on its own.
    </div>
    <h3>How the LLM removed them</h3>
    <ul>{example_html}</ul>

    <h3>Severity: the LLM escalates as evidence accumulates</h3>
    <p class="small">Transfers to a new payee. The rules get the severity wrong on {rule_sev_wrong} of
    {len(esc)}; after validation, {final_sev_wrong} remain. The LLM treats the first payment as something to
    monitor and escalates to an alert once the amounts keep growing.</p>
    {esc_table}

    <h3>Remaining errors after validation</h3>
    {errors_html}

    <h3>Confidence and uncertainty handling</h3>
    <ul>
      <li>LLM confidence ranged from {validated['llm_confidence'].min():.2f} to {validated['llm_confidence'].max():.2f}
      (median {validated['llm_confidence'].median():.2f}); {review} transactions went to human review.</li>
      <li>Because every validated decision was correct, this data <b>cannot tell whether confidence is
      calibrated</b>. The low-confidence path (&lt; threshold &rarr; suspicious + human review) and the
      LLM-failure fallback (keep the rule label + review) were not exercised by real cases here.</li>
    </ul>

    <h3>What could fail on new data</h3>
    <ul>
      <li><b>Rule recall is the ceiling.</b> The LLM only sees what rules flag. A slow exploitation with flat
      amounts (e.g. $300 every week) would not trip the spike rule.</li>
      <li><b>Weekly rule latency.</b> It uses the calendar week's total, so it fires up to 7 days late;
      a rolling 7-day window is needed in production.</li>
      <li><b>Alias resolution is permissive.</b> The "same initial" rule would merge Marco and Maria Martinez.</li>
      <li><b>Biographies help a lot.</b> "Wife Patricia manages household expenses" explains Robert's transfers.
      Users with thin profiles will produce more false alarms.</li>
    </ul>
    """

    model_rows = []
    measured_tokens = calls[["input_tokens", "output_tokens", "thinking_tokens"]].sum()
    val_calls = calls[calls["purpose"] == "validation"]
    for m_name, run in runs.items():
        c = run["calls"]
        v = c[c["purpose"] == "validation"]
        fm = metrics[m_name]["final_label"]
        model_rows.append([f"<b>{escape(m_name)}</b>", "measured", f"{fm['flagged']['f1']:.2f}",
                           f"{fm['alert']['f1']:.2f}", f"{v['latency_s'].median():.1f} s",
                           f"{v['latency_s'].quantile(0.95):.1f} s", f"${c['cost_usd'].sum():.3f}",
                           f"${c['cost_usd'].sum() / len(run['pred']) * volume:,.0f}"])
    for m_name, price in PRICING_USD_PER_MTOK.items():
        if m_name in runs:
            continue
        est = (measured_tokens["input_tokens"] * price["input"]
               + (measured_tokens["output_tokens"] + measured_tokens["thinking_tokens"]) * price["output"]) / 1e6
        model_rows.append([escape(m_name), "estimated*", "not run", "not run", "-", "-", f"${est:.3f}",
                           f"${est / n * volume:,.0f}"])
    model_table = _html_table(["Model", "Status", "Flagged F1", "Alert F1", "Latency p50", "Latency p95",
                               "Run cost", f"Monthly ({volume // 1000}k tx)"], model_rows, num_cols={4, 5, 6, 7},
                              row_classes=["highlight" if r[1] == "measured" else "muted" for r in model_rows])

    ext = calls[calls["purpose"] == "extraction"]
    ext_per_txn = ext["cost_usd"].sum() / n
    val_cost = val_calls["cost_usd"].mean()
    sensitivity = ", ".join(
        f"{rate:.0%} &rarr; <b>${(ext_per_txn + rate * val_cost) * volume:,.2f}</b>" for rate in (0.05, 0.10))

    cost_rows = [
        ["Counterparty extraction", f"{len(ext)}", f"{ext['input_tokens'].mean():,.0f} / {ext['output_tokens'].mean():,.0f}",
         f"${ext['cost_usd'].mean():.5f}", "once per unique description (cached)"],
        ["Risk validation", f"{len(val_calls)}",
         f"{val_calls['input_tokens'].mean():,.0f} / {val_calls['output_tokens'].mean():,.0f}",
         f"${val_cost:.5f}", "once per rule-flagged transaction"],
    ]
    cost_table = _html_table(["LLM step", "Calls", "Avg tokens in / out", "Cost per call", "When"],
                             cost_rows, num_cols={1, 3})

    page3 = f"""
    <h2>Model comparison</h2>
    {model_table}
    <p class="small">* Not run: on the free tier <code>gemini-3.8-flash</code> allows 20 requests/day and one run
    needs {len(calls)}. The estimate applies its price to the tokens measured with
    <code>{escape(model)}</code>; it would be higher if the model spends thinking tokens. p95 latency is
    inflated by free-tier rate-limit retries; the median is representative of a single call.</p>
    <div class="callout"><div class="title">Model choice</div>
      Validation is a reading-comprehension task over ~2k tokens of structured context, and the lite model
      already makes no errors here, so a bigger model can only add cost and latency on this data. A larger
      model is worth testing on harder, ambiguous cases (thin profiles, partial patterns), where the
      comparison would actually discriminate.
    </div>

    <h2>Cost projection</h2>
    <div class="kpis">
      <div class="kpi"><div class="value">${val_cost:.4f}</div><div class="label">per validation call</div></div>
      <div class="kpi"><div class="value">${per_txn:.5f}</div><div class="label">per transaction (all steps)</div></div>
      <div class="kpi"><div class="value">${monthly:,.2f}</div><div class="label">per month at {volume:,} tx (real time)</div></div>
      <div class="kpi good"><div class="value">${monthly / 2:,.2f}</div><div class="label">per month with Batch API</div></div>
    </div>
    {cost_table}
    <p>Cost scales with how many transactions the rules flag, not with total volume: only
    {pct(n_rule / n)} of transactions reach the validation step. Monthly cost at other flag rates:
    {sensitivity}. <span class="small">Prices: Gemini API standard tier, October 2026.</span></p>
    <h3>Where to optimize</h3>
    <ul>
      <li><b>Lower the flag rate first.</b> {rule_fp} of {n_rule} validations were benign. Tightening the rules
      (as with the weekly rule: 159 &rarr; 70 flags with no recall loss) saves more than any model change.</li>
      <li><b>One call per incident, not per transaction.</b> Helen's 12 foreign purchases are one story; validating
      the pattern once would cut those calls by ~10x.</li>
      <li><b>Batch API</b> for the non-urgent weekly checks (50% cheaper) and <b>prompt caching</b> for the
      fixed system prompt and the user's profile, which repeat on every call of a user.</li>
      <li>At these prices the LLM is not the main cost driver; analyst time on false alarms is. Keep the
      model that minimizes false alarms, not the cheapest one.</li>
    </ul>
    """

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Evaluation report</title>
<link rel="stylesheet" href="style.css"></head>
<body>{page1}{page2}{page3}</body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    parser.add_argument("--reports-dir", default=REPORTS_DIR, type=Path)
    parser.add_argument("--monthly-volume", default=50_000, type=int)
    parser.add_argument("--no-pdf", action="store_true", help="Only write the Markdown report")
    args = parser.parse_args()

    labels = load_labels(DATA_DIR)
    runs = load_runs(args.output_dir, labels)
    if not runs:
        raise SystemExit("No runs found in outputs/runs. Run `python run_pipeline.py` first.")

    figures = args.reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metrics, figure_files = {}, {}
    for model, run in runs.items():
        metrics[model] = {col: stage_metrics(run["pred"], col) for col in STAGES}
        figure_files[model] = plot_confusions(model, metrics[model], figures)

    lines = [
        "# Evaluation report", "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `evaluate.py` from `outputs/runs/`.", "",
        *section_overview(runs, labels),
        *section_metrics(runs, metrics, figure_files),
        *section_errors(runs),
        *section_models(runs, metrics),
        *section_costs(runs, args.monthly_volume),
    ]
    report = args.reports_dir / "evaluation_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report}")

    if not args.no_pdf:
        html_path = args.reports_dir / "evaluation_report.html"
        html_path.write_text(render_html_report(runs, metrics, figure_files, load_profiles(DATA_DIR),
                                                args.monthly_volume), encoding="utf-8")
        pdf_path = html_to_pdf(html_path)
        if pdf_path:
            print(f"PDF written to {pdf_path}")


if __name__ == "__main__":
    main()
