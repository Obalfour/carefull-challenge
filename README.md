# Carefull challenge: LLM-powered transaction alert pipeline

**Live demo: [oli.com.ar/carefull-challenge](https://oli.com.ar/carefull-challenge/)**, a results explorer for the last run: who is alerted, each transaction on a timeline, and the LLM's
reasoning behind every decision. It is a static page; nothing runs live and no API key is involved.

A pipeline that reads bank transactions of older adults, enriches them, flags suspicious activity with
rules and validates every flag with an LLM that sees the user's profile and history. It answers
**which users should be alerted, and why**.

**Deliverables:** [evaluation report (PDF)](reports/evaluation_report.pdf) ·
[product writeup (PDF)](reports/product_writeup.pdf) · results of the last run in
[`outputs/runs/gemini-3.1-flash-lite/`](outputs/runs/gemini-3.1-flash-lite/user_summary.csv).

**Result in one line:** the rules flag 70 of 291 transactions (all 32 labeled cases included) and the
LLM keeps exactly the 32, alerting the three victimized users and none of the two decoys, for about
$11 per month at 50,000 transactions. A second run from a clean checkout, without the cache, produced
the same 291 labels. These numbers come from the same data the thresholds were tuned on; see the
report's caveats.

```
transactions.csv ─┐
                  ├─► 1. Enrichment ─► 2. Rule detection ─► 3. LLM validation ─► 4. Decision ─► outputs/runs/<model>/
user_profiles.csv ┘     (parse +         (high recall)        (risk, reasoning,      (final label,
                        LLM entities)                          confidence)            human review)
ground_truth_labels.csv ───────────────────────────────────────────────────────────► evaluate.py ─► reports/
```

## LLM provider and models

- **Provider:** Google Gemini API (`google-genai` SDK), with native JSON-schema structured output.
- **Models:** `gemini-3.1-flash-lite` (evaluated) and `gemini-3.8-flash` (supported, pending: the
  free tier allows it only 20 requests per day and a run needs about 100). The same model is used for
  both LLM steps of a run (entity extraction and risk validation), so each run's cost covers the whole
  pipeline.
- Prices used for the cost projection (USD per 1M tokens, standard tier, from the
  [Gemini pricing page](https://ai.google.dev/gemini-api/docs/pricing), checked October 2026):
  `gemini-3.1-flash-lite` $0.25 in / $1.50 out; `gemini-3.8-flash` $0.75 in / $3.75 out
  (introductory price until 2026-12-31). They are in `carefull/config.py`.
- The provider layer (`carefull/llm.py`) is an abstract `LLMProvider` with one Gemini implementation;
  adding Anthropic, OpenAI or Bedrock means adding a subclass that implements `_call`.

## Setup

Requirements: Python 3.11 and a Google AI Studio API key.

```bash
conda env create -f environment.yml
```

```bash
conda activate carefull-challenge
```

Or with pip in any Python 3.11 environment:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` (it is git-ignored) and fill in `GOOGLE_API_KEY`:

```bash
cp .env.example .env
```

```
LLM_PROVIDER=google
GOOGLE_API_KEY=<your key>
GOOGLE_MODEL_ID=gemini-3.1-flash-lite
```

`GOOGLE_MODEL_ID` is the default model when `--models` is not given. Optional: `LLM_MAX_RPM`
(default 14) caps requests per minute per model to stay under the Gemini free tier (15 RPM); set it to
`0` with a paid key.

Every rule threshold and validation setting can also be set in `.env` (an empty or missing value
uses the default). `.env.example` lists them all with the defaults used in the evaluation.

## Verify it works (no API key needed)

The tests cover description parsing, alias resolution, each detection rule, the LLM retry/cache
logic (with a scripted fake model) and the final decision mapping. One test checks that the rules
still catch all 32 labeled cases with 70 flags.

```bash
python -m pytest
```

Expected: `32 passed`. Then rebuild the report from the committed run in `outputs/runs/`:

```bash
python evaluate.py
```

### Trying thresholds without the LLM

`check_rules.py` runs only the detection rules (no API calls) and prints, per rule, how many
transactions it flags, how many are true / false positives against the ground truth, and which
positives the rules miss. Change a value in `.env`, re-run, compare:

```bash
python check_rules.py
```

Add `--show-fp` to list the false positives.

## How to run

Run the pipeline. One run makes about 100 LLM calls, so on the free tier (15 requests per minute) it takes
about 8 minutes per model; models have separate quotas, so two terminals can run them in parallel.
LLM answers are cached per model in `outputs/cache/`, so re-runs are almost free:

```bash
python run_pipeline.py --models gemini-3.1-flash-lite
```

Several models can be passed at once (`--models gemini-3.1-flash-lite gemini-3.8-flash`); each gets
its own folder in `outputs/runs/` and the report compares them.

Build the evaluation report (PDF + Markdown) from every run in `outputs/runs/`:

```bash
python evaluate.py
```

The PDFs are HTML + `reports/style.css` printed by a headless Edge or Chrome (set `PDF_BROWSER` if
neither is in a standard location). The product writeup is written by hand in
`reports/product_writeup.html`; rebuild its PDF after editing it:

```bash
python build_pdf.py reports/product_writeup.html
```

Useful flags: `--no-cache` forces fresh LLM calls (needed to re-measure latency after changing nothing),
`--monthly-volume` changes the volume in the cost projection (default 50,000), `-v` turns on debug logs.

## Docker

The image runs the tests by default; the API key is passed at run time and never copied into the
image. The image has no browser, so `evaluate.py` writes the Markdown and HTML report but skips the
PDF there.

```bash
docker build -t carefull .
```

```bash
docker run --rm carefull
```

```bash
docker run --rm carefull python check_rules.py
```

```bash
docker run --rm --env-file .env -v "$(pwd)/outputs:/app/outputs" carefull python run_pipeline.py
```

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" -v "$(pwd)/reports:/app/reports" carefull python evaluate.py
```

## Live demo

`site/` is a static page (plain HTML and JavaScript, no build step) published with GitHub Pages
from the `gh-pages` branch. Regenerate its data after a new run and preview it locally:

```bash
python build_site.py
```

```bash
python -m http.server 8000 --directory site
```

Then open http://localhost:8000. To publish, push the folder to `gh-pages`:

```bash
git subtree push --prefix site origin gh-pages
```

### Outputs

| Path | Content |
|---|---|
| `outputs/runs/<model>/predictions.csv` | One row per transaction: enrichment, every rule signal, rule label, LLM verdict (risk, confidence, pattern, reasoning) and final label |
| `outputs/runs/<model>/user_summary.csv` | Per user: alert or not, number of alerts, patterns, amount at risk, main reason |
| `outputs/runs/<model>/llm_calls.csv` | One row per LLM call: purpose, tokens, latency, cost, retries, errors |
| `outputs/runs/<model>/run_meta.json` | Model, configuration and run statistics |
| `reports/evaluation_report.pdf` | **Evaluation report** (3 pages): key numbers, metrics, error analysis, model comparison, cost projection |
| `reports/product_writeup.pdf` | **Product writeup** (2 pages): approaches considered, false positives, tradeoffs, next steps |
| `reports/evaluation_report.md` | Every evaluation table in full (Markdown) |
| `reports/figures/` | Confusion matrices made with Altair (PNG for the report, HTML interactive) |

## Project layout

```
carefull/
  config.py       paths, rule thresholds, validation settings, model pricing
  data.py         CSV loading
  llm.py          provider abstraction: structured output, retries, usage tracking, disk cache
  enrichment.py   description parsing, LLM counterparty extraction, alias resolution
  detection.py    the three detection rules
  validation.py   context building, prompt, risk assessment schema, final decision
  pipeline.py     orchestration and output files
run_pipeline.py   CLI to run the pipeline
check_rules.py    CLI to test rule thresholds against the ground truth, without the LLM
evaluate.py       CLI to build the evaluation report
build_pdf.py      CLI to print an HTML report to PDF
build_site.py     exports a run to the live demo in site/
site/             the live demo (static results explorer)
tests/            pytest suite (no API calls)
Dockerfile        container that runs the tests or any of the CLIs
.env.example      every configurable setting with its default
notebooks/        data exploration and enrichment prototypes the pipeline is based on
```

## How it works

### 1. Enrichment (`enrichment.py`, from `notebooks/data_enrichment.ipynb`)

- **Description parsing (rules):** merchant name without store numbers, trailing US state or country
  code, and an `is_international` flag (`PACHA IBIZA ESP` -> Spain, international).
- **Counterparty extraction (LLM):** only for descriptions with a P2P keyword (Zelle, Venmo, wire) or in
  the `transfer`/`deposit` categories. The model returns the normalized name, an **entity type**
  (`person` / `business` / `unknown`) and a confidence. This handles the "not every name-like string is a
  person" requirement: `MERIDIAN HOLDINGS LLC` is a business, and anything below 0.7 confidence is
  treated as unidentified. One call per unique description.
- **Alias resolution:** per user, names are merged when first and last names are compatible (equal,
  prefix or same initial) or fuzzy similarity is at least 85, keeping the longest spelling:
  `KEVIN R`, `KEVIN RICH`, `KEVIN RICHARDS` -> Kevin Richards.
- **Recipient history:** prior transfers to the same recipient, first time seen, previous amount.

### 2. Rule detection (`detection.py`)

The rules are tuned for recall; precision is the LLM's job.

| Rule | Logic | Rule label |
|---|---|---|
| Money mule | A credit followed within 72 h by an outgoing transfer of 85%-100% of its amount (`notebooks/data_exploration.ipynb`, section 17). Both legs are flagged. | alert |
| Amount spike | Debit > 1.5 x the rolling mean of the user's debits in the previous 30 days (current one excluded). If the user had another spike in the previous 7 days, it escalates. On an outgoing P2P transfer this is the *escalating transfers* pattern. | suspicious, or alert if escalated |
| Weekly overspend | The user's weekly debit total is above 1.5 x their average weekly spend over the previous 4 weeks (needs at least 2 weeks of history). Only the week's debits at merchants the user never used before are flagged. | alert |

All thresholds are in `DetectionConfig` (`carefull/config.py`) and can be overridden from `.env`. On this dataset the rules flag 70 of 291
transactions, including all 32 ground-truth positives:

| Rule | Flagged | True positives | False positives |
|---|---|---|---|
| money_mule | 12 | 12 | 0 |
| amount_spike | 51 | 21 | 30 |
| weekly_overspend | 37 | 21 | 16 |
| any rule | 70 | 32 | 38 |

The first version of the weekly rule (1.0 x the average, every debit of the week) flagged 146
transactions on its own, 122 of them benign: in an expensive week it also flagged the usual grocery
store. Restricting it to first-time merchants kept every positive and cut the total from 159 to 70.

### 3. LLM validation and risk assessment (`validation.py`)

Every transaction with a non-benign rule label gets one LLM call. The prompt (XML-delimited sections)
contains:

- the **full user profile** (age, occupation, income, location, account type, biography);
- the **transaction under review** with its enrichment;
- **which rules fired and the numbers behind them** (e.g. "8.1x the 30-day mean");
- the user's **established behavior** (spending by category, frequent merchants, income sources,
  countries, known recipients) from 90 to 30 days before the transaction, so an ongoing incident does
  not become the "normal" baseline;
- the **last 30 days** of transactions;
- pattern-specific evidence: **all transfers to the same recipient** for escalating transfers, the
  **paired leg** for money mule, and the **week's debits so far** for weekly overspend.

The system prompt explains the population, the cost of each error type, the three patterns, how to
use the profile to rule out false positives, and what each risk level and confidence range mean. The
model answers with a Pydantic-validated schema; `reasoning` comes first so the model lays out the
evidence before committing to a verdict:

```json
{
  "reasoning": {"evidence_for_risk": ["..."], "evidence_against_risk": ["..."], "summary": "..."},
  "pattern": "escalating_transfers | money_mule | out_of_profile_spending | other | none",
  "risk_level": "high | medium | low",
  "confidence": 0.0,
  "recommended_action": "alert_trusted_contact | monitor | no_action"
}
```

### 4. Decision

| LLM output | Final label |
|---|---|
| `high` | alert |
| `medium` | suspicious |
| `low` | benign |
| confidence < 0.6 | suspicious + `needs_review` (uncertain cases go to a human, never silently dropped) |
| LLM error after retries | rule label + `needs_review` (fail safe) |

### Robustness

- Native structured output plus Pydantic validation; on a schema error the model is re-prompted with
  the error message.
- A client-side requests-per-minute limiter, plus retries with backoff and jitter on rate limits, 5xx
  errors and timeouts (using the server's suggested delay when it sends one); other errors fail fast.
- An exhausted daily quota stops the run instead of silently falling back to the rule labels, which
  would make the evaluation meaningless.
- Calls run concurrently (thread pool, 8 workers).
- Validated responses are cached on disk keyed by model and prompt; changing a prompt invalidates only
  the affected entries.
- The ground truth is only read by `evaluate.py` and `check_rules.py`; the pipeline never sees it.

## Evaluation

`evaluate.py` scores both stages (rules only, and rules + LLM) for every model:

- accuracy, macro F1, per-class precision / recall / F1, and binary metrics for *flagged*
  (suspicious or alert) and *alert*;
- confusion matrices (tables, plus Altair charts as PNG and interactive HTML);
- recall by ground-truth alert type and user-level accuracy (does each user get an alert?);
- error analysis: false negatives split into "missed by rules" vs. "dismissed by LLM", false
  positives by user, rule and category, and accuracy of the LLM by confidence bucket;
- model comparison: quality, latency p50/p95, tokens, cost, retries, and where the models disagree;
- cost projection for 50,000 transactions per month and its sensitivity to the rule flag rate.

## Limitations

- 291 transactions and 5 users: every number in the report has wide uncertainty, and the rule
  thresholds and the few-shot examples were designed while looking at this same data.
- The weekly-overspend rule uses the full calendar week's total, so it can only fire once the week's
  spending is known; in production it would run on a rolling 7-day window instead.
- A money-mule inflow is only flagged once the matching outflow happens (the pair defines the pattern).
- LLM confidence is self-reported, not calibrated; the report checks whether it tracks correctness.
