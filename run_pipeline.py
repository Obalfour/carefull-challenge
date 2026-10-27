"""Run the alert pipeline for one or more models.

    python run_pipeline.py                                    # model from GOOGLE_MODEL_ID
    python run_pipeline.py --models gemini-3.1-flash-lite gemini-3.8-flash
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from carefull.config import DATA_DIR, DEFAULT_MODEL, OUTPUT_DIR
from carefull.llm import QuotaExhaustedError
from carefull.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL], help="LLM model ids to run")
    parser.add_argument("--data-dir", default=DATA_DIR, type=Path)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached LLM responses")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    for model in args.models:
        try:
            run_dir = run_pipeline(model, args.data_dir, args.output_dir, use_cache=not args.no_cache)
        except QuotaExhaustedError as err:
            raise SystemExit(f"\n{err}. No results were written for this model; the answers obtained so far "
                             f"are cached, so re-running after the daily reset resumes the run. Or pick "
                             f"another model with --models.") from None
        summary = pd.read_csv(run_dir / "user_summary.csv")
        print(f"\n=== {model} ===")
        print(summary[["user_id", "name", "alert_user", "n_alert", "n_suspicious", "patterns"]].to_string(index=False))

    print("\nNext: python evaluate.py")


if __name__ == "__main__":
    main()
