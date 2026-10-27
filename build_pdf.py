"""Print HTML reports to PDF with a headless Edge or Chrome.

    python build_pdf.py reports/product_writeup.html
"""
import argparse
from pathlib import Path

from carefull.pdf import html_to_pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html", nargs="+", type=Path, help="HTML files; each PDF is written next to its HTML")
    args = parser.parse_args()
    for html in args.html:
        pdf = html_to_pdf(html)
        if pdf:
            print(f"PDF written to {pdf}")


if __name__ == "__main__":
    main()
