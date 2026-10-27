"""HTML -> PDF with a headless Chromium browser (Edge or Chrome), so the reports keep their CSS."""
import os
import shutil
import subprocess
import time
from pathlib import Path

CANDIDATES = [
    os.environ.get("PDF_BROWSER", ""),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
    shutil.which("msedge") or "",
]


def find_browser() -> str | None:
    return next((c for c in CANDIDATES if c and Path(c).exists()), None)


def html_to_pdf(html_path: Path, pdf_path: Path | None = None) -> Path | None:
    """Print an HTML file to PDF. Returns None (and keeps the HTML) if no browser is found."""
    html_path = Path(html_path).resolve()
    pdf_path = Path(pdf_path or html_path.with_suffix(".pdf")).resolve()
    browser = find_browser()
    if browser is None:
        print("No Chrome/Edge found (set PDF_BROWSER); open the HTML and print it to PDF instead.")
        return None
    started = time.time()
    subprocess.run(
        [browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", html_path.as_uri()],
        check=True, capture_output=True, timeout=120,
    )
    # On Windows the browser can hand the job to another process and return before the file is
    # written: wait until a fresh file exists and its size stops changing.
    last_size = -1
    for _ in range(60):
        if pdf_path.exists() and pdf_path.stat().st_mtime >= started - 1:
            size = pdf_path.stat().st_size
            if size > 0 and size == last_size:
                return pdf_path
            last_size = size
        time.sleep(0.5)
    raise RuntimeError(f"The browser did not write {pdf_path}")

