import pathlib
import subprocess
import sys

import markdown

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

CSS = """
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 900px;
         margin: 40px auto; line-height: 1.55; color: #1a1a1a; font-size: 15px; }
  h1 { font-size: 26px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }
  h2 { font-size: 20px; margin-top: 34px; border-bottom: 1px solid #eee; padding-bottom: 4px; }
  h3 { font-size: 16px; margin-top: 24px; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
  th { background: #f2f2f2; }
  code { background: #f4f4f4; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }
  pre { background: #f4f4f4; padding: 12px; border-radius: 5px; overflow-x: auto; }
  pre code { background: none; padding: 0; }
  blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 14px; color: #444; }
  img { max-width: 100%; }
  hr { border: none; border-top: 1px solid #ddd; margin: 28px 0; }
  h1, h2, h3 { page-break-after: avoid; }
  table, pre, blockquote { page-break-inside: avoid; }
</style>
"""


def convert(md_path: str):
    src = pathlib.Path(md_path).resolve()
    html_body = markdown.markdown(
        src.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
    )
    html_path = src.with_suffix(".html")
    pdf_path = src.with_suffix(".pdf")
    html_path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>{CSS}</head>"
        f"<body>{html_body}</body></html>",
        encoding="utf-8",
    )

    subprocess.run(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            f"--print-to-pdf={pdf_path}",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            str(html_path),
        ],
        check=True,
    )
    html_path.unlink()
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(p)
