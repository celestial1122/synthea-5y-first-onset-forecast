"""Render report/report.md to PDF: Markdown -> styled HTML -> headless Chrome print. Usage: python report/build_pdf.py [out.pdf]"""
import sys, subprocess, markdown
from pathlib import Path
HERE = Path(__file__).resolve().parent; src = HERE / "report.md"; out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.pdf"
body = markdown.markdown(src.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
css = """
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.42; color: #111; max-width: 180mm; margin: 0 auto; }
h1 { font-size: 20pt; margin: 0 0 4pt 0; } h2 { font-size: 14pt; margin: 18pt 0 6pt 0; border-bottom: 1px solid #999; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt 0; } p { margin: 5pt 0; text-align: justify; } em { color: #444; }
table { border-collapse: collapse; margin: 6pt 0 8pt 0; font-size: 9.3pt; width: 100%; } th, td { border: 1px solid #bbb; padding: 3pt 5pt; vertical-align: top; }
th { background: #eee; text-align: left; } code { font-family: Consolas, monospace; font-size: 9pt; background: #f3f3f3; padding: 0 2pt; }
pre { background: #f3f3f3; padding: 6pt; font-size: 8.8pt; white-space: pre-wrap; } ul, ol { margin: 4pt 0 4pt 18pt; } li { margin: 2pt 0; }
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><title>Five-year first-onset prediction from synthetic EHR</title><style>{css}</style></head><body>{body}</body></html>"
(HERE / "report.html").write_text(html, encoding="utf-8")
chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
subprocess.run([str(chrome), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}", (HERE / "report.html").resolve().as_uri()], check=True, timeout=120)
print("wrote", out, out.stat().st_size // 1024, "KB")
