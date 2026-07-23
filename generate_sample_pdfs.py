"""One-off script: render each corpus/*.md file as a PDF into corpus_pdf/,
so the exact same content can be run through PDF text extraction instead of
markdown's plain-text read -- a clean way to compare extraction quality
without introducing any new/unvetted document content.

A few unicode punctuation marks (em dash, curly quotes) are normalized to
ASCII equivalents before rendering, since fpdf2's built-in core font here
only supports Latin-1.

Run once, then point ingest.py at the output directory:
    python generate_sample_pdfs.py
    python ingest.py corpus_pdf
"""
import pathlib

from fpdf import FPDF

CORPUS_DIR = pathlib.Path(__file__).parent / "corpus"
OUT_DIR = pathlib.Path(__file__).parent / "corpus_pdf"

_ASCII_REPLACEMENTS = {
    "—": "-", "–": "-",
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
}


def _to_ascii(text: str) -> str:
    for src, dst in _ASCII_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def render_pdf(md_path: pathlib.Path, out_path: pathlib.Path) -> None:
    text = _to_ascii(md_path.read_text(encoding="utf-8"))
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in text.splitlines():
        pdf.write(6, line + "\n")
    pdf.output(str(out_path))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for md_path in sorted(CORPUS_DIR.glob("*.md")):
        out_path = OUT_DIR / (md_path.stem + ".pdf")
        render_pdf(md_path, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
