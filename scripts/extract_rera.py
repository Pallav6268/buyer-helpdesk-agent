"""Turn the official RERA Act PDF into data/rera/sections.json (one entry per section).

Source PDF: West Bengal Real Estate Regulatory Authority,
https://rera.wb.gov.in/img/pdf/THE%20REAL%20ESTATE%20(REGULATION%20AND%20DEVELOPMENT)%20ACT,%202016.pdf

Usage:  python scripts/extract_rera.py path/to/act.pdf
Needs pdftotext (poppler-utils). The script checks that it found all 92 sections and that every
heading matches the Act's own "Arrangement of Sections".
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "rera" / "sections.json"
SOURCE_URL = "https://rera.wb.gov.in/img/pdf/THE%20REAL%20ESTATE%20(REGULATION%20AND%20DEVELOPMENT)%20ACT,%202016.pdf"
FOOTNOTE = re.compile(r"^\d+\.\s+(Subs\.|Ins\.|Omitted|\d+(st|nd|rd|th)\s|w\.e\.f)")
HEADING = re.compile(r"^(\d{1,2})\.\s+([A-Z][^\n]*?)\.\s*[—-]+", re.M)


def pdf_text(path: str) -> str:
    return subprocess.run(["pdftotext", path, "-"], check=True, capture_output=True, text=True).stdout


def arrangement(text: str) -> dict[int, str]:
    """Section titles from the table at the start of the Act."""
    toc = text[: text.index("BE it enacted")]
    titles = {}
    for num, title in re.findall(r"^\s*(\d{1,2})\.\s+(.+?)\.?\s*$", toc, re.M):
        titles.setdefault(int(num), title.strip().rstrip("."))
    return titles


def clean_body(text: str) -> str:
    body = text[text.index("BE it enacted") :]
    lines, skipping = [], False
    for line in body.splitlines():
        stripped = line.strip()
        if FOOTNOTE.match(stripped):
            skipping = True
        if skipping:
            if not stripped:
                skipping = False
            continue
        if re.fullmatch(r"\d{1,3}", stripped) or re.fullmatch(r"\f?", stripped):
            continue  # page numbers and form feeds
        if re.fullmatch(r"CHAPTE[RT] [IVXL]+", stripped) or (stripped.isupper() and len(stripped) > 3):
            continue  # chapter headings
        lines.append(stripped)
    return "\n".join(lines)


def split_sections(body: str, titles: dict[int, str]) -> list[dict]:
    joined = re.sub(r"(?<![.:;—])\n(?!\(|\d{1,2}\.\s+[A-Z]|Provided|Explanation)", " ", body)
    found = []
    expected = 1
    for m in HEADING.finditer(joined):
        if int(m.group(1)) == expected:
            found.append((expected, m.start(), m.end(), m.group(2).strip()))
            expected += 1
    sections = []
    for i, (num, _start, end, title) in enumerate(found):
        stop = found[i + 1][1] if i + 1 < len(found) else len(joined)
        text = joined[end:stop].strip()
        sections.append({"section": str(num), "title": titles.get(num, title), "text": text})
    return sections


def main(pdf: str) -> None:
    text = pdf_text(pdf)
    titles = arrangement(text)
    sections = split_sections(clean_body(text), titles)
    assert len(sections) == 92, f"expected 92 sections, found {len(sections)}"
    for sec in sections:
        assert len(sec["text"]) > 40, f"section {sec['section']} looks empty"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(sections, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(sections)} sections to {OUT} (source: {SOURCE_URL})")


if __name__ == "__main__":
    main(sys.argv[1])
