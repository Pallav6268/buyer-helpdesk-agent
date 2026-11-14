"""Load the knowledge base: fictional project documents and the RERA Act, split into citable passages.

Every passage has a short, stable ID that the agent cites in its answers, for example
``payment_plan-3`` (third section of the payment plan) or ``rera-18`` (Section 18 of the Act).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_PASSAGE_CHARS = 1500  # long Act sections are split at sub-section boundaries


@dataclass(frozen=True)
class Passage:
    id: str
    source: str  # "project" or "rera"
    title: str
    text: str

    def as_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "title": self.title, "text": self.text}


def load_project_docs(folder: Path = DATA_DIR / "project") -> list[Passage]:
    """Each ``## Heading`` in a project Markdown file becomes one passage."""
    passages: list[Passage] = []
    for path in sorted(folder.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc_title = re.search(r"^# (.+)$", text, re.M)
        doc_name = doc_title.group(1).split("—")[-1].strip() if doc_title else path.stem
        for n, (heading, body) in enumerate(re.findall(r"^## (.+?)\n(.*?)(?=^## |\Z)", text, re.M | re.S), 1):
            passages.append(Passage(f"{path.stem}-{n}", "project", f"{doc_name} › {heading.strip()}", body.strip()))
    return passages


def split_section(text: str, limit: int = MAX_PASSAGE_CHARS) -> list[str]:
    """Split a long Act section at sub-section or clause markers like "(2)" or "(zk)" so each
    piece stays readable."""
    if len(text) <= limit:
        return [text]
    pieces = re.split(r"\n(?=\((?:\d+[A-Z]?|[a-z]{1,3})\)\s)", text)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > limit:
            chunks.append(current.strip())
            current = piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def load_rera(path: Path = DATA_DIR / "rera" / "sections.json") -> list[Passage]:
    """Sections of the Act, extracted from the official text by scripts/extract_rera.py."""
    if not path.exists():
        return []
    passages: list[Passage] = []
    for sec in json.loads(path.read_text(encoding="utf-8")):
        chunks = split_section(sec["text"])
        for i, chunk in enumerate(chunks, 1):
            suffix = f"-part{i}" if len(chunks) > 1 else ""
            part = f" (part {i} of {len(chunks)})" if len(chunks) > 1 else ""
            title = f"RERA Act 2016 › Section {sec['section']}. {sec['title']}{part}"
            passages.append(Passage(f"rera-{sec['section']}{suffix}", "rera", title, chunk))
    return passages


def rera_section_text(section: str, path: Path = DATA_DIR / "rera" / "sections.json") -> dict | None:
    """The full text of one section, unsplit."""
    if not path.exists():
        return None
    wanted = section.strip().lower().removeprefix("section").strip()
    for sec in json.loads(path.read_text(encoding="utf-8")):
        if sec["section"].lower() == wanted:
            return sec
    return None


def load_buyers(path: Path = DATA_DIR / "buyers.json") -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_all() -> list[Passage]:
    return load_project_docs() + load_rera()
