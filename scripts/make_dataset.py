"""
scripts/make_dataset.py
Data loading and text cleaning for lecture materials (PPTX slides,
auto-generated transcripts, student notes).
"""

import re
from pathlib import Path
from typing import List, Dict

from pptx import Presentation


# ── Text cleaning ──────────────────────────────────────────────────────

def clean_transcript(raw: str) -> str:
    """Remove [Auto-generated ...] header, join caption fragments, collapse whitespace."""
    lines = raw.splitlines()
    if lines and lines[0].startswith("[Auto-generated"):
        lines = lines[1:]
    text = " ".join(ln.strip() for ln in lines if ln.strip())
    return re.sub(r" {2,}", " ", text).strip()


def clean_notes(raw: str) -> str:
    """Normalize line endings and collapse 3+ blank lines to 2."""
    text = raw.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Loaders ────────────────────────────────────────────────────────────

def load_txt(path: Path) -> Dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if "Captions" in path.name:
        doc_type = "transcript"
        text = clean_transcript(raw)
    else:
        doc_type = "notes"
        text = clean_notes(raw)
    return {
        "source":   path.name,
        "doc_type": doc_type,
        "text":     text,
        "metadata": {"path": str(path), "chars": len(text)},
    }


def load_pptx(path: Path) -> Dict:
    """Extract text from every slide. Returns full-doc text and per-slide list."""
    prs = Presentation(path)
    slides_data = []

    for slide_num, slide in enumerate(prs.slides, start=1):
        title_text, body_parts = "", []
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            frame = shape.text_frame.text.strip()
            if not frame:
                continue
            try:
                is_title = shape.is_placeholder and shape.placeholder_format.idx == 0
            except AttributeError:
                is_title = False
            if is_title:
                title_text = frame
            else:
                body_parts.append(frame)

        full = "\n".join(filter(None, [title_text] + body_parts)).strip()
        if full:
            slides_data.append({"slide_num": slide_num, "title": title_text, "text": full})

    full_doc = "\n\n--- SLIDE BREAK ---\n\n".join(s["text"] for s in slides_data)
    return {
        "source":   path.name,
        "doc_type": "slides",
        "text":     full_doc,
        "slides":   slides_data,
        "metadata": {"path": str(path), "num_slides": len(slides_data), "chars": len(full_doc)},
    }


def load_all_documents(data_dir: Path) -> List[Dict]:
    """Load all .txt and .pptx files from data_dir."""
    documents = []
    for path in sorted(data_dir.iterdir()):
        ext = path.suffix.lower()
        if ext == ".txt":
            documents.append(load_txt(path))
        elif ext == ".pptx":
            documents.append(load_pptx(path))
    return documents
