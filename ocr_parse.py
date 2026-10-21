"""OCR student card text and parse into form fields."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np


PROGRAMME_HINTS = (
    "diploma",
    "foundation",
    "higher diploma",
    "bachelor",
    "programme",
    "program",
    "studies",
    "engineering",
    "business",
    "design",
    "information",
    "technology",
    "science",
    "nursing",
    "hotel",
    "tourism",
    "accounting",
    "electronics",
    "mechanical",
    "electrical",
    "civil",
)

SKIP_LINE_PATTERNS = re.compile(
    r"(student\s*card|hong\s*kong|institute|vocational|education|"
    r"ive\b|vtc|valid|expiry|issue|date|name|number|programme|"
    r"program|undertaking|surname|given)",
    re.I,
)

STUDENT_NO_RE = re.compile(
    r"\b([A-Z]{1,4}\d{6,10})\b|\b(\d{8,12})\b",
    re.I,
)


@lru_cache(maxsize=1)
def get_reader():
    import easyocr

    # English only for IVE English fields; first call downloads models.
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def run_ocr(image_bgr: np.ndarray) -> list[tuple[Any, str, float]]:
    """Return EasyOCR results: (bbox, text, confidence)."""
    reader = get_reader()
    rgb = image_bgr[:, :, ::-1]
    return reader.readtext(rgb)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_name(text: str) -> bool:
    t = _clean(text)
    if len(t) < 3 or len(t) > 60:
        return False
    if SKIP_LINE_PATTERNS.search(t):
        return False
    if STUDENT_NO_RE.search(t) and not re.search(r"[A-Za-z]{2,}", t.replace(STUDENT_NO_RE.search(t).group(0), "")):
        return False
    letters = sum(c.isalpha() for c in t)
    if letters < 3:
        return False
    words = t.split()
    if not (1 <= len(words) <= 5):
        return False
    # Prefer ALL-CAPS or Title Case English names
    return all(w.replace("-", "").isalpha() for w in words)


def _looks_like_programme(text: str) -> bool:
    t = _clean(text).lower()
    if len(t) < 8:
        return False
    if STUDENT_NO_RE.search(text):
        return False
    return any(h in t for h in PROGRAMME_HINTS)


def split_name(full_name: str) -> tuple[str, str]:
    """IVE-style English name: first token = surname (often CAPS)."""
    parts = _clean(full_name).split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0].title(), ""
    surname = parts[0].title()
    given = " ".join(p.title() for p in parts[1:])
    return surname, given


def parse_card_fields(ocr_results: list[tuple[Any, str, float]]) -> dict[str, str]:
    """Heuristic parse → Surname, Given name, Student number, Programme."""
    lines = []
    for _bbox, text, conf in ocr_results:
        t = _clean(str(text))
        if not t or conf < 0.15:
            continue
        lines.append((t, float(conf)))

    student_number = ""
    for text, conf in sorted(lines, key=lambda x: -x[1]):
        m = STUDENT_NO_RE.search(text.replace(" ", ""))
        if not m:
            m = STUDENT_NO_RE.search(text)
        if m:
            student_number = (m.group(1) or m.group(2)).upper()
            break

    name_candidates = [(t, c) for t, c in lines if _looks_like_name(t)]
    # Prefer lines that look like SURNAME Given Names (first word all caps)
    name_candidates.sort(
        key=lambda x: (
            1 if x[0].split()[0].isupper() and len(x[0].split()[0]) > 1 else 0,
            x[1],
            len(x[0]),
        ),
        reverse=True,
    )
    full_name = name_candidates[0][0] if name_candidates else ""
    surname, given = split_name(full_name)

    programme = ""
    prog_cands = [(t, c) for t, c in lines if _looks_like_programme(t)]
    if prog_cands:
        prog_cands.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        programme = prog_cands[0][0].title()
    else:
        # Fallback: longest non-name, non-number alphabetic line
        for text, conf in sorted(lines, key=lambda x: (-len(x[0]), -x[1])):
            if SKIP_LINE_PATTERNS.search(text):
                continue
            if text.upper() == student_number or text.upper().replace(" ", "") == student_number:
                continue
            if full_name and text.lower() == full_name.lower():
                continue
            if sum(c.isalpha() for c in text) >= 10:
                programme = text.title()
                break

    return {
        "surname": surname,
        "given_name": given,
        "student_number": student_number,
        "programme": programme,
        "raw_lines": [t for t, _ in lines],
    }


def ocr_and_parse(image_bgr: np.ndarray) -> dict[str, str]:
    results = run_ocr(image_bgr)
    return parse_card_fields(results)
