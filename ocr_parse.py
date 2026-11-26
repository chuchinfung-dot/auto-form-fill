"""OCR: student card and Home Return Permit recognition + field parsing.

Student card      -> surname, given_name, student_number, programme, chinese_name.
Home Return Permit -> chinese_name, english_name, permit_number, dob, gender.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


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
    r"program|undertaking|surname|given|permit|home\s*return)",
    re.I,
)

CHINESE_LABELS = re.compile(
    r"(姓名|出生|日期|性別|簽發|有效期|證件|號碼|通行證|來往|港澳|"
    r"回鄉|居民|內地|中華|人民|共和國|公安部|出入境|管理局|持證人|本證|證號)"
)

STUDENT_NO_RE = re.compile(r"\b([A-Z]{1,4}\d{6,10})\b|\b(\d{8,12})\b", re.I)
PERMIT_NO_RE = re.compile(r"\b([A-Z]\d{7,9})\b")

# OCR frequently confuses these symbols for letters inside ID codes.
_ID_REPAIR = str.maketrans(
    {"$": "S", "€": "E", "£": "E", "|": "I", "!": "I", "¥": "Y"}
)

# Chinese label words that must never be mistaken for a person's name.
CHINESE_LABEL_WORDS = {
    "姓名", "英文名", "中文名", "名字", "別名", "出生", "出生日期", "日期",
    "性別", "性別sex", "證件", "證件號碼", "號碼", "簽發", "簽發日期",
    "有效期", "往來", "內地", "通行證", "往來內地通行證", "港澳居民",
    "港澳居民來往內地通行證", "居民", "中華人民共和國", "公安部", "出入境",
    "管理局", "持證人", "本證", "證號", "男", "女",
}

# Lines that are field labels, not values.
_LABEL_ONLY = {
    "programme",
    "program",
    "programme undertaking",
    "programme of study",
    "programme title",
}
DOB_RE = re.compile(
    r"(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})"
    r"|(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{4})"
)
GENDER_RE = re.compile(r"^(男|女|M|F)$", re.I)


@lru_cache(maxsize=1)
def get_reader():
    import easyocr

    # English + Traditional Chinese (the permit is in Traditional Chinese).
    return easyocr.Reader(["en", "ch_tra"], gpu=False, verbose=False)


def run_ocr(image_bgr) -> list[tuple[Any, str, float]]:
    """Return EasyOCR results: (bbox, text, confidence)."""
    import cv2

    from detect import preprocess_for_ocr

    prepared = preprocess_for_ocr(image_bgr)
    prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
    reader = get_reader()
    return reader.readtext(prepared)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _collect_lines(ocr_results: list[tuple[Any, str, float]]) -> list[dict[str, Any]]:
    lines = []
    for bbox, text, conf in ocr_results:
        t = _clean(str(text))
        if not t or conf < 0.03:
            continue
        y = 0.0
        try:
            y = float(bbox[0][1])
        except Exception:
            pass
        lines.append({"text": t, "conf": float(conf), "y": y})
    return lines


def _looks_like_name(text: str) -> bool:
    t = _clean(text)
    if len(t) < 3 or len(t) > 60:
        return False
    if SKIP_LINE_PATTERNS.search(t):
        return False
    letters = sum(c.isalpha() for c in t)
    if letters < 3:
        return False
    words = t.split()
    if not (1 <= len(words) <= 5):
        return False
    return all(w.replace("-", "").replace("'", "").isalpha() for w in words)


def _looks_like_programme(text: str) -> bool:
    t = _clean(text).lower().strip(" .:")
    if t in _LABEL_ONLY:
        return False
    if len(t) < 8:
        return False
    if STUDENT_NO_RE.search(text):
        return False
    return any(h in t for h in PROGRAMME_HINTS)


def split_name(full_name: str) -> tuple[str, str]:
    parts = _clean(full_name).split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0].title(), ""
    surname = parts[0].title()
    given = " ".join(p.title() for p in parts[1:])
    return surname, given


def _find_student_number(lines: list[dict[str, Any]]) -> str:
    for ln in sorted(lines, key=lambda x: -x["conf"]):
        raw = ln["text"].replace(" ", "")
        for candidate in (raw, raw.translate(_ID_REPAIR), ln["text"]):
            m = STUDENT_NO_RE.search(candidate)
            if m:
                return (m.group(1) or m.group(2)).upper()
    return ""


def _find_permit_number(lines: list[dict[str, Any]]) -> str:
    for ln in sorted(lines, key=lambda x: -x["conf"]):
        m = PERMIT_NO_RE.search(ln["text"].replace(" ", "")) or PERMIT_NO_RE.search(
            ln["text"]
        )
        if m:
            return m.group(1).upper()
    return ""


def _is_label_word(token: str) -> bool:
    if token in CHINESE_LABEL_WORDS:
        return True
    if CHINESE_LABELS.search(token):
        return True
    return any(token in label for label in CHINESE_LABEL_WORDS if len(label) > len(token))


def _find_chinese_name(lines: list[dict[str, Any]]) -> str:
    cands = []
    for ln in lines:
        for m in re.finditer(r"[\u4e00-\u9fff]{2,4}", ln["text"]):
            token = m.group(0)
            if _is_label_word(token):
                continue
            cands.append((ln["y"], -ln["conf"], token))
    if not cands:
        return ""
    cands.sort(key=lambda x: (x[0], x[1]))
    return cands[0][2]


def _find_english_name(lines: list[dict[str, Any]]) -> str:
    cands = [ln for ln in lines if _looks_like_name(ln["text"])]
    if not cands:
        return ""
    cands.sort(
        key=lambda x: (
            1 if (x["text"].split()[0].isupper() and len(x["text"].split()[0]) > 1) else 0,
            x["conf"],
            len(x["text"]),
        ),
        reverse=True,
    )
    return cands[0]["text"]


def _find_dob(lines: list[dict[str, Any]]) -> str:
    for ln in lines:
        m = DOB_RE.search(ln["text"])
        if m:
            return m.group(0)
    return ""


def _find_gender(lines: list[dict[str, Any]]) -> str:
    for ln in lines:
        m = GENDER_RE.search(ln["text"])
        if m:
            g = m.group(0).upper()
            if g in ("男", "M"):
                return "M"
            if g in ("女", "F"):
                return "F"
    return ""


LOW_CONFIDENCE_THRESHOLD = 0.50


def _conf_for(value: str, lines: list[dict[str, Any]], default: float = 0.0) -> float:
    """Best-effort confidence of the OCR line(s) that produced a parsed value."""
    if not value:
        return 0.0
    key = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", value.lower())
    if not key:
        return default
    for ln in lines:
        lnk = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", ln["text"].lower())
        if lnk and (key in lnk or lnk in key):
            return ln["conf"]
    words = re.findall(r"[a-z\u4e00-\u9fff]{4,}", value.lower())
    confs = []
    for ln in lines:
        low = ln["text"].lower()
        if any(w in low for w in words):
            confs.append(ln["conf"])
    return min(confs) if confs else default


def _join_programme_block(cands: list[dict[str, Any]], max_gap: float = 80.0) -> str:
    """A programme title may wrap across lines; join the strongest block."""
    ordered = sorted(cands, key=lambda x: x["y"])
    blocks: list[list[dict[str, Any]]] = [[ordered[0]]]
    for ln in ordered[1:]:
        if 0 < ln["y"] - blocks[-1][-1]["y"] < max_gap:
            blocks[-1].append(ln)
        else:
            blocks.append([ln])
    best = max(
        blocks,
        key=lambda b: sum(len(x["text"]) for x in b) * max(x["conf"] for x in b),
    )
    return " ".join(x["text"] for x in best)


def parse_card_fields(ocr_results: list[tuple[Any, str, float]]) -> dict[str, str]:
    """Heuristic parse of an IVE student card."""
    lines = _collect_lines(ocr_results)

    student_number = _find_student_number(lines)

    name_candidates = [ln for ln in lines if _looks_like_name(ln["text"])]
    name_candidates.sort(
        key=lambda x: (
            1 if (x["text"].split()[0].isupper() and len(x["text"].split()[0]) > 1) else 0,
            x["conf"],
            len(x["text"]),
        ),
        reverse=True,
    )
    full_name = name_candidates[0]["text"] if name_candidates else ""
    surname, given = split_name(full_name)

    programme = ""
    prog_cands = [ln for ln in lines if _looks_like_programme(ln["text"])]
    if prog_cands:
        programme = _join_programme_block(prog_cands).title()
    else:
        for ln in sorted(lines, key=lambda x: (-len(x["text"]), -x["conf"])):
            text = ln["text"]
            if SKIP_LINE_PATTERNS.search(text):
                continue
            if _clean(text).lower().strip(" .:") in _LABEL_ONLY:
                continue
            if text.upper() == student_number or text.upper().replace(" ", "") == student_number:
                continue
            if full_name and text.lower() == full_name.lower():
                continue
            if sum(c.isalpha() for c in text) >= 10:
                programme = text.title()
                break

    chinese_name = _find_chinese_name(lines)
    values = {
        "surname": surname,
        "given_name": given,
        "student_number": student_number,
        "programme": programme,
        "chinese_name": chinese_name,
    }
    return {
        **values,
        "confidence": {
            k: round(_conf_for(v, lines), 3) for k, v in values.items()
        },
        "low_confidence": [
            k for k, v in values.items()
            if v and _conf_for(v, lines) < LOW_CONFIDENCE_THRESHOLD
        ],
        "raw_lines": [ln["text"] for ln in lines],
    }


def parse_permit_fields(ocr_results: list[tuple[Any, str, float]]) -> dict[str, str]:
    """Heuristic parse of a Home Return Permit (港澳居民來往內地通行證)."""
    lines = _collect_lines(ocr_results)
    values = {
        "chinese_name": _find_chinese_name(lines),
        "english_name": _find_english_name(lines),
        "permit_number": _find_permit_number(lines),
        "dob": _find_dob(lines),
        "gender": _find_gender(lines),
    }
    return {
        **values,
        "confidence": {
            k: round(_conf_for(v, lines), 3) for k, v in values.items()
        },
        "low_confidence": [
            k for k, v in values.items()
            if v and _conf_for(v, lines) < LOW_CONFIDENCE_THRESHOLD
        ],
        "raw_lines": [ln["text"] for ln in lines],
    }


def ocr_and_parse(image_bgr) -> dict[str, str]:
    return parse_card_fields(run_ocr(image_bgr))


def ocr_and_parse_permit(image_bgr) -> dict[str, str]:
    return parse_permit_fields(run_ocr(image_bgr))
