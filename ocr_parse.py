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
    # simplified variants (PP-OCR sometimes emits these for traditional text)
    "内地", "通行证", "往来", "证件", "号码", "性别", "签发", "有效期",
    "居民", "中华", "人民", "共和国", "出生", "日期", "姓名", "英文名",
    "中文名", "往来内地通行证", "港澳居民",
}

# Words that appear on cards as labels. OCR fragments of these ("Tudent" from
# "Student") must not be mistaken for a person's name.
LABEL_WORDS = {
    "student", "name", "card", "programme", "program", "permit", "holder",
    "identity", "number", "valid", "expiry", "issue", "institute",
    "vocational", "education", "surname", "given", "chinese", "english",
    "birth", "date", "gender", "home", "return", "travel", "undertaking",
    "through", "registrar", "faculty", "department", "campus",
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
    """PP-OCRv4 via onnxruntime: strong on both Latin and CJK, no PyTorch needed."""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(params={"Global.print_verbose": False})


def run_ocr(image_bgr) -> list[tuple[Any, str, float]]:
    """Return OCR results as (bbox, text, confidence)."""
    import cv2

    from detect import preprocess_for_ocr

    prepared = preprocess_for_ocr(image_bgr)
    prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)

    results, _elapsed = get_reader()(prepared)
    if not results:
        return []
    return [(item[0], str(item[1]), float(item[2])) for item in results]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _collect_lines(ocr_results: list[tuple[Any, str, float]]) -> list[dict[str, Any]]:
    lines = []
    for bbox, text, conf in ocr_results:
        t = _clean(str(text))
        if not t or conf < 0.03:
            continue
        y = h = x0 = y0 = x1 = y1 = 0.0
        try:
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            y, h = y0, max(y1 - y0, 1.0)
        except Exception:
            pass
        lines.append(
            {
                "text": t,
                "conf": float(conf),
                "y": y,
                "h": h,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            }
        )
    return lines


LABEL_PATTERNS = {
    "english_name": (r"^name$", r"^student\s*name$", r"^english\s*name$",
                     r"英文名", r"英文姓名"),
    "chinese_name": (r"^chinese\s*name$", r"^name\s*in\s*chinese$",
                     r"中文名", r"中文姓名", r"姓名"),
    "programme": (r"^programme", r"^program\b"),
    "student_number": (r"^student\s*(no|number|id)\b", r"^student\s*no\.?$"),
    "permit_number": (r"^證件號碼", r"^证件号码", r"^passport\s*no"),
    "dob": (r"^出生日期", r"^date\s*of\s*birth", r"^dob\b"),
    "gender": (r"^性別", r"^性别", r"^sex\b", r"^gender\b"),
}


def _line_height(lines: list[dict[str, Any]]) -> float:
    hs = sorted(ln.get("h", 0) for ln in lines if ln.get("h"))
    return hs[len(hs) // 2] if hs else 20.0


def _row_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    top = max(a.get("y0", a["y"]), b.get("y0", b["y"]))
    bottom = min(a.get("y1", a["y"] + a.get("h", 1)), b.get("y1", b["y"] + b.get("h", 1)))
    span = min(a.get("h", 1), b.get("h", 1)) or 1
    return max(0.0, bottom - top) / span


def _label_lines(lines: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    pats = LABEL_PATTERNS[key]
    return [ln for ln in lines if any(re.search(p, ln["text"].lower()) for p in pats)]


def _value_of_label(
    label: dict[str, Any], lines: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Prefer the text to the right on the same row, else the row below."""
    same_row = [
        ln
        for ln in lines
        if ln is not label and _row_overlap(label, ln) > 0.5
        and ln.get("x0", 0) >= label.get("x1", 0) - 5
    ]
    if same_row:
        return min(same_row, key=lambda ln: ln.get("x0", 0))

    line_h = _line_height(lines)
    below = [
        ln
        for ln in lines
        if ln is not label
        and 0 < ln.get("y0", 0) - label.get("y1", 0) < 2.5 * line_h
    ]
    return min(below, key=lambda ln: ln.get("y0", 0)) if below else None


def _from_label(
    lines: list[dict[str, Any]], key: str, accept
) -> dict[str, Any] | None:
    """First label row whose neighbour passes `accept`."""
    for label in _label_lines(lines, key):
        value = _value_of_label(label, lines)
        if value is not None and accept(value["text"]):
            return value
    return None


def _within_one_edit(a: str, b: str) -> bool:
    """True when a and b differ by at most one insert/delete/substitute."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(a) == len(b):
            i += 1
            j += 1
        else:
            j += 1
    return edits + (len(b) - j) <= 1


def _is_label_like(word: str) -> bool:
    w = word.lower()
    if len(w) < 4:
        return False
    return any(_within_one_edit(w, lbl) for lbl in LABEL_WORDS)


def _split_camel(word: str) -> list[str]:
    """SO LongYin -> ['SO', 'Long', 'Yin'] (OCR drops the inner space)."""
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+", word)
    return parts or [word]


def _normalize_name_text(text: str) -> str:
    """Repair punctuation and missing spaces in an OCR'd Latin name."""
    t = re.sub(r"[,;:]+", " ", _clean(text))
    t = re.sub(r"\s+", " ", t).strip()
    words: list[str] = []
    for w in t.split():
        if re.search(r"[a-z][A-Z]", w):
            words.extend(_split_camel(w))
        else:
            words.append(w)
    return " ".join(words)


def _looks_like_name(text: str) -> bool:
    t = _normalize_name_text(text)
    if len(t) < 3 or len(t) > 60:
        return False
    # English names are ASCII only: "出生日期DOB" is a label, not a name.
    if any("\u4e00" <= c <= "\u9fff" for c in t):
        return False
    if SKIP_LINE_PATTERNS.search(t):
        return False
    letters = sum(c.isalpha() for c in t)
    if letters < 3:
        return False
    words = t.split()
    if not (1 <= len(words) <= 5):
        return False
    if any(len(w) > 14 for w in words):
        return False
    # "TECHNOLOGY", "INFORMATION" are programme words, not surnames.
    if any(h in t.lower() for h in PROGRAMME_HINTS):
        return False
    # A lone word that is (nearly) a field label is an OCR fragment, e.g.
    # "Tudent" cut from "Student".
    if len(words) == 1 and _is_label_like(words[0]):
        return False
    return all(w.replace("-", "").replace("'", "").isascii() and
               w.replace("-", "").replace("'", "").isalpha() for w in words)


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
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", ln["text"])
        # A long run of Han characters is a title/label line, never a name.
        if len(cjk_chars) > 5:
            continue
        for m in re.finditer(r"[\u4e00-\u9fff]{2,4}", ln["text"]):
            token = m.group(0)
            if _is_label_word(token):
                continue
            cands.append((ln["y"], -ln["conf"], token))
    if not cands:
        return ""
    cands.sort(key=lambda x: (x[0], x[1]))
    return cands[0][2]


def _latin_name_near_cjk(
    lines: list[dict[str, Any]], anchor: dict[str, Any] | None
) -> str:
    """These cards print no label: the Latin name sits beside the Chinese one."""
    cands = []
    for ln in lines:
        if ln is anchor:
            continue
        text = _normalize_name_text(ln["text"])
        if not _looks_like_name(text):
            continue
        if anchor is not None:
            dist = abs(ln.get("y0", ln["y"]) - anchor.get("y0", anchor["y"]))
        else:
            dist = ln.get("y0", ln["y"])
        cands.append((dist, ln, text))
    if not cands:
        return ""
    cands.sort(key=lambda x: x[0])
    best = cands[0][1]

    # A name can be split across several boxes on the same row.
    row = [
        ln
        for ln in lines
        if _row_overlap(best, ln) > 0.5 and _looks_like_name(ln["text"])
    ]
    row.sort(key=lambda ln: ln.get("x0", 0))
    texts = [_normalize_name_text(ln["text"]) for ln in row]
    return " ".join(t for t in texts if t)


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
    return _normalize_name_text(cands[0]["text"])


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
    # Gender is normalised to M/F but printed as 男/女.
    alias = {"M": "男", "F": "女"}.get(value.upper(), value)
    key = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", alias.lower())
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


def _absorb_continuation(primary: dict[str, Any], lines: list[dict[str, Any]]) -> str:
    """Grow a value downward while the next row still belongs to it.

    Programme titles wrap, and the wrapped line ("Administration") often
    carries no keyword to match on, so continuation is decided by layout:
    same left edge, next row, similar size.
    """
    line_h = _line_height(lines)
    max_step = max(line_h * 2.5, 60.0)
    primary_x = primary.get("x0", 0.0)
    tol = max(abs(primary.get("x1", 0.0) - primary_x) * 0.25, 40.0)

    ordered = sorted(lines, key=lambda ln: ln.get("y0", ln["y"]))
    try:
        idx = ordered.index(primary)
    except ValueError:
        return primary["text"]

    parts = [primary["text"]]
    cursor = primary
    for ln in ordered[idx + 1:]:
        step = ln.get("y0", ln["y"]) - cursor.get("y0", cursor["y"])
        if not (0 < step < max_step):
            break
        if abs(ln.get("x0", 0.0) - primary_x) > tol:
            break
        text = ln["text"]
        if SKIP_LINE_PATTERNS.search(text):
            break
        if _clean(text).lower().strip(" .:") in _LABEL_ONLY:
            break
        if re.fullmatch(r"[\d\s.:\-/]+", text):
            break
        parts.append(text)
        cursor = ln
    return " ".join(parts)


def _programme_from(lines: list[dict[str, Any]]) -> str:
    cands = [ln for ln in lines if _looks_like_programme(ln["text"])]
    if not cands:
        return ""
    line_h = _line_height(lines)
    max_step = max(line_h * 2.5, 60.0)

    ordered = sorted(cands, key=lambda ln: ln.get("y0", ln["y"]))
    blocks: list[list[dict[str, Any]]] = [[ordered[0]]]
    for ln in ordered[1:]:
        step = ln.get("y0", ln["y"]) - blocks[-1][-1].get("y0", blocks[-1][-1]["y"])
        if 0 < step < max_step:
            blocks[-1].append(ln)
        else:
            blocks.append([ln])

    best = max(
        blocks,
        key=lambda b: sum(len(x["text"]) for x in b) * max(x["conf"] for x in b),
    )
    return _absorb_continuation(best[0], lines)


def parse_card_fields(ocr_results: list[tuple[Any, str, float]]) -> dict[str, str]:
    """Heuristic parse of an IVE student card."""
    lines = _collect_lines(ocr_results)

    student_number = ""
    sn_line = _from_label(
        lines,
        "student_number",
        lambda t: bool(STUDENT_NO_RE.search(t.replace(" ", ""))
                       or STUDENT_NO_RE.search(t.translate(_ID_REPAIR))),
    )
    if sn_line is not None:
        student_number = _find_student_number([sn_line]) or ""
    if not student_number:
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

    # Layout wins: the value sitting beside the "Name" label, when there is one.
    name_line = _from_label(
        lines,
        "english_name",
        lambda t: not any("\u4e00" <= c <= "\u9fff" for c in t)
        and sum(c.isalpha() for c in t) >= 3,
    )
    if name_line is not None:
        full_name = _normalize_name_text(name_line["text"])
    else:
        # IVE cards print no label: the Latin name sits beside the Chinese one.
        anchor = _from_label(
            lines,
            "chinese_name",
            lambda t: bool(re.search(r"[\u4e00-\u9fff]{2,4}", t)),
        )
        near = _latin_name_near_cjk(lines, anchor)
        if near:
            full_name = near

    full_name = _normalize_name_text(full_name)
    surname, given = split_name(full_name)

    programme = _programme_from(lines)
    if not programme:
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
                programme = text
                break

    prog_value = _from_label(
        lines,
        "programme",
        lambda t: _clean(t).lower().strip(" .:") not in _LABEL_ONLY and len(t) >= 4,
    )
    if prog_value is not None:
        programme = _absorb_continuation(prog_value, lines)

    chinese_name = _find_chinese_name(lines)
    cn_line = _from_label(
        lines, "chinese_name", lambda t: bool(re.search(r"[\u4e00-\u9fff]{2,4}", t))
    )
    if cn_line is not None:
        m = re.search(r"[\u4e00-\u9fff]{2,4}", cn_line["text"])
        if m and not _is_label_word(m.group(0)):
            chinese_name = m.group(0)

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

    chinese_name = ""
    cn = _from_label(
        lines, "chinese_name", lambda t: bool(re.search(r"[\u4e00-\u9fff]{2,4}", t))
    )
    if cn is not None:
        m = re.search(r"[\u4e00-\u9fff]{2,4}", cn["text"])
        if m and not _is_label_word(m.group(0)):
            chinese_name = m.group(0)
    if not chinese_name:
        chinese_name = _find_chinese_name(lines)

    english_name = ""
    en = _from_label(
        lines,
        "english_name",
        lambda t: not any("\u4e00" <= c <= "\u9fff" for c in t)
        and sum(c.isalpha() for c in t) >= 3,
    )
    if en is not None:
        english_name = _normalize_name_text(en["text"])
    if not english_name:
        cn_anchor = _from_label(
            lines,
            "chinese_name",
            lambda t: bool(re.search(r"[\u4e00-\u9fff]{2,4}", t)),
        )
        english_name = _latin_name_near_cjk(lines, cn_anchor)
    if not english_name:
        english_name = _find_english_name(lines)

    def _pick(label_key, finder, accept):
        line = _from_label(lines, label_key, accept)
        if line is not None:
            got = finder([line])
            if got:
                return got
        return finder(lines)

    values = {
        "chinese_name": chinese_name,
        "english_name": english_name,
        "permit_number": _pick(
            "permit_number", _find_permit_number, lambda t: bool(PERMIT_NO_RE.search(t))
        ),
        "dob": _pick("dob", _find_dob, lambda t: bool(DOB_RE.search(t))),
        "gender": _pick("gender", _find_gender, lambda t: bool(GENDER_RE.search(t))),
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
