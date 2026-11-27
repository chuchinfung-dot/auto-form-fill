"""Compare shared fields between the student card and the Home Return Permit."""

from __future__ import annotations

import re
from typing import Any

from zhconv import convert as _zh_convert


def to_simplified(text: str) -> str:
    """Fold Traditional/Simplified variants to one script for comparison."""
    return _zh_convert(text, "zh-hans")


STATUS_LABELS = {
    "match": "✅ Match",
    "mismatch": "⚠️ Mismatch",
    "missing_student": "❌ Missing on student card",
    "missing_permit": "❌ Missing on permit",
    "missing": "❌ Missing on both",
}


def _normalize(value: str) -> str:
    if not value:
        return ""
    # 蘇朗言 and 苏朗言 are the same name written in different scripts.
    value = to_simplified(value)
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", value.lower())


def _english_name(student: dict[str, Any]) -> str:
    parts = [student.get("surname", ""), student.get("given_name", "")]
    return " ".join(p for p in parts if p).strip()


def _compare_value(student_val: str, permit_val: str) -> str:
    a = _normalize(student_val)
    b = _normalize(permit_val)
    if not a and not b:
        return "missing"
    if not a:
        return "missing_student"
    if not b:
        return "missing_permit"
    return "match" if a == b else "mismatch"


def _script_note(student_val: str, permit_val: str) -> str:
    """Explain a match that only held after folding Traditional/Simplified."""
    if (student_val or "").strip() == (permit_val or "").strip():
        return ""
    if to_simplified(student_val or "") == to_simplified(permit_val or ""):
        return "same name, Traditional/Simplified variants"
    return ""


def compare_documents(
    student: dict[str, Any], permit: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return one row per shared field, each with a status."""
    return [
        {
            "field": "English name",
            "student": _english_name(student),
            "permit": permit.get("english_name", ""),
            "status": _compare_value(
                _english_name(student), permit.get("english_name", "")
            ),
            "note": _script_note(
                _english_name(student), permit.get("english_name", "")
            ),
        },
        {
            "field": "Chinese name",
            "student": student.get("chinese_name", ""),
            "permit": permit.get("chinese_name", ""),
            "status": _compare_value(
                student.get("chinese_name", ""), permit.get("chinese_name", "")
            ),
            "note": _script_note(
                student.get("chinese_name", ""), permit.get("chinese_name", "")
            ),
        },
    ]


def build_summary(
    student: dict[str, Any],
    permit: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    required_student = [
        ("surname", "Surname"),
        ("given_name", "Given name"),
        ("student_number", "Student number"),
        ("programme", "Programme"),
    ]
    required_permit = [
        ("chinese_name", "Chinese name"),
        ("english_name", "English name"),
        ("permit_number", "Permit number"),
    ]
    missing: list[str] = []
    for key, label in required_student:
        if not student.get(key):
            missing.append(f"Student card — {label}")
    for key, label in required_permit:
        if not permit.get(key):
            missing.append(f"Permit — {label}")

    mismatch_count = sum(1 for r in results if r["status"] == "mismatch")
    compare_missing = sum(
        1
        for r in results
        if r["status"] in ("missing", "missing_student", "missing_permit")
    )

    return {
        "missing_fields": missing,
        "mismatch_count": mismatch_count,
        "compare_missing_count": compare_missing,
        "all_consistent": mismatch_count == 0 and compare_missing == 0,
    }
