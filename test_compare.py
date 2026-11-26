"""Focused tests for the two-card comparison and permit parsing logic."""

import unittest

from compare import build_summary, compare_documents
from ocr_parse import (
    _find_chinese_name,
    _find_permit_number,
    _find_student_number,
    _looks_like_name,
    _looks_like_programme,
    split_name,
)


def _mk_lines(items):
    return [{"text": t, "conf": 0.9, "y": float(i)} for i, t in enumerate(items)]


class TestCompare(unittest.TestCase):
    def test_english_and_chinese_match(self):
        student = {"surname": "CHAN", "given_name": "Tai Man", "chinese_name": "陳大明"}
        permit = {"english_name": "CHAN TAI MAN", "chinese_name": "陳大明"}
        statuses = {r["field"]: r["status"] for r in compare_documents(student, permit)}
        self.assertEqual(statuses["English name"], "match")
        self.assertEqual(statuses["Chinese name"], "match")

    def test_english_name_mismatch(self):
        student = {"surname": "CHAN", "given_name": "Tai Man", "chinese_name": ""}
        permit = {"english_name": "CHAN TAI MUN", "chinese_name": ""}
        statuses = {r["field"]: r["status"] for r in compare_documents(student, permit)}
        self.assertEqual(statuses["English name"], "mismatch")

    def test_missing_permit(self):
        student = {"surname": "CHAN", "given_name": "Tai Man", "chinese_name": ""}
        permit = {"english_name": "", "chinese_name": ""}
        statuses = {r["field"]: r["status"] for r in compare_documents(student, permit)}
        self.assertEqual(statuses["English name"], "missing_permit")

    def test_normalization_ignores_case_and_spaces(self):
        student = {"surname": "chan", "given_name": "TAI MAN", "chinese_name": ""}
        permit = {"english_name": "Chan Tai Man", "chinese_name": ""}
        statuses = {r["field"]: r["status"] for r in compare_documents(student, permit)}
        self.assertEqual(statuses["English name"], "match")

    def test_traditional_and_simplified_names_match(self):
        student = {"surname": "SO", "given_name": "Long Yin", "chinese_name": "蘇朗言"}
        permit = {"english_name": "SO LONG YIN", "chinese_name": "苏朗言"}
        rows = {r["field"]: r for r in compare_documents(student, permit)}
        self.assertEqual(rows["Chinese name"]["status"], "match")
        self.assertTrue(rows["Chinese name"]["note"])

    def test_different_chinese_names_still_mismatch(self):
        student = {"surname": "", "given_name": "", "chinese_name": "蘇朗言"}
        permit = {"english_name": "", "chinese_name": "陳大明"}
        rows = {r["field"]: r for r in compare_documents(student, permit)}
        self.assertEqual(rows["Chinese name"]["status"], "mismatch")

    def test_summary_lists_missing(self):
        student = {
            "surname": "",
            "given_name": "Tai Man",
            "student_number": "S1234567",
            "programme": "Higher Diploma in IT",
            "chinese_name": "",
        }
        permit = {
            "chinese_name": "陳大明",
            "english_name": "CHAN TAI MAN",
            "permit_number": "H12345678",
        }
        results = compare_documents(student, permit)
        summary = build_summary(student, permit, results)
        self.assertIn("Student card — Surname", summary["missing_fields"])
        self.assertFalse(summary["all_consistent"])


class TestPermitParse(unittest.TestCase):
    def test_permit_number(self):
        self.assertEqual(_find_permit_number(_mk_lines(["H12345678", "CHAN TAI MAN"])), "H12345678")

    def test_chinese_name_skips_labels(self):
        lines = _mk_lines(["CHAN TAI MAN", "陳大明", "出生日期"])
        self.assertEqual(_find_chinese_name(lines), "陳大明")

    def test_split_name(self):
        self.assertEqual(split_name("CHAN TAI MAN"), ("Chan", "Tai Man"))

    def test_student_number_repairs_dollar_prefix(self):
        # OCR reads "S1234567" as "$1234567" on the student card.
        self.assertEqual(_find_student_number(_mk_lines(["$1234567"])), "S1234567")

    def test_programme_label_is_not_a_value(self):
        self.assertFalse(_looks_like_programme("Programme"))
        self.assertFalse(_looks_like_programme("Programme undertaking"))
        self.assertTrue(_looks_like_programme("HIGHER DIPLOMA IN INFORMATION TECHNOLOGY"))

    def test_chinese_name_skips_label_words(self):
        lines = _mk_lines(["英文名", "陳大明"])
        self.assertEqual(_find_chinese_name(lines), "陳大明")

    def test_chinese_name_skips_title_line(self):
        # The document title has too many Han characters to be a name.
        lines = _mk_lines(["往來内地通行證", "陳大明"])
        self.assertEqual(_find_chinese_name(lines), "陳大明")

    def test_programme_word_is_not_a_name(self):
        # "TECHNOLOGY" passed as a name before programme words were excluded.
        self.assertFalse(_looks_like_name("TECHNOLOGY"))
        self.assertFalse(_looks_like_name("INFORMATION"))
        self.assertTrue(_looks_like_name("CHAN TAI MAN"))

    def test_label_fragment_is_not_a_name(self):
        # "Tudent" is "Student" read with the first letter clipped.
        self.assertFalse(_looks_like_name("Tudent"))
        self.assertFalse(_looks_like_name("Permlt"))
        self.assertTrue(_looks_like_name("SO LONG YIN"))

    def test_cjk_line_is_not_an_english_name(self):
        # "出生日期DOB" is a label, not a name.
        self.assertFalse(_looks_like_name("出生日期DOB"))


if __name__ == "__main__":
    unittest.main()
