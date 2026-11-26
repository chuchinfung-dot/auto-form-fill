"""Focused tests for the two-card comparison and permit parsing logic."""

import unittest

from compare import build_summary, compare_documents
from ocr_parse import (
    parse_card_fields,
    parse_permit_fields,
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


def _mk_ocr(rows):
    """Build EasyOCR-style results: (text, x0, y0, width, height, conf)."""
    out = []
    for text, x0, y0, w, h, conf in rows:
        box = [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]
        out.append((box, text, conf))
    return out


class TestLayoutParsing(unittest.TestCase):
    """Names come from the value beside the label, not from text heuristics."""

    def test_card_ignores_card_heading_fragment(self):
        # "Card" was split by OCR, leaving "Ard" as its own line.
        rows = _mk_ocr([
            ("IVE", 40, 30, 80, 40, 0.95),
            ("STUDENT CARD", 140, 30, 300, 40, 0.95),
            ("Ard", 500, 30, 60, 40, 0.90),
            ("Name", 40, 200, 120, 30, 0.99),
            ("SO LONG YIN", 380, 200, 300, 30, 0.95),
            ("Chinese Name", 40, 300, 220, 30, 0.99),
            ("蘇朗言", 380, 300, 150, 30, 0.95),
            ("Student No.", 40, 400, 200, 30, 0.99),
            ("S1234567", 380, 400, 200, 30, 0.99),
            ("Programme", 40, 500, 180, 30, 0.99),
            ("HIGHER DIPLOMA IN INFORMATION", 380, 500, 560, 30, 0.96),
            ("TECHNOLOGY", 380, 545, 260, 30, 0.98),
        ])
        parsed = parse_card_fields(rows)
        self.assertEqual(parsed["surname"], "So")
        self.assertEqual(parsed["given_name"], "Long Yin")
        self.assertEqual(parsed["chinese_name"], "蘇朗言")
        self.assertEqual(parsed["student_number"], "S1234567")
        self.assertEqual(parsed["programme"], "HIGHER DIPLOMA IN INFORMATION TECHNOLOGY")

    def test_unlabelled_card_anchors_latin_name_on_chinese(self):
        # Real IVE cards print no "Name" label: SO Long Yin sits above 蘇朗言.
        rows = _mk_ocr([
            ("SO LongYin", 600, 100, 350, 65, 0.94),
            ("CARD", 1380, 110, 65, 240, 0.99),
            ("STUDENT", 1444, 140, 60, 370, 0.98),
            ("蘇朗言", 608, 166, 175, 60, 0.92),
            ("Higher Diploma in Computer", 596, 341, 548, 55, 0.98),
            ("and Electronic Engineering", 592, 390, 520, 59, 0.97),
            ("21882501958355", 149, 667, 442, 50, 1.00),
            ("EG114403", 770, 678, 204, 40, 1.00),
        ])
        parsed = parse_card_fields(rows)
        self.assertEqual(parsed["surname"], "So")
        self.assertEqual(parsed["given_name"], "Long Yin")
        self.assertEqual(parsed["chinese_name"], "蘇朗言")
        self.assertEqual(
            parsed["programme"],
            "Higher Diploma in Computer and Electronic Engineering",
        )

    def test_permit_unlabelled_english_name_below_chinese(self):
        # Real permits print 苏朗言 then SO, LONG YIN with no field label.
        rows = _mk_ocr([
            ("港澳居民来往内地通行证", 300, 85, 880, 95, 0.99),
            ("苏朗言", 478, 232, 215, 70, 0.98),
            ("SO,LONG YIN", 472, 305, 440, 65, 0.94),
            ("出生日期", 477, 376, 134, 44, 1.00),
            ("2005.09.29", 478, 425, 362, 59, 0.99),
            ("证件号码", 490, 704, 136, 37, 1.00),
            ("H61089371", 500, 764, 396, 62, 1.00),
        ])
        parsed = parse_permit_fields(rows)
        self.assertEqual(parsed["chinese_name"], "苏朗言")
        self.assertEqual(parsed["english_name"], "SO LONG YIN")
        self.assertEqual(parsed["permit_number"], "H61089371")

    def test_permit_english_name_from_label_row(self):
        rows = _mk_ocr([
            ("往來內地通行證", 40, 20, 320, 40, 0.95),
            ("姓名 Name", 40, 150, 180, 30, 0.99),
            ("蘇朗言", 380, 150, 150, 30, 0.90),
            ("英文名 Eng", 40, 250, 200, 30, 0.98),
            ("SO LONG YIN", 380, 250, 300, 30, 0.95),
            ("證件號碼 No.", 40, 350, 220, 30, 0.95),
            ("H12345678", 380, 350, 200, 30, 0.99),
            ("出生日期 DOB", 40, 450, 240, 30, 0.99),
            ("1990-05-12", 380, 450, 220, 30, 0.99),
            ("性別 Sex", 40, 550, 160, 30, 0.98),
            ("男", 380, 550, 60, 30, 0.99),
        ])
        parsed = parse_permit_fields(rows)
        self.assertEqual(parsed["chinese_name"], "蘇朗言")
        self.assertEqual(parsed["english_name"], "SO LONG YIN")
        self.assertEqual(parsed["permit_number"], "H12345678")
        self.assertEqual(parsed["dob"], "1990-05-12")
        self.assertEqual(parsed["gender"], "M")


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
