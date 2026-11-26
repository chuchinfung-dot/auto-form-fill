# Student Document Pre-check

Upload a **Home Return Permit (回鄉證)** and an **IVE student card** → OpenCV preprocess → RapidOCR (PP-OCRv4) → compare shared fields → editable **Student Registration Form** → printable HTML export.

## Features

- OCR two cards with RapidOCR (PP-OCRv4, ONNX) — no PyTorch required:
  - Student card → surname, given name, student number, programme, Chinese name.
  - Home Return Permit → Chinese name, English name, permit number, date of birth, gender.
- OpenCV preprocessing for OCR: denoise, 2x upscale, CLAHE contrast.
- Compare shared fields (English name, Chinese name) and flag **match / mismatch / missing**.
- Treat Traditional and Simplified spellings of the same name as a match (via `zhconv`).
- Flag low-confidence OCR values for human review.
- Completeness summary listing missing or unreadable fields.
- Editable form + printable HTML export.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

Open http://127.0.0.1:8502

OCR runs on CPU through onnxruntime; models ship with the package.

## Layout

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI (upload / webcam / form / comparison) |
| `detect.py` | OpenCV card crop + OCR preprocessing |
| `ocr_parse.py` | RapidOCR + field parsing (student card + permit) |
| `compare.py` | Shared-field comparison + completeness summary |
| `test_compare.py` | Unit tests for comparison & parsing |
| `requirements.txt` | Dependencies |

## Test

```bash
source .venv/bin/activate
python -m unittest test_compare -v
```
