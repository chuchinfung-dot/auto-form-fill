# Student Document Pre-check

Upload a **Home Return Permit (回鄉證)** and an **IVE student card** → OpenCV crop → EasyOCR parse (English + Traditional Chinese) → compare shared fields → editable **Student Registration Form** → printable HTML export.

## Features

- OCR two cards:
  - Student card → surname, given name, student number, programme, Chinese name.
  - Home Return Permit → Chinese name, English name, permit number, date of birth, gender.
- OpenCV preprocessing for OCR (grayscale, denoise, CLAHE contrast, upscale).
- Compare shared fields (English name, Chinese name) and flag **match / mismatch / missing**.
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

First OCR run may download EasyOCR models (English + Traditional Chinese) and take a minute.

## Layout

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI (upload / webcam / form / comparison) |
| `detect.py` | OpenCV card & passport-photo crop |
| `ocr_parse.py` | EasyOCR + field parsing (student card + permit) |
| `compare.py` | Shared-field comparison + completeness summary |
| `test_compare.py` | Unit tests for comparison & parsing |
| `requirements.txt` | Dependencies |

## Test

```bash
python3 -m unittest test_compare -v
```
