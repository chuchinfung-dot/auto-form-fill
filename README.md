# Automatic Form Filling MVP

Upload a passport photo and an IVE student card → OpenCV crop → EasyOCR parse → editable **Student Registration Form**.

## Setup

```bash
cd auto_form_fill
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

First OCR run may download EasyOCR models and take a minute.

## Layout

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI (upload / webcam / form) |
| `detect.py` | OpenCV card & passport-photo crop |
| `ocr_parse.py` | EasyOCR + field parsing |
| `requirements.txt` | Dependencies |
