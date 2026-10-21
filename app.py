"""Automatic Form Filling MVP — Streamlit UI.

Pipeline: upload / webcam → detect & crop → OCR + parse → editable registration form.
"""

from __future__ import annotations

import io
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detect import (
    bgr_to_rgb,
    decode_upload,
    detect_card_region,
    detect_passport_photo,
)
from ocr_parse import ocr_and_parse

st.set_page_config(
    page_title="Student Registration — Auto Form Fill",
    page_icon="🪪",
    layout="wide",
)

st.markdown(
    """
<style>
  .main-title { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; }
  .sub { color: #555; margin-bottom: 1.25rem; }
  .form-card {
    background: #fafafa;
    border: 1px solid #d0d0d0;
    border-radius: 8px;
    padding: 1.5rem 1.75rem;
  }
  .form-heading {
    font-size: 1.35rem;
    font-weight: 700;
    border-bottom: 2px solid #222;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
  }
  div[data-testid="stHorizontalBlock"] > div { align-items: flex-start; }
</style>
""",
    unsafe_allow_html=True,
)


def _init_state() -> None:
    defaults = {
        "surname": "",
        "given_name": "",
        "student_number": "",
        "programme": "",
        "photo_rgb": None,
        "card_rgb": None,
        "raw_lines": [],
        "ocr_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _bytes_to_bgr(uploaded) -> Optional[np.ndarray]:
    if uploaded is None:
        return None
    return decode_upload(uploaded.getvalue())


def _pil_from_rgb(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb)


def _set_photo_from_bgr(bgr: np.ndarray) -> None:
    st.session_state.photo_rgb = bgr_to_rgb(bgr)


def _run_ocr_on_card(card_bgr: np.ndarray) -> None:
    with st.spinner("Running OCR (first run may download models)…"):
        parsed = ocr_and_parse(card_bgr)
    st.session_state.surname = parsed.get("surname", "")
    st.session_state.given_name = parsed.get("given_name", "")
    st.session_state.student_number = parsed.get("student_number", "")
    st.session_state.programme = parsed.get("programme", "")
    st.session_state.raw_lines = parsed.get("raw_lines", [])
    st.session_state.card_rgb = bgr_to_rgb(card_bgr)
    st.session_state.ocr_done = True


def main() -> None:
    _init_state()

    st.markdown('<div class="main-title">Automatic Form Filling</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub">Upload a passport photo and IVE student card → detect, OCR, fill the Student Registration Form.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.subheader("1. Inputs")
        tab_upload, tab_cam = st.tabs(["Upload images", "Webcam (optional)"])

        with tab_upload:
            photo_file = st.file_uploader(
                "Passport-size photo",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                key="photo_up",
            )
            card_file = st.file_uploader(
                "Student card (full card or pre-cropped)",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                key="card_up",
            )
            scene_file = st.file_uploader(
                "Optional: one scene with both (auto-detect)",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                key="scene_up",
            )

            c1, c2 = st.columns(2)
            with c1:
                try_detect = st.checkbox("Auto-detect / crop regions", value=True)
            with c2:
                run_btn = st.button("Process & fill form", type="primary", use_container_width=True)

            if run_btn:
                photo_bgr = _bytes_to_bgr(photo_file)
                card_bgr = _bytes_to_bgr(card_file)
                scene_bgr = _bytes_to_bgr(scene_file)

                if scene_bgr is not None and try_detect:
                    card_crop = detect_card_region(scene_bgr)
                    photo_crop = detect_passport_photo(scene_bgr)
                    if card_crop is not None and card_bgr is None:
                        card_bgr = card_crop
                        st.info("Detected student-card region from scene.")
                    if photo_crop is not None and photo_bgr is None:
                        photo_bgr = photo_crop
                        st.info("Detected passport-photo region from scene.")

                if card_bgr is not None and try_detect:
                    refined = detect_card_region(card_bgr)
                    if refined is not None:
                        # Prefer refined crop only if meaningfully smaller (actual crop)
                        if refined.size < card_bgr.size * 0.95:
                            card_bgr = refined

                if photo_bgr is not None and try_detect:
                    refined_p = detect_passport_photo(photo_bgr)
                    if refined_p is not None and refined_p.size < photo_bgr.size * 0.95:
                        photo_bgr = refined_p

                if photo_bgr is not None:
                    _set_photo_from_bgr(photo_bgr)
                elif photo_file is None and scene_file is None:
                    st.warning("No passport photo provided — form photo box will stay empty.")

                if card_bgr is None:
                    st.error("Please upload a student card (or a scene containing one).")
                else:
                    _run_ocr_on_card(card_bgr)
                    st.success("OCR complete — check / edit the form on the right.")

        with tab_cam:
            st.caption("Uses the browser camera if available on this Mac.")
            cam_photo = st.camera_input("Capture passport photo")
            cam_card = st.camera_input("Capture student card")
            if st.button("Use webcam captures", use_container_width=True):
                if cam_photo is not None:
                    _set_photo_from_bgr(decode_upload(cam_photo.getvalue()))
                if cam_card is not None:
                    card_bgr = decode_upload(cam_card.getvalue())
                    if card_bgr is not None:
                        refined = detect_card_region(card_bgr)
                        if refined is not None:
                            card_bgr = refined
                        _run_ocr_on_card(card_bgr)
                        st.success("Webcam card processed.")
                if cam_photo is None and cam_card is None:
                    st.warning("Capture at least one image first.")

        if st.session_state.card_rgb is not None:
            st.markdown("**Card crop used for OCR**")
            st.image(st.session_state.card_rgb, use_container_width=True)
        if st.session_state.raw_lines:
            with st.expander("Raw OCR lines"):
                for line in st.session_state.raw_lines:
                    st.text(line)

    with right:
        st.subheader("2. Student registration form")
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="form-heading">Student registration form</div>', unsafe_allow_html=True)

        form_l, form_r = st.columns([1.4, 1])
        with form_r:
            st.markdown("**Photo**")
            if st.session_state.photo_rgb is not None:
                st.image(st.session_state.photo_rgb, use_container_width=True)
            else:
                st.markdown(
                    '<div style="border:2px dashed #999;height:180px;display:flex;'
                    'align-items:center;justify-content:center;color:#888;">No photo</div>',
                    unsafe_allow_html=True,
                )
            manual_photo = st.file_uploader(
                "Replace photo",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                key="manual_photo",
            )
            if manual_photo is not None:
                bgr = decode_upload(manual_photo.getvalue())
                if bgr is not None:
                    _set_photo_from_bgr(bgr)

        with form_l:
            st.text_input("Surname", key="surname")
            st.text_input("Given name", key="given_name")
            st.text_input("Student number", key="student_number")
            st.text_input("Programme undertaking", key="programme")

        st.markdown("</div>", unsafe_allow_html=True)

        # Printable / export HTML
        photo_data_uri = ""
        if st.session_state.photo_rgb is not None:
            buf = io.BytesIO()
            _pil_from_rgb(st.session_state.photo_rgb).save(buf, format="JPEG", quality=90)
            import base64

            photo_data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Student registration form</title>
<style>
  body {{ font-family: Georgia, 'Times New Roman', serif; margin: 2rem; }}
  h1 {{ border-bottom: 2px solid #111; padding-bottom: 0.4rem; }}
  .wrap {{ display: flex; gap: 2rem; }}
  .fields {{ flex: 1; }}
  .photo {{ width: 160px; height: 200px; border: 1px solid #333; object-fit: cover; background: #eee; }}
  .row {{ margin: 1rem 0; }}
  .label {{ font-weight: bold; display: inline-block; width: 11rem; }}
  .val {{ border-bottom: 1px solid #333; min-width: 16rem; display: inline-block; }}
</style></head><body>
<h1>Student registration form</h1>
<div class="wrap">
  <div class="fields">
    <div class="row"><span class="label">Surname:</span> <span class="val">{st.session_state.surname}</span></div>
    <div class="row"><span class="label">Given name:</span> <span class="val">{st.session_state.given_name}</span></div>
    <div class="row"><span class="label">Student number:</span> <span class="val">{st.session_state.student_number}</span></div>
    <div class="row"><span class="label">Programme undertaking:</span> <span class="val">{st.session_state.programme}</span></div>
  </div>
  {"<img class='photo' src='" + photo_data_uri + "' alt='photo'/>" if photo_data_uri else "<div class='photo'></div>"}
</div>
</body></html>"""

        st.download_button(
            "Download printable HTML",
            data=html.encode("utf-8"),
            file_name="student_registration_form.html",
            mime="text/html",
            use_container_width=True,
        )

    st.caption(
        "MVP: OpenCV crop + EasyOCR parse. Edit any field before exporting. "
        "First OCR run may take a minute while models download."
    )


if __name__ == "__main__":
    main()
