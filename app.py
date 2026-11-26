"""Student Document Pre-check — Streamlit UI.

Upload a Home Return Permit and an IVE student card → OCR → compare shared
fields → editable form + discrepancy/completeness summary.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import streamlit as st

from compare import STATUS_LABELS, build_summary, compare_documents
from detect import bgr_to_rgb, decode_upload, detect_card_region
from ocr_parse import ocr_and_parse, ocr_and_parse_permit

st.set_page_config(page_title="Student Document Pre-check", page_icon="🪪", layout="wide")

st.markdown(
    """
<style>
  .main-title { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; }
  .sub { color: #555; margin-bottom: 1.25rem; }
  .form-heading { font-size: 1.35rem; font-weight: 700; border-bottom: 2px solid #222; padding-bottom: 0.4rem; margin-bottom: 1rem; }
  .ok { color: #0a7d33; font-weight: 700; }
  .bad { color: #b00020; font-weight: 700; }
  div[data-testid="stHorizontalBlock"] > div { align-items: flex-start; }
</style>
""",
    unsafe_allow_html=True,
)

FIELD_LABELS = {
    "surname": "Surname",
    "given_name": "Given name",
    "student_number": "Student number",
    "programme": "Programme",
    "chinese_name": "Chinese name",
    "english_name": "English name",
    "permit_number": "Permit number",
    "dob": "Date of birth",
    "gender": "Gender",
}

DEFAULTS: dict[str, Any] = {
    "surname": "",
    "given_name": "",
    "student_number": "",
    "programme": "",
    "student_chinese_name": "",
    "permit_chinese_name": "",
    "permit_english_name": "",
    "permit_number": "",
    "dob": "",
    "gender": "",
    "card_rgb": None,
    "permit_rgb": None,
    "raw_lines_card": [],
    "raw_lines_permit": [],
    "confidence_card": {},
    "confidence_permit": {},
    "low_conf_card": [],
    "low_conf_permit": [],
    "ocr_done": False,
}


def _init_state() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _bytes_to_bgr(uploaded) -> Optional[np.ndarray]:
    if uploaded is None:
        return None
    return decode_upload(uploaded.getvalue())


def _student_record() -> dict[str, Any]:
    return {
        "surname": st.session_state.surname,
        "given_name": st.session_state.given_name,
        "student_number": st.session_state.student_number,
        "programme": st.session_state.programme,
        "chinese_name": st.session_state.student_chinese_name,
    }


def _permit_record() -> dict[str, Any]:
    return {
        "chinese_name": st.session_state.permit_chinese_name,
        "english_name": st.session_state.permit_english_name,
        "permit_number": st.session_state.permit_number,
        "dob": st.session_state.dob,
        "gender": st.session_state.gender,
    }


def _run_ocr_on_cards(card_bgr, permit_bgr) -> None:
    with st.spinner("Running OCR (first run may download models)…"):
        card = ocr_and_parse(card_bgr)
        permit = ocr_and_parse_permit(permit_bgr) if permit_bgr is not None else {}

    st.session_state.surname = card.get("surname", "")
    st.session_state.given_name = card.get("given_name", "")
    st.session_state.student_number = card.get("student_number", "")
    st.session_state.programme = card.get("programme", "")
    st.session_state.student_chinese_name = card.get("chinese_name", "")
    st.session_state.raw_lines_card = card.get("raw_lines", [])
    st.session_state.confidence_card = card.get("confidence", {})
    st.session_state.low_conf_card = card.get("low_confidence", [])
    st.session_state.card_rgb = bgr_to_rgb(card_bgr)

    if permit_bgr is not None:
        st.session_state.permit_chinese_name = permit.get("chinese_name", "")
        st.session_state.permit_english_name = permit.get("english_name", "")
        st.session_state.permit_number = permit.get("permit_number", "")
        st.session_state.dob = permit.get("dob", "")
        st.session_state.gender = permit.get("gender", "")
        st.session_state.raw_lines_permit = permit.get("raw_lines", [])
        st.session_state.confidence_permit = permit.get("confidence", {})
        st.session_state.low_conf_permit = permit.get("low_confidence", [])
        st.session_state.permit_rgb = bgr_to_rgb(permit_bgr)

    st.session_state.ocr_done = True


def _render_inputs() -> None:
    st.subheader("1. Inputs")
    tab_upload, tab_cam = st.tabs(["Upload images", "Webcam (optional)"])

    with tab_upload:
        card_file = st.file_uploader(
            "Student card (full card or pre-cropped)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="card_up",
        )
        permit_file = st.file_uploader(
            "Home Return Permit (回鄉證)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="permit_up",
        )
        scene_file = st.file_uploader(
            "Optional: one scene with both cards (auto-detect)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="scene_up",
        )

        c1, c2 = st.columns(2)
        with c1:
            try_detect = st.checkbox("Auto-detect / crop regions", value=True)
        with c2:
            run_btn = st.button("Process & fill form", type="primary", use_container_width=True)

        if run_btn:
            card_bgr = _bytes_to_bgr(card_file)
            permit_bgr = _bytes_to_bgr(permit_file)
            scene_bgr = _bytes_to_bgr(scene_file)

            if scene_bgr is not None and try_detect and card_bgr is None:
                card_crop = detect_card_region(scene_bgr)
                if card_crop is not None:
                    card_bgr = card_crop
                    st.info("Detected student-card region from scene.")

            for name, key in (("student card", "card"), ("permit", "permit")):
                img = card_bgr if key == "card" else permit_bgr
                if img is None or not try_detect:
                    continue
                refined = detect_card_region(img)
                if refined is not None and refined.size < img.size * 0.95:
                    if key == "card":
                        card_bgr = refined
                    else:
                        permit_bgr = refined

            if card_bgr is None:
                st.error("Please upload a student card (or a scene containing one).")
            else:
                _run_ocr_on_cards(card_bgr, permit_bgr)
                if permit_bgr is None:
                    st.warning("No Home Return Permit provided — comparison needs both cards.")
                st.success("OCR complete — review / edit on the right.")

    with tab_cam:
        st.caption("Uses the browser camera if available on this Mac.")
        cam_card = st.camera_input("Capture student card")
        cam_permit = st.camera_input("Capture Home Return Permit")
        if st.button("Use webcam captures", use_container_width=True):
            card_bgr = None
            permit_bgr = None
            if cam_card is not None:
                card_bgr = decode_upload(cam_card.getvalue())
                refined = detect_card_region(card_bgr)
                if refined is not None:
                    card_bgr = refined
            if cam_permit is not None:
                permit_bgr = decode_upload(cam_permit.getvalue())
                refined_p = detect_card_region(permit_bgr)
                if refined_p is not None:
                    permit_bgr = refined_p
            if card_bgr is not None:
                _run_ocr_on_cards(card_bgr, permit_bgr)
                st.success("Webcam card(s) processed.")
            else:
                st.warning("Capture a student card first.")

    if st.session_state.card_rgb is not None:
        st.markdown("**Student-card crop used for OCR**")
        st.image(st.session_state.card_rgb, use_container_width=True)
    if st.session_state.permit_rgb is not None:
        st.markdown("**Permit crop used for OCR**")
        st.image(st.session_state.permit_rgb, use_container_width=True)
    if st.session_state.raw_lines_card:
        with st.expander("Raw OCR lines — student card"):
            for line in st.session_state.raw_lines_card:
                st.text(line)
    if st.session_state.raw_lines_permit:
        with st.expander("Raw OCR lines — permit"):
            for line in st.session_state.raw_lines_permit:
                st.text(line)


def _render_form() -> None:
    st.subheader("2. Student registration form")
    st.markdown('<div class="form-heading">Student registration form</div>', unsafe_allow_html=True)

    st.text_input("Surname", key="surname")
    st.text_input("Given name", key="given_name")
    st.text_input("Student number", key="student_number")
    st.text_input("Programme undertaking", key="programme")
    st.text_input("Chinese name (student card)", key="student_chinese_name")

    st.markdown(
        '<div class="form-heading" style="margin-top:1rem;">Home Return Permit</div>',
        unsafe_allow_html=True,
    )
    pl, pr = st.columns(2)
    with pl:
        st.text_input("Chinese name (permit)", key="permit_chinese_name")
        st.text_input("English name (permit)", key="permit_english_name")
    with pr:
        st.text_input("Permit number", key="permit_number")
        st.text_input("Date of birth", key="dob")
        st.text_input("Gender", key="gender")

    chinese_name = st.session_state.permit_chinese_name or st.session_state.student_chinese_name
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Student registration form</title>
<style>
  body {{ font-family: Georgia, 'Times New Roman', serif; margin: 2rem; }}
  h1 {{ border-bottom: 2px solid #111; padding-bottom: 0.4rem; }}
  .row {{ margin: 1rem 0; }}
  .label {{ font-weight: bold; display: inline-block; width: 11rem; }}
  .val {{ border-bottom: 1px solid #333; min-width: 16rem; display: inline-block; }}
</style></head><body>
<h1>Student registration form</h1>
<div class="row"><span class="label">Surname:</span> <span class="val">{st.session_state.surname}</span></div>
<div class="row"><span class="label">Given name:</span> <span class="val">{st.session_state.given_name}</span></div>
<div class="row"><span class="label">Chinese name:</span> <span class="val">{chinese_name}</span></div>
<div class="row"><span class="label">Student number:</span> <span class="val">{st.session_state.student_number}</span></div>
<div class="row"><span class="label">Programme:</span> <span class="val">{st.session_state.programme}</span></div>
<div class="row"><span class="label">Permit number:</span> <span class="val">{st.session_state.permit_number}</span></div>
</body></html>"""

    st.download_button(
        "Download printable HTML",
        data=html.encode("utf-8"),
        file_name="student_registration_form.html",
        mime="text/html",
        use_container_width=True,
    )


def _render_comparison() -> None:
    if not st.session_state.ocr_done:
        return

    st.subheader("3. Field comparison & summary")

    low_lines = [
        f"Student card — {FIELD_LABELS.get(k, k)} "
        f"({st.session_state.confidence_card.get(k, 0):.0%} confidence)"
        for k in st.session_state.low_conf_card
    ] + [
        f"Permit — {FIELD_LABELS.get(k, k)} "
        f"({st.session_state.confidence_permit.get(k, 0):.0%} confidence)"
        for k in st.session_state.low_conf_permit
    ]
    if low_lines:
        st.warning(
            "Low OCR confidence — please check these values against the images:\n\n"
            + "\n".join(f"- {item}" for item in low_lines)
        )

    student = _student_record()
    permit = _permit_record()
    results = compare_documents(student, permit)
    summary = build_summary(student, permit, results)

    rows = [
        {
            "Field": r["field"],
            "Student card": r["student"],
            "Permit": r["permit"],
            "Status": STATUS_LABELS.get(r["status"], r["status"]),
        }
        for r in results
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if summary["all_consistent"]:
        st.markdown(
            '<span class="ok">✅ All shared fields are consistent.</span>',
            unsafe_allow_html=True,
        )
    else:
        if summary["mismatch_count"]:
            st.markdown(
                f'<span class="bad">⚠️ {summary["mismatch_count"]} field(s) do not match — review before confirming.</span>',
                unsafe_allow_html=True,
            )
        if summary["missing_fields"]:
            st.markdown("**Missing / unreadable fields:**")
            for m in summary["missing_fields"]:
                st.markdown(f"- {m}")


def main() -> None:
    _init_state()

    st.markdown('<div class="main-title">Student Document Pre-check</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub">Home Return Permit + IVE student card → OCR → compare → editable form.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        _render_inputs()
    with right:
        _render_form()

    _render_comparison()

    st.caption(
        "MVP: OpenCV crop + EasyOCR (EN + Traditional Chinese) + field comparison. "
        "Edit any field before exporting. First OCR run may take a minute while models download."
    )


if __name__ == "__main__":
    main()
