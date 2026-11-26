"""OpenCV helpers: find and crop card regions, and prepare them for OCR."""

from __future__ import annotations

import cv2
import numpy as np


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = int(max(width_a, width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = int(max(height_a, height_b))
    if max_w < 10 or max_h < 10:
        return image
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_w, max_h))


def _candidate_quads(image: np.ndarray, min_area_ratio: float = 0.02):
    h, w = image.shape[:2]
    area = float(h * w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        cnt_area = cv2.contourArea(approx)
        if cnt_area < min_area_ratio * area:
            continue
        quads.append(approx.reshape(4, 2))
    return quads


def detect_card_region(image: np.ndarray) -> np.ndarray | None:
    """Return a deskewed crop of the largest card-like rectangle, or None."""
    quads = _candidate_quads(image, min_area_ratio=0.05)
    if not quads:
        return None
    best = None
    best_score = -1.0
    for quad in quads:
        crop = _four_point_transform(image, quad)
        ch, cw = crop.shape[:2]
        if ch < 40 or cw < 40:
            continue
        aspect = max(cw, ch) / max(1, min(cw, ch))
        # IVE student cards are roughly credit-card shaped (~1.5–1.8)
        if 1.2 <= aspect <= 2.2:
            score = cw * ch / (1.0 + abs(aspect - 1.58))
            if score > best_score:
                best_score = score
                best = crop
    return best


def preprocess_for_ocr(image: np.ndarray, min_width: int = 1000) -> np.ndarray:
    """Grayscale + denoise + CLAHE contrast, upscaled if small.

    Card images are often low-contrast and small; this gives OCR a cleaner,
    larger input. Returns a single-channel image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    h, w = gray.shape[:2]
    if w < min_width:
        scale = min_width / float(max(w, 1))
        gray = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_upload(file_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img
