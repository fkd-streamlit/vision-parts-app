# -*- coding: utf-8 -*-
"""
刻印・ロゴの OCR と型番パターンマッチ。
黒ベース製品向けに複数の前処理バリアントを試し、CNN結果と融合する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from config import (
    CLASS_NAMES,
    FUSION_AGREE_BOOST,
    OCR_MODE,
    OCR_OVERRIDE_THRESHOLD,
    OCR_PATTERNS,
)

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import easyocr

    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False


@dataclass
class OCRDetection:
    text: str
    confidence: float
    variant: str


@dataclass
class OCRClassScore:
    class_name: str
    score: float
    matched: List[str] = field(default_factory=list)


@dataclass
class OCRResult:
    available: bool
    error: Optional[str]
    detections: List[OCRDetection]
    class_scores: Dict[str, float]
    best_class: Optional[str]
    best_score: float
    combined_text: str


@dataclass
class FusedPrediction:
    label: str
    confidence: float
    method: str  # cnn | ocr | fusion_agree | fusion_ocr | fusion_cnn
    cnn_label: str
    cnn_conf: float
    ocr_label: Optional[str]
    ocr_conf: float
    ocr_text: str


_reader = None


def ocr_dependencies_ok() -> Tuple[bool, str]:
    if not HAS_EASYOCR:
        return False, "easyocr が未インストールです: pip install easyocr opencv-python-headless"
    if not HAS_CV2:
        return False, "opencv-python-headless が未インストールです"
    return True, ""


def get_ocr_reader():
    """EasyOCR Reader（初回のみモデルDL）。"""
    global _reader
    ok, msg = ocr_dependencies_ok()
    if not ok:
        raise RuntimeError(msg)
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def _pil_to_bgr(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _resize_min_side(im: Image.Image, min_side: int = 640) -> Image.Image:
    w, h = im.size
    scale = min_side / min(w, h)
    if scale <= 1.0:
        return im
    return im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def preprocess_variants(pil_rgb: Image.Image, mode: str = OCR_MODE) -> List[Tuple[str, np.ndarray]]:
    """黒製品の刻印向けに複数の前処理画像を生成（RGB uint8）。"""
    min_side = 960 if mode == "thorough" else 720
    base = _resize_min_side(pil_rgb.convert("RGB"), min_side)
    variants: List[Tuple[str, np.ndarray]] = []

    if not HAS_CV2:
        variants.append(("original", np.array(base)))
        variants.append(("gray_contrast", np.array(base.convert("L"))))
        return variants

    rgb = np.array(base)
    bgr = _pil_to_bgr(rgb)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(gray)

    variants.append(("clahe", cv2.cvtColor(g_clahe, cv2.COLOR_GRAY2RGB)))
    _, otsu = cv2.threshold(g_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(otsu) > 127:
        otsu = cv2.bitwise_not(otsu)
    variants.append(("otsu_inv", cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)))

    if mode == "thorough":
        variants.insert(0, ("original", rgb))
        adaptive = cv2.adaptiveThreshold(
            g_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
        )
        if np.mean(adaptive) > 127:
            adaptive = cv2.bitwise_not(adaptive)
        variants.append(("adaptive_inv", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)))
        edges = cv2.Canny(g_clahe, 40, 120)
        variants.append(("edges", cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)))
        hi = ImageEnhance.Contrast(base).enhance(2.2)
        hi = ImageEnhance.Sharpness(hi).enhance(2.0)
        variants.append(("pil_hi_contrast", np.array(hi)))
        inv = ImageOps.invert(base.convert("L")).convert("RGB")
        variants.append(("pil_invert", np.array(inv)))

    return variants


def _normalize_ocr_text(text: str) -> str:
    t = text.upper()
    t = re.sub(r"[^A-Z0-9\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_DIGIT_CLASS = {
    "7100": "FCR7100",
    "8100": "FCR8100",
    "9200": "FCR9200",
}


def _compact_alnum(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _score_digit_hints(compact: str) -> Dict[str, float]:
    """OCR誤認識を考慮し 7100 / 8100 / 9200 を探す。"""
    scores: Dict[str, float] = {c: 0.0 for c in CLASS_NAMES}
    if not compact:
        return scores

    for digits, cls in _DIGIT_CLASS.items():
        if digits in compact:
            scores[cls] = max(scores[cls], 0.62)
        if f"R{digits}" in compact or f"FCR{digits}" in compact or f"FC{digits}" in compact:
            scores[cls] = max(scores[cls], 0.88)

    # 7I00 / 7l00 等
    fuzzy = [
        (r"7[1Il|][0O0]{2}", "FCR7100", 0.58),
        (r"8[1Il|][0O0]{2}", "FCR8100", 0.58),
        (r"9[2Zz][0O0]{2}", "FCR9200", 0.58),
    ]
    for pat, cls, w in fuzzy:
        if re.search(pat, compact):
            scores[cls] = max(scores[cls], w)
    return scores


def score_text_against_patterns(text: str) -> List[OCRClassScore]:
    """読み取り文字列から各クラスのスコアを算出。"""
    norm = _normalize_ocr_text(text)
    if not norm:
        return []

    compact = _compact_alnum(norm)
    digit_hints = _score_digit_hints(compact)

    results: List[OCRClassScore] = []
    for cls in CLASS_NAMES:
        matched: List[str] = []
        best = digit_hints.get(cls, 0.0)
        if best > 0:
            matched.append(f"digits:{cls}")

        for pattern, weight in OCR_PATTERNS[cls]:
            m = re.search(pattern, norm, re.IGNORECASE)
            if m:
                matched.append(m.group(0))
                best = max(best, weight)

        if best > 0:
            results.append(OCRClassScore(class_name=cls, score=best, matched=matched))
    results.sort(key=lambda x: -x.score)
    return results


def _spatial_crops(pil_rgb: Image.Image, mode: str = OCR_MODE) -> List[Tuple[str, Image.Image]]:
    """刻印が写りやすい領域を切り出す。"""
    w, h = pil_rgb.size
    crops = [("full", pil_rgb)]
    if mode == "fast":
        crops.append(
            ("center", pil_rgb.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))))
        )
        return crops
    boxes = [
        ("center", (int(w * 0.12), int(h * 0.12), int(w * 0.88), int(h * 0.88))),
        ("arm_left", (0, int(h * 0.15), int(w * 0.55), int(h * 0.85))),
        ("arm_right", (int(w * 0.45), int(h * 0.15), w, int(h * 0.85))),
        ("lower", (int(w * 0.1), int(h * 0.45), int(w * 0.9), h)),
    ]
    for name, box in boxes:
        crops.append((name, pil_rgb.crop(box)))
    return crops


def _run_easyocr_on_variant(
    reader,
    rgb: np.ndarray,
    variant_name: str,
    *,
    allowlist: Optional[str] = None,
) -> List[OCRDetection]:
    out: List[OCRDetection] = []
    kwargs = {"detail": 1, "paragraph": False}
    if allowlist:
        kwargs["allowlist"] = allowlist
    try:
        raw = reader.readtext(rgb, **kwargs)
    except Exception:
        return out
    for item in raw:
        if len(item) < 3:
            continue
        _bbox, text, conf = item[0], item[1], float(item[2])
        if not str(text).strip():
            continue
        out.append(OCRDetection(text=str(text).strip(), confidence=conf, variant=variant_name))
    return out


def recognize_from_image(
    pil_rgb: Image.Image,
    reader=None,
    mode: str = OCR_MODE,
) -> OCRResult:
    """1枚の画像から OCR + パターンマッチ。"""
    ok, msg = ocr_dependencies_ok()
    if not ok:
        return OCRResult(
            available=False,
            error=msg,
            detections=[],
            class_scores={},
            best_class=None,
            best_score=0.0,
            combined_text="",
        )

    if reader is None:
        reader = get_ocr_reader()

    allowlist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- "
    all_detections: List[OCRDetection] = []
    base = _resize_min_side(pil_rgb.convert("RGB"), 800)

    if mode == "fast":
        w, h = base.size
        center = base.crop((int(w * 0.12), int(h * 0.12), int(w * 0.88), int(h * 0.88)))
        for vname, arr in preprocess_variants(center, mode):
            if vname == "otsu_inv":
                all_detections.extend(
                    _run_easyocr_on_variant(
                        reader, arr, "center/otsu+alnum", allowlist=allowlist
                    )
                )
    else:
        for cname, crop in _spatial_crops(base, mode):
            variants = preprocess_variants(crop, mode)
            for vname, arr in variants:
                tag = f"{cname}/{vname}"
                all_detections.extend(_run_easyocr_on_variant(reader, arr, tag))
            for vname, arr in variants:
                if vname == "otsu_inv":
                    tag = f"{cname}/{vname}+alnum"
                    all_detections.extend(
                        _run_easyocr_on_variant(reader, arr, tag, allowlist=allowlist)
                    )

    combined = " | ".join(d.text for d in all_detections)
    # 全文 + 各検出行を個別にスコア化し最大を採用
    texts_to_score = [combined] + [d.text for d in all_detections]
    agg: Dict[str, float] = {c: 0.0 for c in CLASS_NAMES}
    for t in texts_to_score:
        for sc in score_text_against_patterns(t):
            det_boost = 0.0
            for d in all_detections:
                if d.text in t or t in d.text:
                    det_boost = max(det_boost, d.confidence * 0.15)
            agg[sc.class_name] = max(agg[sc.class_name], min(1.0, sc.score + det_boost))

    best_class = None
    best_score = 0.0
    if agg:
        best_class = max(agg, key=agg.get)
        best_score = agg[best_class]
        if best_score <= 0:
            best_class = None

    return OCRResult(
        available=True,
        error=None,
        detections=all_detections,
        class_scores=agg,
        best_class=best_class,
        best_score=best_score,
        combined_text=combined,
    )


def fuse_cnn_and_ocr(
    cnn_probs: np.ndarray,
    class_names: List[str],
    ocr: OCRResult,
) -> FusedPrediction:
    """CNN softmax と OCR スコアを融合。"""
    cnn_idx = int(np.argmax(cnn_probs))
    cnn_label = class_names[cnn_idx]
    cnn_conf = float(cnn_probs[cnn_idx])

    ocr_label = ocr.best_class
    ocr_conf = float(ocr.best_score) if ocr.best_class else 0.0
    ocr_text = ocr.combined_text or ""

    if not ocr.available or ocr_label is None:
        return FusedPrediction(
            label=cnn_label,
            confidence=cnn_conf,
            method="cnn",
            cnn_label=cnn_label,
            cnn_conf=cnn_conf,
            ocr_label=None,
            ocr_conf=0.0,
            ocr_text=ocr_text,
        )

    if ocr_conf >= OCR_OVERRIDE_THRESHOLD:
        if ocr_label == cnn_label:
            conf = min(1.0, max(cnn_conf, ocr_conf) + FUSION_AGREE_BOOST)
            method = "fusion_agree"
        else:
            conf = ocr_conf
            method = "fusion_ocr"
        return FusedPrediction(
            label=ocr_label,
            confidence=conf,
            method=method,
            cnn_label=cnn_label,
            cnn_conf=cnn_conf,
            ocr_label=ocr_label,
            ocr_conf=ocr_conf,
            ocr_text=ocr_text,
        )

    if ocr_label == cnn_label and ocr_conf >= 0.4:
        conf = min(1.0, cnn_conf + ocr_conf * 0.25)
        return FusedPrediction(
            label=cnn_label,
            confidence=conf,
            method="fusion_agree",
            cnn_label=cnn_label,
            cnn_conf=cnn_conf,
            ocr_label=ocr_label,
            ocr_conf=ocr_conf,
            ocr_text=ocr_text,
        )

    if cnn_conf >= 0.55 and ocr_label != cnn_label and ocr_conf < 0.55:
        return FusedPrediction(
            label=cnn_label,
            confidence=cnn_conf,
            method="fusion_cnn",
            cnn_label=cnn_label,
            cnn_conf=cnn_conf,
            ocr_label=ocr_label,
            ocr_conf=ocr_conf,
            ocr_text=ocr_text,
        )

    # OCR が弱いが CNN も低い → やや OCR を参考
    if ocr_conf > cnn_conf and ocr_conf >= 0.5:
        return FusedPrediction(
            label=ocr_label,
            confidence=ocr_conf,
            method="fusion_ocr",
            cnn_label=cnn_label,
            cnn_conf=cnn_conf,
            ocr_label=ocr_label,
            ocr_conf=ocr_conf,
            ocr_text=ocr_text,
        )

    return FusedPrediction(
        label=cnn_label,
        confidence=cnn_conf,
        method="cnn",
        cnn_label=cnn_label,
        cnn_conf=cnn_conf,
        ocr_label=ocr_label,
        ocr_conf=ocr_conf,
        ocr_text=ocr_text,
    )
