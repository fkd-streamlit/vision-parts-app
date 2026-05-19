# -*- coding: utf-8 -*-
"""
ocr_engine.py
刻印・ロゴの OCR と型番パターンマッチ（A：刻印優先OCR対応版）

【ポイント】
- target="stamp" : 刻印優先（背景文字を拾いにくいROI + 強いallowlist）
- target="general": 全体文字（従来のfull/center/lower）
- 回転スキャン（0/90/180/270）で向き不明を吸収
- 早期終了（best_scoreが十分なら回転スキャンを打ち切る）
- 数字誤読補正（O↔0, Z↔2, I/L↔1, S↔5, B↔8）を digits 判定側にだけ適用
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance

from config import (
    CLASS_NAMES,
    FUSION_AGREE_BOOST,
    OCR_MODE,
    OCR_OVERRIDE_THRESHOLD,
    OCR_PATTERNS,
)

# -----------------------------
# Optional deps
# -----------------------------
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    import easyocr
    HAS_EASYOCR = True
except Exception:
    HAS_EASYOCR = False


# -----------------------------
# Data classes
# -----------------------------
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
    method: str  # cnn | fusion_agree | fusion_ocr | fusion_cnn
    cnn_label: str
    cnn_conf: float
    ocr_label: Optional[str]
    ocr_conf: float
    ocr_text: str


# -----------------------------
# Globals
# -----------------------------
_reader = None

OCR_MAX_SIDE_FAST = 896
OCR_MAX_SIDE_THOROUGH = 1024

ROTATION_DEGREES = [0, 90, 180, 270]
FALLBACK_SMALL_TILTS = [-15, 15]  # 必要時のみ

# STAMP（刻印）向け：強いallowlist
ALLOWLIST_STAMP = "0123456789FCR- "
# GENERAL（全体）向け：制限なし（ロゴ等も拾える）
ALLOWLIST_GENERAL = None

# 早期終了（OCRが十分強く決まったら回転を打ち切る）
EARLY_STOP_SCORE = 0.78


# -----------------------------
# Dependencies
# -----------------------------
def ocr_dependencies_ok() -> Tuple[bool, str]:
    if not HAS_EASYOCR:
        return False, "easyocr が未インストールです: pip install easyocr opencv-python-headless"
    if not HAS_CV2:
        return False, "opencv-python-headless が未インストールです"
    return True, ""


def get_ocr_reader():
    """EasyOCR Reader（初回のみモデルDL）。Cloud安定化のため gpu=False 固定。"""
    global _reader
    ok, msg = ocr_dependencies_ok()
    if not ok:
        raise RuntimeError(msg)
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


# -----------------------------
# Image helpers
# -----------------------------
def _resize_max_side(im: Image.Image, max_side: int) -> Image.Image:
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    scale = max_side / m
    return im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def _pil_to_rgb_uint8(im: Image.Image) -> np.ndarray:
    if im.mode != "RGB":
        im = im.convert("RGB")
    arr = np.array(im)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _rgb_to_gray(arr_rgb: np.ndarray) -> np.ndarray:
    if not HAS_CV2:
        r = arr_rgb[..., 0].astype(np.float32)
        g = arr_rgb[..., 1].astype(np.float32)
        b = arr_rgb[..., 2].astype(np.float32)
        y = (0.299 * r + 0.587 * g + 0.114 * b).clip(0, 255).astype(np.uint8)
        return y
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _clahe_gray(gray: np.ndarray) -> np.ndarray:
    if not HAS_CV2:
        return gray
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _otsu_inv(gray: np.ndarray) -> np.ndarray:
    if not HAS_CV2:
        thr = int(np.mean(gray))
        bw = (gray > thr).astype(np.uint8) * 255
        if np.mean(bw) > 127:
            bw = 255 - bw
        return bw
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw) > 127:
        bw = cv2.bitwise_not(bw)
    return bw


# -----------------------------
# Preprocess variants (軽量)
# -----------------------------
def preprocess_variants(pil_rgb: Image.Image, mode: str = OCR_MODE) -> List[Tuple[str, np.ndarray]]:
    max_side = OCR_MAX_SIDE_THOROUGH if mode == "thorough" else OCR_MAX_SIDE_FAST
    base = _resize_max_side(pil_rgb.convert("RGB"), max_side=max_side)

    rgb = _pil_to_rgb_uint8(base)
    variants: List[Tuple[str, np.ndarray]] = [("original", rgb)]

    gray = _rgb_to_gray(rgb)
    g_clahe = _clahe_gray(gray)
    clahe_rgb = np.stack([g_clahe, g_clahe, g_clahe], axis=-1)
    variants.append(("clahe", clahe_rgb))

    bw = _otsu_inv(g_clahe)
    bw_rgb = np.stack([bw, bw, bw], axis=-1)
    variants.append(("otsu_inv", bw_rgb))

    if mode == "thorough":
        hi = ImageEnhance.Contrast(base).enhance(2.0)
        hi = ImageEnhance.Sharpness(hi).enhance(1.6)
        variants.append(("pil_hi_contrast", _pil_to_rgb_uint8(hi)))

    return variants[:4]


# -----------------------------
# Text scoring
# -----------------------------
def _normalize_ocr_text(text: str) -> str:
    t = (text or "").upper()
    t = re.sub(r"[^A-Z0-9\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _compact_alnum(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _digit_fuzzy_compact(compact: str) -> str:
    """数字誤読補正（digits判定のみに適用）"""
    if not compact:
        return compact
    t = compact
    t = t.replace("O", "0").replace("Q", "0")
    t = t.replace("Z", "2")
    t = t.replace("S", "5")
    t = t.replace("I", "1").replace("L", "1")
    t = t.replace("B", "8")
    return t


_DIGIT_CLASS = {
    "7100": "FCR7100",
    "8100": "FCR8100",
    "9200": "FCR9200",
}


def _score_digit_hints(compact: str) -> Dict[str, float]:
    scores: Dict[str, float] = {c: 0.0 for c in CLASS_NAMES}
    if not compact:
        return scores

    for digits, cls in _DIGIT_CLASS.items():
        if digits in compact:
            scores[cls] = max(scores[cls], 0.62)
        if f"R{digits}" in compact or f"FCR{digits}" in compact or f"FC{digits}" in compact:
            scores[cls] = max(scores[cls], 0.88)

    fuzzy = [
        (r"7[1IL\|][0O]{2}", "FCR7100", 0.58),
        (r"8[1IL\|][0O]{2}", "FCR8100", 0.58),
        (r"9[2Z][0O]{2}", "FCR9200", 0.58),
    ]
    for pat, cls, w in fuzzy:
        if re.search(pat, compact):
            scores[cls] = max(scores[cls], w)

    return scores


def score_text_against_patterns(text: str) -> List[OCRClassScore]:
    norm = _normalize_ocr_text(text)
    if not norm:
        return []

    compact = _compact_alnum(norm)
    compact_fuzzy = _digit_fuzzy_compact(compact)
    digit_hints = _score_digit_hints(compact_fuzzy)

    results: List[OCRClassScore] = []
    for cls in CLASS_NAMES:
        matched: List[str] = []
        best = float(digit_hints.get(cls, 0.0))
        if best > 0:
            matched.append(f"digits:{cls}")

        for pattern, weight in OCR_PATTERNS.get(cls, []):
            m = re.search(pattern, norm, re.IGNORECASE)
            if m:
                matched.append(m.group(0))
                best = max(best, float(weight))

        if best > 0:
            results.append(OCRClassScore(class_name=cls, score=best, matched=matched))

    results.sort(key=lambda x: -x.score)
    return results


# -----------------------------
# Crops
# -----------------------------
def _spatial_crops_general(pil_rgb: Image.Image, mode: str) -> List[Tuple[str, Image.Image]]:
    w, h = pil_rgb.size
    crops: List[Tuple[str, Image.Image]] = [("full", pil_rgb)]
    center = pil_rgb.crop((int(w * 0.12), int(h * 0.12), int(w * 0.88), int(h * 0.88)))
    crops.append(("center", center))
    if mode == "thorough":
        lower = pil_rgb.crop((int(w * 0.10), int(h * 0.45), int(w * 0.90), int(h * 0.98)))
        crops.append(("lower", lower))
    return crops[:3]


def _spatial_crops_stamp(pil_rgb: Image.Image) -> List[Tuple[str, Image.Image]]:
    """
    刻印優先（STAMP）：
    背景文字を拾いにくい帯を複数試す（向き不明は回転スキャンで吸収）。
    """
    w, h = pil_rgb.size
    crops: List[Tuple[str, Image.Image]] = []

    crops.append(("no_top", pil_rgb.crop((0, int(h * 0.25), w, h))))

    crops.append((
        "center_low",
        pil_rgb.crop((
            int(w * 0.15), int(h * 0.35),
            int(w * 0.85), int(h * 0.95)
        ))
    ))

    crops.append(("bottom_band", pil_rgb.crop((int(w * 0.10), int(h * 0.55), int(w * 0.90), h))))

    crops.append((
        "mid_band",
        pil_rgb.crop((
            int(w * 0.10), int(h * 0.40),
            int(w * 0.90), int(h * 0.80)
        ))
    ))

    return crops[:4]


# -----------------------------
# EasyOCR runner
# -----------------------------
def _run_easyocr_on_variant(
    reader,
    rgb: np.ndarray,
    variant_name: str,
    *,
    allowlist: Optional[str] = None,
) -> List[OCRDetection]:
    out: List[OCRDetection] = []
    if reader is None:
        return out

    kwargs = {"detail": 1, "paragraph": False}
    if allowlist:
        kwargs["allowlist"] = allowlist

    try:
        raw = reader.readtext(rgb, **kwargs)
    except Exception:
        return out

    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        _bbox, text, conf = item[0], item[1], float(item[2])
        if not str(text).strip():
            continue
        out.append(OCRDetection(text=str(text).strip(), confidence=float(conf), variant=variant_name))
    return out


# -----------------------------
# Single-orientation OCR core
# -----------------------------
def _recognize_single_orientation(
    pil_rgb: Image.Image,
    reader,
    mode: str,
    target: str,
) -> OCRResult:
    max_side = OCR_MAX_SIDE_THOROUGH if mode == "thorough" else OCR_MAX_SIDE_FAST
    base = _resize_max_side(pil_rgb.convert("RGB"), max_side=max_side)

    if target == "stamp":
        allowlist = ALLOWLIST_STAMP
        crops = _spatial_crops_stamp(base)
    else:
        allowlist = ALLOWLIST_GENERAL
        crops = _spatial_crops_general(base, mode=mode)

    all_detections: List[OCRDetection] = []

    # raw救済（STAMPはallowlistで絞る）
    try:
        raw_rgb = _pil_to_rgb_uint8(base)
        all_detections.extend(_run_easyocr_on_variant(reader, raw_rgb, "raw/original", allowlist=allowlist))
    except Exception:
        pass

    for cname, crop in crops:
        variants = preprocess_variants(crop, mode=mode)
        for vname, arr in variants:
            tag = f"{cname}/{vname}"

            if target == "stamp" and vname == "otsu_inv":
                all_detections.extend(_run_easyocr_on_variant(reader, arr, tag + "+alnum", allowlist=ALLOWLIST_STAMP))
            else:
                all_detections.extend(_run_easyocr_on_variant(reader, arr, tag, allowlist=allowlist))

        if len(all_detections) > 90:
            break

    combined = " | ".join(d.text for d in all_detections)

    texts_to_score = [combined] + [d.text for d in all_detections]
    agg: Dict[str, float] = {c: 0.0 for c in CLASS_NAMES}

    for t in texts_to_score:
        if not t:
            continue
        for sc in score_text_against_patterns(t):
            det_boost = 0.0
            for d in all_detections:
                if d.text == t or t in d.text:
                    det_boost = max(det_boost, float(d.confidence) * 0.15)
            agg[sc.class_name] = max(agg[sc.class_name], min(1.0, float(sc.score) + det_boost))

    best_class: Optional[str] = None
    best_score: float = 0.0
    if agg:
        best_class = max(agg, key=agg.get)
        best_score = float(agg[best_class])
        if best_score <= 0:
            best_class = None

    return OCRResult(True, None, all_detections, agg, best_class, best_score, combined)


def _ocr_result_rank_key(res: OCRResult) -> Tuple[int, float, int]:
    has_best = 1 if (res.best_class is not None and res.best_score > 0) else 0
    return (has_best, float(res.best_score), len(res.detections))


# -----------------------------
# Public API: recognize_from_image
# -----------------------------
def recognize_from_image(
    pil_rgb: Image.Image,
    reader=None,
    mode: str = OCR_MODE,
    target: str = "general",   # "stamp" or "general"
) -> OCRResult:
    ok, msg = ocr_dependencies_ok()
    if not ok:
        return OCRResult(False, msg, [], {}, None, 0.0, "")
    if pil_rgb is None:
        return OCRResult(False, "入力画像がNoneです", [], {}, None, 0.0, "")

    if reader is None:
        try:
            reader = get_ocr_reader()
        except Exception as e:
            return OCRResult(False, f"OCR Reader 初期化失敗: {e}", [], {}, None, 0.0, "")

    candidates: List[OCRResult] = []
    best: Optional[OCRResult] = None
    best_key = (0, 0.0, 0)

    for deg in ROTATION_DEGREES:
        try:
            im_rot = pil_rgb if deg == 0 else pil_rgb.rotate(deg, expand=True)
            res = _recognize_single_orientation(im_rot, reader=reader, mode=mode, target=target)
            if res.detections:
                for d in res.detections:
                    d.variant = f"rot{deg}/" + d.variant

            candidates.append(res)
            key = _ocr_result_rank_key(res)
            if key > best_key:
                best, best_key = res, key

            # 早期終了（負荷軽減）
            if res.best_class is not None and float(res.best_score) >= EARLY_STOP_SCORE:
                break
        except Exception:
            continue

    if best is None:
        best = OCRResult(True, None, [], {}, None, 0.0, "")

    # 90°刻みで検出ゼロの時だけ微調整（fastのみ）
    if len(best.detections) == 0 and mode == "fast":
        tilt_candidates: List[OCRResult] = []
        for deg in ROTATION_DEGREES:
            base_rot = pil_rgb if deg == 0 else pil_rgb.rotate(deg, expand=True)
            for tilt in FALLBACK_SMALL_TILTS:
                try:
                    im_tilt = base_rot.rotate(tilt, expand=True)
                    res = _recognize_single_orientation(im_tilt, reader=reader, mode=mode, target=target)
                    if res.detections:
                        for d in res.detections:
                            d.variant = f"rot{deg}_tilt{tilt}/" + d.variant
                    tilt_candidates.append(res)
                except Exception:
                    continue
        if tilt_candidates:
            best2 = max(tilt_candidates, key=_ocr_result_rank_key)
            if _ocr_result_rank_key(best2) > _ocr_result_rank_key(best):
                best = best2

    return best


# -----------------------------
# Public API: fuse_cnn_and_ocr
# -----------------------------
def fuse_cnn_and_ocr(
    cnn_probs: np.ndarray,
    class_names: List[str],
    ocr: OCRResult,
) -> FusedPrediction:
    cnn_idx = int(np.argmax(cnn_probs))
    cnn_label = class_names[cnn_idx]
    cnn_conf = float(cnn_probs[cnn_idx])

    ocr_label = ocr.best_class
    ocr_conf = float(ocr.best_score) if ocr.best_class else 0.0
    ocr_text = ocr.combined_text or ""

    if (not ocr.available) or (ocr_label is None):
        return FusedPrediction(cnn_label, cnn_conf, "cnn", cnn_label, cnn_conf, None, 0.0, ocr_text)

    if ocr_conf >= OCR_OVERRIDE_THRESHOLD:
        if ocr_label == cnn_label:
            conf = min(1.0, max(cnn_conf, ocr_conf) + FUSION_AGREE_BOOST)
            method = "fusion_agree"
        else:
            conf = ocr_conf
            method = "fusion_ocr"
        return FusedPrediction(ocr_label, conf, method, cnn_label, cnn_conf, ocr_label, ocr_conf, ocr_text)

    if ocr_label == cnn_label and ocr_conf >= 0.4:
        conf = min(1.0, cnn_conf + ocr_conf * 0.25)
        return FusedPrediction(cnn_label, conf, "fusion_agree", cnn_label, cnn_conf, ocr_label, ocr_conf, ocr_text)

    if cnn_conf >= 0.55 and ocr_label != cnn_label and ocr_conf < 0.55:
        return FusedPrediction(cnn_label, cnn_conf, "fusion_cnn", cnn_label, cnn_conf, ocr_label, ocr_conf, ocr_text)

    if ocr_conf > cnn_conf and ocr_conf >= 0.5:
        return FusedPrediction(ocr_label, ocr_conf, "fusion_ocr", cnn_label, cnn_conf, ocr_label, ocr_conf, ocr_text)

    return FusedPrediction(cnn_label, cnn_conf, "cnn", cnn_label, cnn_conf, ocr_label, ocr_conf, ocr_text)
