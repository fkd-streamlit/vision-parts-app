# -*- coding: utf-8 -*-
"""
刻印・ロゴの OCR と型番パターンマッチ（軽量・安定版）
黒ベース製品向けに前処理バリアントを試し、CNN結果と融合する。

【安定化パッチ（2026-05）】
- 入力画像は必ず縮小（max_side）してメモリピークを抑制
- 前処理バリアント数を削減（thoroughでも暴れない）
- crop数も制限（fast=中心のみ / thoroughでも最大4）
- EasyOCR Reader は gpu=False 固定（Cloud安定化）
- OCR失敗時は例外を抑え、呼び出し側でCNN継続できる設計
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

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

# 安定化：OCRに渡す最大辺（Cloud向け）
OCR_MAX_SIDE_FAST = 896
OCR_MAX_SIDE_THOROUGH = 1024


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
        # verbose=False でログを抑制、gpu=False で安定化
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


# -----------------------------
# Image helpers
# -----------------------------
def _resize_max_side(im: Image.Image, max_side: int) -> Image.Image:
    """最大辺が max_side を超える場合は縮小（必ず縮小方向）。"""
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    scale = max_side / m
    return im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def _pil_to_rgb_uint8(im: Image.Image) -> np.ndarray:
    """PIL RGB -> numpy RGB uint8"""
    if im.mode != "RGB":
        im = im.convert("RGB")
    arr = np.array(im)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _rgb_to_gray(arr_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> Gray uint8"""
    if not HAS_CV2:
        # fallback: simple luminance
        r = arr_rgb[..., 0].astype(np.float32)
        g = arr_rgb[..., 1].astype(np.float32)
        b = arr_rgb[..., 2].astype(np.float32)
        y = (0.299 * r + 0.587 * g + 0.114 * b).clip(0, 255).astype(np.uint8)
        return y
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _clahe_gray(gray: np.ndarray) -> np.ndarray:
    """Gray -> CLAHE Gray"""
    if not HAS_CV2:
        return gray
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _otsu_inv(gray: np.ndarray) -> np.ndarray:
    """Gray -> Otsu -> invert if needed (刻印が白っぽく出る想定)"""
    if not HAS_CV2:
        # 代替：単純二値化
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
# Preprocess variants (軽量版)
# -----------------------------
def preprocess_variants(pil_rgb: Image.Image, mode: str = OCR_MODE) -> List[Tuple[str, np.ndarray]]:
    """
    黒製品の刻印向けに少数の前処理画像を生成（RGB uint8）。
    - fast: original / clahe / otsu_inv の3種程度
    - thorough: 上記 + pil_hi_contrast を追加（最大4種）
    """
    max_side = OCR_MAX_SIDE_THOROUGH if mode == "thorough" else OCR_MAX_SIDE_FAST
    base = _resize_max_side(pil_rgb.convert("RGB"), max_side=max_side)

    rgb = _pil_to_rgb_uint8(base)
    variants: List[Tuple[str, np.ndarray]] = [("original", rgb)]

    # gray->clahe
    gray = _rgb_to_gray(rgb)
    g_clahe = _clahe_gray(gray)
    clahe_rgb = np.stack([g_clahe, g_clahe, g_clahe], axis=-1)
    variants.append(("clahe", clahe_rgb))

    # otsu_inv
    bw = _otsu_inv(g_clahe)
    bw_rgb = np.stack([bw, bw, bw], axis=-1)
    variants.append(("otsu_inv", bw_rgb))

    if mode == "thorough":
        # PILでコントラスト&シャープ（重すぎない範囲）
        hi = ImageEnhance.Contrast(base).enhance(2.0)
        hi = ImageEnhance.Sharpness(hi).enhance(1.6)
        variants.append(("pil_hi_contrast", _pil_to_rgb_uint8(hi)))

    # 安全：最大4種類まで
    return variants[:4]


# -----------------------------
# Text scoring
# -----------------------------
def _normalize_ocr_text(text: str) -> str:
    t = (text or "").upper()
    t = re.sub(r"[^A-Z0-9\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_DIGIT_CLASS = {
    "7100": "FCR7100",
    "8100": "FCR8100",
    "9200": "FCR9200",
}


def _compact_alnum(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


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

    # 7I00 / 7l00 / 9Z00 等の誤認識を軽く救う
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
    """読み取り文字列から各クラスのスコアを算出。"""
    norm = _normalize_ocr_text(text)
    if not norm:
        return []

    compact = _compact_alnum(norm)
    digit_hints = _score_digit_hints(compact)

    results: List[OCRClassScore] = []
    for cls in CLASS_NAMES:
        matched: List[str] = []
        best = float(digit_hints.get(cls, 0.0))
        if best > 0:
            matched.append(f"digits:{cls}")

        # config.OCR_PATTERNS を利用（既存互換）
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
# Crops (軽量版)
# -----------------------------
def _spatial_crops(pil_rgb: Image.Image, mode: str = OCR_MODE) -> List[Tuple[str, Image.Image]]:
    """
    刻印が写りやすい領域を切り出す（軽量版）。
    - fast: centerのみ
    - thorough: full + center + lower（最大3）
    """
    w, h = pil_rgb.size
    crops: List[Tuple[str, Image.Image]] = []

    # fullは必ず入れる（ただし縮小済み前提）
    crops.append(("full", pil_rgb))

    # center
    center = pil_rgb.crop((int(w * 0.12), int(h * 0.12), int(w * 0.88), int(h * 0.88)))
    crops.append(("center", center))

    if mode == "thorough":
        lower = pil_rgb.crop((int(w * 0.10), int(h * 0.45), int(w * 0.90), int(h * 0.98)))
        crops.append(("lower", lower))

    # 安全：最大3
    return crops[:3]


# -----------------------------
# EasyOCR runner (ガード強化)
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
# Public API: recognize_from_image
# -----------------------------
def recognize_from_image(
    pil_rgb: Image.Image,
    reader=None,
    mode: str = OCR_MODE,
) -> OCRResult:
    """1枚の画像から OCR + パターンマッチ（軽量・安定版）。"""
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

    if pil_rgb is None:
        return OCRResult(
            available=False,
            error="入力画像がNoneです",
            detections=[],
            class_scores={},
            best_class=None,
            best_score=0.0,
            combined_text="",
        )

    if reader is None:
        try:
            reader = get_ocr_reader()
        except Exception as e:
            return OCRResult(
                available=False,
                error=f"OCR Reader 初期化失敗: {e}",
                detections=[],
                class_scores={},
                best_class=None,
                best_score=0.0,
                combined_text="",
            )

    # 入力全体を縮小（最重要：必ず縮小する）
    max_side = OCR_MAX_SIDE_THOROUGH if mode == "thorough" else OCR_MAX_SIDE_FAST
    base = _resize_max_side(pil_rgb.convert("RGB"), max_side=max_side)

    # allowlist は英数記号に制限して誤検出を減らす
    allowlist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- "

    all_detections: List[OCRDetection] = []

    # crop数を制限（fast: full+center だけ、thorough: +lower）
    crops = _spatial_crops(base, mode=mode)
# ★追加：raw（加工前）を allowlist なしで一度OCRに通す
    # カタログ画像や太字文字、反射で二値化が壊れるケースの救済
    try:
        raw_rgb = np.array(base)  # base は既にRGB & 縮小済み
        all_detections.extend(
            _run_easyocr_on_variant(reader, raw_rgb, "raw/original", allowlist=None)
        )
    except Exception:
        pass

    # variant数も制限（max 4）
    for cname, crop in crops:
        variants = preprocess_variants(crop, mode=mode)
        for vname, arr in variants:
            tag = f"{cname}/{vname}"
            # otsu_invのみ allowlist を付けて精度を稼ぐ（負荷も増えない）
        if vname == "otsu_inv":
                all_detections.extend(
                    _run_easyocr_on_variant(reader, arr, tag + "+alnum", allowlist=allowlist)
                )
            elif cname == "full" and vname == "original":
                # ★追加：full/original は allowlist なし（太字や記号混じり救済）
                all_detections.extend(
                    _run_easyocr_on_variant(reader, arr, tag + "+raw", allowlist=None)
                )
            else:
                all_detections.extend(_run_easyocr_on_variant(reader, arr, tag))
``

        # 安全：検出が大量なら打ち切り（メモリ/時間対策）
        if len(all_detections) > 80:
            break

    combined = " | ".join(d.text for d in all_detections)

    # 全文 + 各検出行を個別にスコア化し最大を採用
    texts_to_score = [combined] + [d.text for d in all_detections]

    agg: Dict[str, float] = {c: 0.0 for c in CLASS_NAMES}
    for t in texts_to_score:
        if not t:
            continue
        scored = score_text_against_patterns(t)
        for sc in scored:
            # 検出confを微小にブースト（上げすぎない）
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

    return OCRResult(
        available=True,
        error=None,
        detections=all_detections,
        class_scores=agg,
        best_class=best_class,
        best_score=best_score,
        combined_text=combined,
    )


# -----------------------------
# Public API: fuse_cnn_and_ocr
# -----------------------------
def fuse_cnn_and_ocr(
    cnn_probs: np.ndarray,
    class_names: List[str],
    ocr: OCRResult,
) -> FusedPrediction:
    """CNN softmax と OCR スコアを融合（既存互換）。"""
    cnn_idx = int(np.argmax(cnn_probs))
    cnn_label = class_names[cnn_idx]
    cnn_conf = float(cnn_probs[cnn_idx])

    ocr_label = ocr.best_class
    ocr_conf = float(ocr.best_score) if ocr.best_class else 0.0
    ocr_text = ocr.combined_text or ""

    if (not ocr.available) or (ocr_label is None):
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

    # OCRが十分強い場合：OCR優先（一致ならブースト）
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

    # OCRとCNNが一致＆OCRがそこそこ：弱融合
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

    # CNNが強くOCRが弱い：CNN優先
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

    # OCRがCNNより強い：OCR優先
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
