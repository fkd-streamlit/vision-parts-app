# -*- coding: utf-8 -*-
"""
自転車部品 自動判定システム（app_poc.py）
画像認識（ONNX）+ 刻印OCR（EasyOCR）の融合判定
torch 不要・Streamlit Cloud 対応
"""

from __future__ import annotations

import io
import zipfile
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import streamlit as st
import onnxruntime as ort

from config import CLASS_NAMES, NUM_CLASSES, get_display_name, is_reject
from product_catalog import render_manual_section
from ocr_engine import (
    FusedResult,
    OCRResult,
    draw_ocr_bboxes,
    fuse_cnn_and_ocr,
    get_ocr_reader,
    ocr_dependencies_ok,
    recognize_from_image,
)
from camera_guide import (
    BAD_EXAMPLES,
    SHOOTING_TIPS,
    draw_guide_overlay,
    get_shooting_examples,
    guide_overlay_html,
)

# =========================================================
# 前処理
# =========================================================
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_SIDE = 1024  # メモリ節約のため画像を縮小


def shrink(im: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    s = max_side / m
    return im.resize((int(w * s), int(h * s)), Image.LANCZOS)


def preprocess(img: Image.Image) -> np.ndarray:
    img  = img.resize((256, 256), Image.LANCZOS)
    left = (256 - 224) // 2
    img  = img.crop((left, left, left + 224, left + 224))
    arr  = np.array(img, dtype=np.float32) / 255.0
    arr  = (arr - MEAN) / STD
    arr  = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# =========================================================
# ユーティリティ
# =========================================================
def pil_open_rgb(file_bytes: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(file_bytes))
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    im = im.convert("RGB") if im.mode != "RGB" else im
    return shrink(im)


def list_images_in_zip(zf: zipfile.ZipFile) -> List[zipfile.ZipInfo]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [i for i in zf.infolist()
            if not i.is_dir() and Path(i.filename).suffix.lower() in exts]


# =========================================================
# モデル読込（ONNX）
# =========================================================
@st.cache_resource(show_spinner="ONNXモデルを読み込んでいます...")
def load_onnx_model(model_path: str) -> ort.InferenceSession:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def predict_images(session: ort.InferenceSession,
                   images: List[Image.Image]) -> np.ndarray:
    """1枚ずつ推論してスタック（バッチサイズ問題を回避）"""
    results = []
    for img in images:
        batch  = preprocess(img)
        inp    = session.get_inputs()[0].name
        out    = session.get_outputs()[0].name
        logits = session.run([out], {inp: batch})[0]
        results.append(softmax(logits)[0])
    return np.stack(results, axis=0)


# =========================================================
# OCR キャッシュ
# =========================================================
@st.cache_resource(show_spinner="OCRエンジンを初期化中（初回のみ数分かかります）...")
def load_ocr_reader_cached():
    return get_ocr_reader()


# =========================================================
# 判定方法ラベル
# =========================================================
METHOD_LABELS = {
    "cnn":          "画像認識のみ",
    "fusion_agree": "画像認識＋OCR（一致）",
    "fusion_ocr":   "刻印OCRを優先",
    "fusion_cnn":   "画像認識を優先",
    "ocr_direct":   "OCR直接検出（学習外モデル）",
}

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="部品判定システム", page_icon="🔧", layout="wide")
st.title("🔧 自転車部品 自動判定システム")

target_classes = [get_display_name(c) for c in CLASS_NAMES if not is_reject(c)]
st.caption("対応部品：" + " / ".join(target_classes))

# =========================================================
# サイドバー
# =========================================================
with st.sidebar:
    st.header("⚙️ モデル設定")
    model_path_input = st.text_input("ONNXモデルのパス (.onnx)",
                                     value="shimano_model_poc.onnx")
    st.caption("または直接アップロード：")
    up_model = st.file_uploader("ONNXモデル (.onnx)", type=["onnx"])

    st.markdown("---")
    st.header("🔍 OCR設定")
    ocr_ok, ocr_msg = ocr_dependencies_ok()
    use_ocr = st.toggle("刻印OCRを使用（推奨）", value=True)
    if use_ocr and not ocr_ok:
        st.warning(ocr_msg)
    ocr_mode = st.selectbox(
        "OCRモード",
        ["fast", "thorough"],
        format_func=lambda x: "高速（カメラ向け）" if x == "fast" else "高精度（時間がかかります）",
    )

    st.markdown("---")
    st.header("🖼️ 画像処理（安定化）")
    max_side = st.slider("入力画像の最大辺（縮小して軽量化）", 512, 2048, MAX_SIDE, 128)

    st.markdown("---")
    st.header("🎚️ 推論設定")
    topk    = st.slider("Top-k 表示数", 1, max(1, NUM_CLASSES - 1), min(2, NUM_CLASSES - 1))
    conf_th = st.slider("確信度しきい値", 0.0, 1.0, 0.5, 0.05)
    BATCH   = st.number_input("バッチサイズ", min_value=1, max_value=16, value=4)

    ocr_debug = st.checkbox("OCRデバッグ表示（実行状況と検出結果）", value=False)

# =========================================================
# モデル読込
# =========================================================
tmp_model_path: Optional[str] = None
if up_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp:
        tmp.write(up_model.read())
        tmp_model_path = tmp.name
    use_model_path = tmp_model_path
    st.sidebar.success(f"アップロードモデルを使用: {up_model.name}")
else:
    use_model_path = model_path_input.strip()

if not use_model_path:
    st.error("モデルパスを指定してください。")
    st.stop()

try:
    onnx_session = load_onnx_model(use_model_path)
except FileNotFoundError as e:
    st.warning(str(e))
    st.info("サイドバーからモデルをアップロードしてください。")
    st.stop()
except Exception as e:
    st.error(f"ONNXモデル読込失敗: {e}")
    st.stop()

st.sidebar.success("✅ モデル読込 OK")

ocr_reader = None
if use_ocr and ocr_ok:
    try:
        ocr_reader = load_ocr_reader_cached()
        st.sidebar.success("✅ OCR読込 OK")
    except Exception as e:
        st.sidebar.warning(f"OCR読込失敗（画像認識のみ）: {e}")

# =========================================================
# 画像入力
# =========================================================
st.subheader("📷 画像入力")
input_mode = st.radio(
    "入力方法",
    ["カメラ撮影（スマホ推奨）", "ファイルアップロード（ZIP / 画像）"],
    horizontal=True,
)

names: List[str]              = []
pil_images: List[Image.Image] = []

if input_mode == "カメラ撮影（スマホ推奨）":
    # 撮影ガイド（折りたたみ）
    with st.expander("📋 撮影ガイド", expanded=False):
        for tip in SHOOTING_TIPS:
            st.markdown(f"- {tip}")

        ex_cols = st.columns(3)
        for col, ex in zip(ex_cols, get_shooting_examples()):
            with col:
                p = ex.get("path")
                if p and Path(str(p)).is_file():
                    try:
                        ex_im = Image.open(p).convert("RGB")
                        st.image(draw_guide_overlay(ex_im, show_label=False),
                                 use_container_width=True)
                    except Exception:
                        st.caption("（画像なし）")
                else:
                    st.caption("（撮影例画像なし）")
                st.markdown(f"**{ex['product']}**")
                st.caption(ex["tip"])

        st.markdown("##### ❌ 避けたい撮影")
        for title, desc in BAD_EXAMPLES:
            st.markdown(f"- {title} — {desc}")

    # 撮影のコツ
    st.markdown(guide_overlay_html(), unsafe_allow_html=True)

    st.info("💡 刻印（DURA-ACE / ULTEGRA / 105 等）が写るよう近づいて撮影してください。")
    img_file = st.camera_input("部品の刻印を撮影")

    if img_file:
        try:
            pil_images = [pil_open_rgb(img_file.read())]
            names      = ["camera_shot.jpg"]
        except Exception as e:
            st.error(f"画像の読み込みに失敗: {e}")

else:
    up_files = st.file_uploader(
        "ZIP または画像ファイル（複数可）",
        type=["zip", "jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True,
    )
    if up_files:
        for f in up_files:
            try:
                if f.name.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
                        for info in list_images_in_zip(zf):
                            with zf.open(info) as zfi:
                                pil_images.append(pil_open_rgb(zfi.read()))
                            names.append(info.filename)
                else:
                    pil_images.append(pil_open_rgb(f.read()))
                    names.append(f.name)
            except Exception as e:
                st.warning(f"{f.name}: 読込失敗（{e}）")
        st.success(f"{len(pil_images)} 枚を読み込みました。")
    else:
        st.info("ファイルをアップロードしてください。")

if not pil_images:
    st.stop()

# =========================================================
# 推論実行
# =========================================================
progress = st.progress(0, text="画像認識中...")
total    = len(pil_images)

try:
    image_probs_all = predict_images(onnx_session, pil_images)
except Exception as ex:
    st.error(f"画像認識エラー: {ex}")
    st.stop()

progress.progress(50, text="OCR処理中...")

fused_results: List[FusedResult] = []
for i, img in enumerate(pil_images):
    if ocr_reader is not None:
        try:
            ocr_res = recognize_from_image(img, reader=ocr_reader, mode=ocr_mode)
            if ocr_debug:
                st.caption(f"OCR[{i}]: {ocr_res.method} / テキスト: {ocr_res.ocr_text[:80]}")
        except Exception as ex:
            if ocr_debug:
                st.caption(f"OCR[{i}] エラー: {ex}")
            ocr_res = OCRResult(False, None, None, None, [], {}, None, 0.0, "OCRエラー", False)
    else:
        ocr_res = OCRResult(False, None, None, None, [], {}, None, 0.0, "OCR無効", False)

    fused_results.append(fuse_cnn_and_ocr(image_probs_all[i], CLASS_NAMES, ocr_res))
    progress.progress(50 + int(50 * (i + 1) / total), text="OCR処理中...")

progress.empty()

pred_label = [f.label       for f in fused_results]
pred_conf  = np.array([f.confidence for f in fused_results])

# =========================================================
# 結果表示
# =========================================================

# --- カメラモード：1枚大きく ---
if input_mode == "カメラ撮影（スマホ推奨）" and len(pil_images) == 1:
    fr     = fused_results[0]
    topk_i = [i for i in np.argsort(-image_probs_all[0])
               if not is_reject(CLASS_NAMES[i])][:topk]

    st.markdown("---")
    col_img, col_result = st.columns([1, 1])

    with col_img:
        # ガイド枠 + OCRバウンディングボックス
        display_im = draw_guide_overlay(pil_images[0])
        if fr.bboxes:
            display_im = draw_ocr_bboxes(display_im, fr.bboxes)
        st.image(display_im, caption="撮影画像（緑枠＝OCR解析範囲）",
                 use_container_width=True)

        if fr.bboxes:
            matched = sum(1 for b in fr.bboxes if b.matched)
            st.caption(
                f"OCR検出: {len(fr.bboxes)}件 "
                f"（🟩 マッチ: {matched}件 / 🟧 その他: {len(fr.bboxes)-matched}件）"
            )

    with col_result:
        st.subheader("判定結果")
        conf  = fr.confidence
        label = fr.label

        if fr.method == "ocr_direct":
            st.warning(f"🔍 **{fr.display_name}**")
            st.metric("OCR確信度", f"{conf*100:.1f}%")
            st.info("📌 学習データにないモデルです。OCRによる型番直接検出です。")
            if fr.category:
                st.caption(f"部品カテゴリ: {fr.category}")
        elif is_reject(label):
            st.error("⚠️ 対象部品が検出されませんでした")
            st.caption("対応部品をカメラに向けて再撮影してください。")
        elif conf < conf_th:
            st.warning(f"⚠️ 確信度が低いため判定不可（{conf*100:.1f}%）")
            st.caption("刻印が写るよう近づいて再撮影してください。")
        else:
            st.success(f"✅ **{fr.display_name}**")
            st.metric("確信度（融合後）", f"{conf*100:.1f}%")

        st.info(f"判定根拠: {METHOD_LABELS.get(fr.method, fr.method)}")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("画像認識",
                      f"{get_display_name(fr.cnn_label)} ({fr.cnn_conf*100:.0f}%)")
        with c2:
            if fr.ocr_display:
                st.metric("OCR",
                          f"{fr.ocr_display} ({fr.ocr_conf*100:.0f}%)")
            else:
                st.metric("OCR", "型番未検出")

        if fr.ocr_text:
            with st.expander("OCR 読み取りテキスト"):
                st.code(fr.ocr_text[:500] or "（なし）")

        st.markdown("**画像認識 Top-k スコア**")
        for i in topk_i:
            val = float(image_probs_all[0, i])
            st.write(f"{get_display_name(CLASS_NAMES[i])} : {val*100:.1f}%")
            st.progress(val)

    # マニュアルセクション
    render_manual_section(
        class_name     = fr.label,
        confidence     = fr.confidence,
        conf_threshold = conf_th,
        expanded       = fr.confidence >= conf_th,
    )

# --- ファイルモード：一覧テーブル ---
else:
    rows = []
    for i, nm in enumerate(names):
        fr   = fused_results[i]
        conf = fr.confidence
        if fr.method == "ocr_direct":
            judge = "OCR検出"
        elif is_reject(fr.label):
            judge = "対象外"
        elif conf >= conf_th:
            judge = "OK"
        else:
            judge = "要確認"

        rows.append({
            "ファイル名": nm,
            "最終判定":  fr.display_name,
            "確信度":    f"{conf*100:.1f}%",
            "画像認識":  f"{get_display_name(fr.cnn_label)} ({fr.cnn_conf*100:.0f}%)",
            "OCR":       fr.ocr_display or "未検出",
            "判定根拠":  METHOD_LABELS.get(fr.method, fr.method),
            "判定":      judge,
        })

    df = pd.DataFrame(rows)

    st.subheader("📋 判定結果一覧")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 結果をCSVでダウンロード", csv_bytes,
                       "parts_result.csv", mime="text/csv")

    st.subheader("📊 部品別判定数")
    df_ok = df[df["判定"].isin(["OK", "OCR検出"])]
    if not df_ok.empty:
        counts = df_ok["最終判定"].value_counts()
        st.bar_chart(counts)

    # マニュアルセクション（OKの製品のみ）
    ok_rows = [(i, nm) for i, nm in enumerate(names)
               if fused_results[i].confidence >= conf_th
               and not is_reject(fused_results[i].label)
               and fused_results[i].label != "OCR_DIRECT"]
    if ok_rows:
        st.subheader("📖 マニュアル・技術資料")
        for i, nm in ok_rows:
            fr = fused_results[i]
            st.markdown(f"**{Path(nm).name}** → {fr.display_name}")
            render_manual_section(
                class_name     = fr.label,
                confidence     = fr.confidence,
                conf_threshold = conf_th,
                expanded       = len(ok_rows) == 1,
            )

    st.subheader("🖼️ サンプル画像（最大12枚）")
    cols = st.columns(4)
    for i in range(min(len(pil_images), 12)):
        fr      = fused_results[i]
        topk_i  = [j for j in np.argsort(-image_probs_all[i])
                   if not is_reject(CLASS_NAMES[j])][:topk]
        thumb   = draw_ocr_bboxes(pil_images[i], fr.bboxes) \
                  if fr.bboxes else pil_images[i]
        caption = (
            f"最終: {fr.display_name} ({fr.confidence*100:.0f}%)\n"
            + " / ".join(
                [f"{get_display_name(CLASS_NAMES[j])}: {image_probs_all[i,j]*100:.1f}%"
                 for j in topk_i]
            )
        )
        with cols[i % 4]:
            st.image(thumb,
                     caption=f"{Path(names[i]).name}\n{caption}",
                     use_container_width=True)
