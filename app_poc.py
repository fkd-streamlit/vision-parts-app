# -*- coding: utf-8 -*-
"""
シマノ製品モデル 自動特定システム (PoC版)
- ResNet-50 転移学習 + 刻印OCR（EasyOCR）の融合判定
- Streamlit Cloud 対応：st.camera_input でスマホカメラ撮影→即時推論
- ローカル実行時：ZIPまたは画像ファイルのアップロードにも対応
"""

import io
import os
import zipfile
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models

from config import CLASS_NAMES, get_display_name
from ocr_engine import (
    OCRResult,
    fuse_cnn_and_ocr,
    ocr_dependencies_ok,
    recognize_from_image,
)
from camera_guide import (
    BAD_EXAMPLES,
    SHOOTING_TIPS,
    draw_guide_overlay,
    get_shooting_examples,
    guide_overlay_html,
    make_viewfinder_template,
)
from product_catalog import render_manual_section

NUM_CLASSES = len(CLASS_NAMES)

# =========================================================
# 2) 前処理（学習時の val_tf と同じ設定）
# =========================================================
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])

# =========================================================
# 3) ユーティリティ関数
# =========================================================
def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pil_open_rgb(file_bytes: bytes) -> Image.Image:
    """バイト列から PIL Image (RGB) を生成。EXIF回転も自動補正。"""
    im = Image.open(io.BytesIO(file_bytes))
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def list_images_in_zip(zf: zipfile.ZipFile) -> List[zipfile.ZipInfo]:
    """ZIP内の画像ファイル一覧を返す。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [
        info for info in zf.infolist()
        if not info.is_dir()
        and Path(info.filename).suffix.lower() in exts
    ]


# =========================================================
# 4) 推論関数
# =========================================================
@torch.inference_mode()
def predict_batch(
    model: torch.nn.Module,
    images: List[Image.Image],
    tfm,
    device: str,
) -> torch.Tensor:
    """PIL Image のリストに対してバッチ推論し、logits を返す。"""
    xs = [tfm(im) for im in images]
    batch = torch.stack(xs, dim=0).to(device)
    out = model(batch)
    if not torch.is_tensor(out):
        raise RuntimeError(f"モデル出力が Tensor ではありません: {type(out)}")
    if out.ndim != 2:
        raise RuntimeError(f"logits.shape が不正: {tuple(out.shape)}（期待 [N, C]）")
    return out.detach().cpu()


# =========================================================
# 5) モデル読込（キャッシュ付き）
# =========================================================
@st.cache_resource(show_spinner="モデルを読み込んでいます...")
def load_model(model_path: str, num_classes: int) -> torch.nn.Module:
    """
    .ckpt ファイルから ResNet-50 を復元する。
    ckpt は {"state_dict": ..., "class_names": ...} 形式を想定。
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")

    # class_names の整合チェック
    saved_classes = ckpt.get("class_names")
    if saved_classes and saved_classes != CLASS_NAMES:
        st.warning(
            f"モデルのクラス順 {saved_classes} と "
            f"アプリ定義 {CLASS_NAMES} が異なります。結果が不正確になる可能性があります。"
        )

    state_dict = ckpt.get("state_dict") or ckpt  # state_dict キーがない場合も考慮

    # ResNet-50 ベースモデルを構築
    backbone = models.resnet50(weights=None)
    backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)

    # "model." / "module." プレフィックスを除去して読み込む
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("model.", "module."):
            if k.startswith(prefix):
                nk = k[len(prefix):]
                break
        new_sd[nk] = v

    backbone.load_state_dict(new_sd, strict=False)
    backbone.eval()
    return backbone


@st.cache_resource(show_spinner="OCRエンジンを初期化しています（初回はモデルDL）...")
def load_ocr_reader():
    from ocr_engine import get_ocr_reader
    return get_ocr_reader()


@st.cache_data
def cached_viewfinder_template():
    return make_viewfinder_template()


# =========================================================
# 6) Streamlit UI
# =========================================================
st.set_page_config(page_title="シマノ クランク判定", layout="wide")
st.title("特定製品 自動特定システム")
st.caption("対応モデル：FCR7100 / FCR8100 / FCR9200 ｜ CNN + 刻印OCR 融合判定")

# --- サイドバー：モデル設定 ---
with st.sidebar:
    st.header("モデル設定")

    # モデルパスの入力
    model_path_input = st.text_input(
        "学習済みモデルのパス (.ckpt)",
        value="shimano_model_poc.ckpt",
        help="Streamlit Cloud の場合はリポジトリに含めてください。"
    )

    # または直接アップロード
    st.caption("または、モデルファイルをここにアップロード：")
    up_model = st.file_uploader(
        "モデルファイル (.ckpt)",
        type=["ckpt"],
        accept_multiple_files=False,
    )

    st.markdown("---")

    # 推論設定
    st.header("推論設定")
    topk    = st.slider("Top-k 表示数", 1, NUM_CLASSES, min(2, NUM_CLASSES))
    conf_th = st.slider("確信度しきい値", 0.0, 1.0, 0.5, 0.05,
                        help="この値未満の結果は「判定不可」として扱います")
    BATCH   = st.number_input("バッチサイズ", min_value=1, max_value=64, value=8)

    st.markdown("---")
    st.header("OCR（刻印読取）")
    use_ocr = st.checkbox(
        "刻印・型番の OCR を使う（推奨）",
        value=True,
        help="黒いクランクでも前処理＋文字読取で精度向上を狙います。初回はモデルDLで数十秒かかります。",
    )
    ocr_mode = st.selectbox(
        "OCRモード",
        ["fast", "thorough"],
        index=0,
        format_func=lambda x: "高速（撮影向け）" if x == "fast" else "高精度（時間がかかります）",
    )
    ocr_ok, ocr_msg = ocr_dependencies_ok()
    if use_ocr and not ocr_ok:
        st.warning(ocr_msg)

    dev = device_str()
    st.caption(f"推論デバイス: {dev.upper()}")

# --- モデルパス決定 ---
tmp_model_path: Optional[str] = None

if up_model is not None:
    suffix = Path(up_model.name).suffix or ".ckpt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(up_model.read())
        tmp_model_path = tmp.name
    use_model_path = tmp_model_path
    st.sidebar.success(f"アップロードモデルを使用: {up_model.name}")
else:
    use_model_path = model_path_input.strip()

if not use_model_path:
    st.error("サイドバーでモデルパスを指定するか、モデルファイルをアップロードしてください。")
    st.stop()

try:
    model = load_model(use_model_path, NUM_CLASSES)
    model = model.to(dev)
except FileNotFoundError as e:
    st.warning(str(e))
    st.info(
        "モデルファイルがまだない場合は、サイドバーからアップロードするか、"
        "train_poc.py で学習を実行してください。"
    )
    st.stop()
except Exception as e:
    st.error(f"モデル読込失敗: {e}")
    st.stop()

st.sidebar.success("モデル読込 OK")

# =========================================================
# 7) 画像入力（3モード）
# =========================================================
st.subheader("画像入力")

input_mode = st.radio(
    "入力方法を選んでください",
    ["カメラ撮影（スマホ推奨）", "ファイルアップロード（ZIP / 画像）"],
    horizontal=True,
)

names: List[str]        = []
pil_images: List[Image.Image] = []

# ---- モード A：カメラ撮影 ----
if input_mode == "カメラ撮影（スマホ推奨）":
    with st.expander("📷 撮影ガイド・撮影例", expanded=True):
        st.markdown("**刻印（R7100 等）を緑の枠内に収めて撮影**すると、OCRの精度が上がります。")
        for tip in SHOOTING_TIPS:
            st.markdown(f"- {tip}")

        st.markdown("##### ✅ 良い撮影例（参考）")
        ex_cols = st.columns(3)
        for col, ex in zip(ex_cols, get_shooting_examples()):
            with col:
                if ex["path"] and ex["path"].is_file():
                    try:
                        ex_im = Image.open(ex["path"]).convert("RGB")
                        thumb = draw_guide_overlay(ex_im, show_label=False)
                        st.image(thumb, use_container_width=True)
                    except Exception:
                        st.caption("（画像を表示できません）")
                else:
                    st.caption("（撮影例画像なし）")
                st.markdown(f"**{ex['product']}**")
                st.caption(ex["tip"])

        st.markdown("##### ❌ 避けたい撮影")
        for title, desc in BAD_EXAMPLES:
            st.markdown(f"- **{title}** — {desc}")

    guide_col1, guide_col2 = st.columns([1, 1])
    with guide_col1:
        st.image(
            cached_viewfinder_template(),
            caption="ビューファインダー（撮影イメージ）",
            use_container_width=True,
        )
    with guide_col2:
        st.markdown(guide_overlay_html(), unsafe_allow_html=True)

    st.info("スマホでこのページを開き、枠を参考に刻印を合わせてから撮影してください。")
    img_file = st.camera_input("クランクの刻印を枠内に合わせて撮影")

    if img_file is not None:
        try:
            im = pil_open_rgb(img_file.read())
            pil_images = [im]
            names      = ["camera_shot.jpg"]
        except Exception as e:
            st.error(f"画像の読み込みに失敗しました: {e}")

# ---- モード B：ファイルアップロード ----
else:
    up_files = st.file_uploader(
        "ZIP または 画像ファイルをアップロード（複数可）",
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
                                im = pil_open_rgb(zfi.read())
                            pil_images.append(im)
                            names.append(info.filename)
                else:
                    im = pil_open_rgb(f.read())
                    pil_images.append(im)
                    names.append(f.name)
            except Exception as e:
                st.warning(f"{f.name}: 読込失敗（{e}）")

        st.success(f"{len(pil_images)} 枚の画像を読み込みました。")
    else:
        st.info("ファイルをアップロードしてください。")

# =========================================================
# 8) 推論実行
# =========================================================
if not pil_images:
    st.stop()

probs_list: List[np.ndarray] = []
progress = st.progress(0, text="推論中...")
total = len(pil_images)

for s in range(0, total, int(BATCH)):
    e = min(s + int(BATCH), total)
    try:
        logits = predict_batch(model, pil_images[s:e], DEFAULT_TRANSFORM, dev)
        probs  = F.softmax(logits, dim=-1).numpy()
        probs_list.append(probs)
    except Exception as ex:
        st.error(f"推論中にエラーが発生しました: {ex}")
        st.stop()
    progress.progress(int(100 * e / total))

progress.empty()

probs_all = np.vstack(probs_list)

# --- OCR + 融合 ---
ocr_reader = None
if use_ocr and ocr_ok:
    try:
        ocr_reader = load_ocr_reader()
    except Exception as e:
        st.warning(f"OCR初期化に失敗しました（CNNのみで続行）: {e}")

fused_results = []
ocr_progress = st.progress(0, text="OCR解析中...") if (use_ocr and ocr_reader) else None
for i, im in enumerate(pil_images):
    if use_ocr and ocr_reader:
        ocr_res = recognize_from_image(im, reader=ocr_reader, mode=ocr_mode)
        fused = fuse_cnn_and_ocr(probs_all[i], CLASS_NAMES, ocr_res)
    else:
        fused = fuse_cnn_and_ocr(
            probs_all[i],
            CLASS_NAMES,
            OCRResult(False, None, [], {}, None, 0.0, ""),
        )
    fused_results.append(fused)
    if ocr_progress is not None:
        ocr_progress.progress(int(100 * (i + 1) / len(pil_images)))
if ocr_progress is not None:
    ocr_progress.empty()

pred_label = [f.label for f in fused_results]
pred_conf  = np.array([f.confidence for f in fused_results])

METHOD_LABELS = {
    "cnn": "画像分類のみ",
    "fusion_agree": "CNNとOCRが一致",
    "fusion_ocr": "刻印OCRを優先",
    "fusion_cnn": "画像分類を優先",
}

# =========================================================
# 9) 結果表示
# =========================================================

# --- カメラモードは大きく1枚表示 ---
if input_mode == "カメラ撮影（スマホ推奨）" and len(pil_images) == 1:
    conf   = float(pred_conf[0])
    label  = pred_label[0]
    topk_i = np.argsort(-probs_all[0])[:topk]

    st.markdown("---")
    col_img, col_result = st.columns([1, 1])

    with col_img:
        show_frame = st.checkbox("ガイド枠を表示（OCR解析範囲）", value=True, key="show_guide_frame")
        display_im = (
            draw_guide_overlay(pil_images[0]) if show_frame else pil_images[0]
        )
        st.image(display_im, caption="撮影画像", use_container_width=True)
        if show_frame:
            st.caption("緑枠内が OCR の解析範囲です。刻印が枠外なら再撮影を推奨します。")

    with col_result:
        st.subheader("判定結果")
        fr = fused_results[0]
        display = get_display_name(label)
        if conf >= conf_th:
            st.success(f"**{display}**")
            st.caption(f"クラスID: {label}")
            st.metric("確信度（融合後）", f"{conf*100:.1f}%")
        else:
            st.warning(f"確信度が低いため判定不可（{conf*100:.1f}%）")
            st.caption("刻印（R7100 等）が写るよう近づいて再撮影してください。")

        st.info(f"判定根拠: {METHOD_LABELS.get(fr.method, fr.method)}")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("CNN", f"{fr.cnn_label} ({fr.cnn_conf*100:.0f}%)")
        with c2:
            if fr.ocr_label:
                st.metric("OCR", f"{fr.ocr_label} ({fr.ocr_conf*100:.0f}%)")
            else:
                st.metric("OCR", "型番未検出")

        if fr.ocr_text:
            with st.expander("OCRで読み取った文字"):
                st.code(fr.ocr_text[:500] or "（なし）")

        st.markdown("**CNN Top-k**")
        for i in topk_i:
            bar_val = float(probs_all[0, i])
            st.write(f"{get_display_name(CLASS_NAMES[i])} : {bar_val*100:.1f}%")
            st.progress(bar_val)

        render_manual_section(
            label,
            confidence=conf,
            conf_threshold=conf_th,
            expanded=conf >= conf_th,
        )

# --- ファイルモードは一覧テーブル + サムネイル ---
else:
    rows = [
        {
            "ファイル名": nm,
            "最終判定": get_display_name(pred_label[i]),
            "確信度": f"{pred_conf[i]*100:.1f}%",
            "CNN": f"{fused_results[i].cnn_label} ({fused_results[i].cnn_conf*100:.0f}%)",
            "OCR": (
                f"{fused_results[i].ocr_label} ({fused_results[i].ocr_conf*100:.0f}%)"
                if fused_results[i].ocr_label
                else "未検出"
            ),
            "根拠": METHOD_LABELS.get(fused_results[i].method, fused_results[i].method),
            "判定": "OK" if pred_conf[i] >= conf_th else "要確認",
        }
        for i, nm in enumerate(names)
    ]
    df = pd.DataFrame(rows)

    st.subheader("判定結果一覧")
    st.dataframe(df, use_container_width=True)

    ok_indices = [i for i in range(len(names)) if pred_conf[i] >= conf_th]
    if ok_indices:
        st.subheader("📖 マニュアル・技術資料（判定 OK のみ）")
        for i in ok_indices:
            st.markdown(f"**{names[i]}** → {get_display_name(pred_label[i])}")
            render_manual_section(
                pred_label[i],
                confidence=float(pred_conf[i]),
                conf_threshold=conf_th,
                expanded=len(ok_indices) == 1,
            )
    else:
        st.info(
            "確信度がしきい値以上の画像がある場合、"
            "ここにマニュアルへのリンクが表示されます。"
        )

    # CSV ダウンロード
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "結果を CSV でダウンロード",
        csv_bytes,
        "shimano_result.csv",
        mime="text/csv",
    )

    # 分布バーチャート
    st.subheader("モデル別判定数（確信度しきい値以上）")
    df_ok  = df[df["判定"] == "OK"]
    if df_ok.empty:
        st.warning("確信度しきい値以上の結果がありません。しきい値を下げてみてください。")
    else:
        counts = df_ok["最終判定"].value_counts()
        st.bar_chart(counts)

    # サムネイル（最大 12 枚）
    st.subheader("サンプル画像（最大 12 枚）")
    show_n   = min(len(pil_images), 12)
    num_cols = 4
    cols     = st.columns(num_cols)

    for i in range(show_n):
        topk_i  = np.argsort(-probs_all[i])[:topk]
        fr = fused_results[i]
        caption = (
            f"最終: {get_display_name(pred_label[i])} ({pred_conf[i]*100:.0f}%)\n"
            + " / ".join(
                [f"{CLASS_NAMES[j]}: {probs_all[i, j]*100:.1f}%" for j in topk_i]
            )
        )
        with cols[i % num_cols]:
            st.image(
                pil_images[i],
                caption=f"{Path(names[i]).name}\n{caption}",
                use_column_width=True,
            )
