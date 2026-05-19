# -*- coding: utf-8 -*-
"""
シマノ製品モデル 自動特定システム (PoC版)
- ResNet-50 転移学習 + 刻印OCR（EasyOCR）の融合判定
- Streamlit Cloud 対応：st.camera_input でスマホカメラ撮影→推論
- ローカル実行時：ZIPまたは画像ファイルのアップロードにも対応

【安定化パッチ（2026-05）】
- OCRを“自動実行しない”（手動ボタン or 低確信度のみ）
- アップロード直後に画像を縮小してメモリピークを下げる
- モデルをCPU固定オプションを追加（Cloudで安全）
- use_container_width 警告に対応（width='stretch' に置換）
- ocr_engine 側が target= 未対応でも落ちないようにフォールバック
"""

import io
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
)
from product_catalog import render_manual_section

NUM_CLASSES = len(CLASS_NAMES)

# =========================================================
# 安定化：画像の最大辺を制限（メモリピーク対策）
# =========================================================
DEFAULT_MAX_SIDE = 1024  # Cloudでの安定性を優先（必要なら 768 に下げる）


def shrink_max_side(im: Image.Image, max_side: int = DEFAULT_MAX_SIDE) -> Image.Image:
    """縦横の最大辺が max_side を超える場合に縮小（アスペクト維持）"""
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    scale = max_side / m
    new_w, new_h = int(w * scale), int(h * scale)
    return im.resize((new_w, new_h), Image.Resampling.LANCZOS)


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
def device_str(force_cpu: bool = True) -> str:
    """Cloud安定化のため、デフォルトはCPU固定にする"""
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def pil_open_rgb(file_bytes: bytes, max_side: int = DEFAULT_MAX_SIDE) -> Image.Image:
    """バイト列から PIL Image (RGB) を生成。EXIF回転も自動補正。さらに縮小。"""
    im = Image.open(io.BytesIO(file_bytes))
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    if im.mode != "RGB":
        im = im.convert("RGB")

    # ★重要：アップロード直後に縮小してメモリピークを下げる
    im = shrink_max_side(im, max_side=max_side)
    return im


def list_images_in_zip(zf: zipfile.ZipFile) -> List[zipfile.ZipInfo]:
    """ZIP内の画像ファイル一覧を返す。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [
        info for info in zf.infolist()
        if (not info.is_dir())
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
def load_model(model_path: str, num_classes: int, device: str) -> torch.nn.Module:
    """
    .ckpt ファイルから ResNet-50 を復元する。
    ckpt は {"state_dict": ..., "class_names": ...} 形式を想定。
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")

    saved_classes = ckpt.get("class_names")
    if saved_classes and saved_classes != CLASS_NAMES:
        st.warning(
            f"モデルのクラス順 {saved_classes} と "
            f"アプリ定義 {CLASS_NAMES} が異なります。結果が不正確になる可能性があります。"
        )

    state_dict = ckpt.get("state_dict") or ckpt

    backbone = models.resnet50(weights=None)
    backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)

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
    backbone = backbone.to(device)
    return backbone


@st.cache_resource(show_spinner="OCRエンジンを初期化しています（初回はモデルDL）...")
def load_ocr_reader():
    from ocr_engine import get_ocr_reader
    return get_ocr_reader()


# =========================================================
# 6) Streamlit UI
# =========================================================
st.set_page_config(page_title="シマノ クランク判定", layout="wide")
st.title("特定製品 自動特定システム")
st.caption("対応モデル：FCR7100 / FCR8100 / FCR9200 ｜ CNN + 刻印OCR 融合判定")


# --- サイドバー ---
with st.sidebar:
    st.header("モデル設定")

    model_path_input = st.text_input(
        "学習済みモデルのパス (.ckpt)",
        value="shimano_model_poc.ckpt",
        help="Streamlit Cloud の場合はリポジトリに含めてください。"
    )

    st.caption("または、モデルファイルをここにアップロード：")
    up_model = st.file_uploader(
        "モデルファイル (.ckpt)",
        type=["ckpt"],
        accept_multiple_files=False,
    )

    st.markdown("---")
    st.header("推論設定")

    force_cpu = st.checkbox(
        "CPU固定（推奨：クラッシュ回避）",
        value=True,
        help="Streamlit CloudではCPU固定の方が安定します。"
    )
    dev = device_str(force_cpu=force_cpu)
    st.caption(f"推論デバイス: {dev.upper()}")

    topk = st.slider("Top-k 表示数", 1, NUM_CLASSES, min(2, NUM_CLASSES))
    conf_th = st.slider(
        "確信度しきい値", 0.0, 1.0, 0.5, 0.05,
        help="この値未満の結果は「判定不可」として扱います"
    )
    BATCH = st.number_input("バッチサイズ", min_value=1, max_value=64, value=8)

    st.markdown("---")
    st.header("画像処理（安定化）")
    max_side = st.select_slider(
        "入力画像の最大辺（縮小して処理）",
        options=[640, 768, 896, 1024, 1280],
        value=1024,
        help="大きいほど精度は上がりやすいが、負荷も増えます。Cloudで落ちるなら 768/640 を推奨。"
    )

    st.markdown("---")
    st.header("OCR（刻印読取）")

    use_ocr = st.checkbox(
        "刻印・型番の OCR を使う（重い・初回DLあり）",
        value=False,
        help="初回は検出/認識モデルのDLが走り、Cloudでは落ちることがあります。必要時のみ有効化してください。",
    )

    # OCR対象（刻印優先/全体）…use_ocr OFFでもキーは定義しておく
    if use_ocr:
        ocr_target = st.selectbox(
            "OCR対象（おすすめ：刻印優先）",
            ["刻印優先（STAMP）", "全体文字（GENERAL）"],
            index=0,
            help="刻印優先は背景文字を拾いにくく、型番（FC-Rxxxx/Rxxxx）に寄せます。"
        )
        ocr_target_key = "stamp" if "STAMP" in ocr_target else "general"
    else:
        ocr_target_key = "general"

    ocr_ok, ocr_msg = ocr_dependencies_ok()
    if use_ocr and not ocr_ok:
        st.warning(ocr_msg)

    ocr_policy = st.radio(
        "OCRの実行タイミング（推奨：低確信度のみ／手動）",
        ["手動（ボタンを押したときだけ）", "低確信度の画像だけOCR", "常にOCR（非推奨）"],
        index=0,
        help="Cloudで安定させるため、手動または低確信度のみを推奨します。",
    )

    ocr_mode = st.selectbox(
        "OCRモード",
        ["fast", "thorough"],
        index=0,
        format_func=lambda x: "高速（撮影向け）" if x == "fast" else "高精度（時間がかかります）",
    )

    debug_ocr = st.checkbox("OCRデバッグ表示（実行状況と検出結果）", value=False)

    run_ocr_button = False
    if use_ocr and ocr_ok and ocr_policy == "手動（ボタンを押したときだけ）":
        run_ocr_button = st.button("OCRを実行（初回はモデルDL）", type="primary")


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
    model = load_model(use_model_path, NUM_CLASSES, dev)
except FileNotFoundError as e:
    st.warning(str(e))
    st.info("モデルファイルがまだない場合は、サイドバーからアップロードするか、train_poc.py で学習を実行してください。")
    st.stop()
except Exception as e:
    st.error(f"モデル読込失敗: {e}")
    st.stop()

st.sidebar.success("モデル読込 OK")


# =========================================================
# 7) 画像入力（2モード）
# =========================================================
st.subheader("画像入力")

input_mode = st.radio(
    "入力方法を選んでください",
    ["カメラ撮影（スマホ推奨）", "ファイルアップロード（ZIP / 画像）"],
    horizontal=True,
)

names: List[str] = []
pil_images: List[Image.Image] = []

# ---- モード A：カメラ撮影 ----
if input_mode == "カメラ撮影（スマホ推奨）":
    with st.expander("📷 撮影ガイド・撮影例", expanded=True):
        st.markdown("**刻印（R7100 等）がはっきり写るよう近づけて撮影**すると、OCRの精度が上がります。")
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
                        st.image(thumb, width="stretch")
                    except Exception:
                        st.caption("（画像を表示できません）")
                else:
                    st.caption("（撮影例画像なし）")
                st.markdown(f"**{ex['product']}**")
                st.caption(ex["tip"])

        st.markdown("##### ❌ 避けたい撮影")
        for title, desc in BAD_EXAMPLES:
            st.markdown(f"- **{title}** — {desc}")

    st.info("刻印（R7100 等）がはっきり写るよう近づけて撮影してください。撮影後、結果画像に緑枠でOCR範囲を表示します。")
    img_file = st.camera_input("クランクの刻印を撮影")

    if img_file is not None:
        try:
            im = pil_open_rgb(img_file.read(), max_side=max_side)
            pil_images = [im]
            names = ["camera_shot.jpg"]
        except Exception as e:
            st.error(f"画像の読み込みに失敗しました: {e}")

# ---- モード B：ファイルアップロード ----
else:
    up_files = st.file_uploader(
        "ZIP または 画像ファイルをアップロード（複数可）",
        type=["zip", "jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True,
    )

    MAX_FILES = 12

    if up_files:
        if len(up_files) > MAX_FILES:
            st.warning(f"アップロードが多いため先頭 {MAX_FILES} 件のみ処理します。")
            up_files = up_files[:MAX_FILES]

        for f in up_files:
            try:
                if f.name.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
                        infos = list_images_in_zip(zf)
                        if len(infos) > MAX_FILES:
                            st.warning(f"{f.name}: 画像が多いため先頭 {MAX_FILES} 枚のみ処理します。")
                            infos = infos[:MAX_FILES]
                        for info in infos:
                            with zf.open(info) as zfi:
                                im = pil_open_rgb(zfi.read(), max_side=max_side)
                            pil_images.append(im)
                            names.append(info.filename)
                else:
                    im = pil_open_rgb(f.read(), max_side=max_side)
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
        probs = F.softmax(logits, dim=-1).numpy()
        probs_list.append(probs)
    except Exception as ex:
        st.error(f"推論中にエラーが発生しました: {ex}")
        st.stop()
    progress.progress(int(100 * e / total))

progress.empty()
probs_all = np.vstack(probs_list)


# =========================================================
# 8.5) OCR 実行判定（自動で重い初期化を走らせない）
# =========================================================
ocr_reader = None
run_ocr_for_indices: List[int] = []

if use_ocr and ocr_ok:
    if ocr_policy == "常にOCR（非推奨）":
        run_ocr_for_indices = list(range(len(pil_images)))
    elif ocr_policy == "低確信度の画像だけOCR":
        run_ocr_for_indices = [i for i in range(len(pil_images)) if float(np.max(probs_all[i])) < conf_th]
    else:
        if run_ocr_button:
            run_ocr_for_indices = list(range(len(pil_images)))

    if run_ocr_for_indices:
        try:
            ocr_reader = load_ocr_reader()
        except Exception as e:
            st.warning(f"OCR初期化に失敗しました（CNNのみで続行）: {e}")
            ocr_reader = None
            run_ocr_for_indices = []
    else:
        st.info("OCRは実行しません（設定：手動 or 低確信度のみ）。")


# --- OCR実行対象の確認（デバッグ） ---
if debug_ocr:
    st.write("use_ocr:", use_ocr, "| ocr_ok:", ocr_ok, "| ocr_policy:", ocr_policy, "| ocr_mode:", ocr_mode)
    st.write("conf_th:", conf_th)
    st.write("ocr_target_key:", ocr_target_key)
    st.write("run_ocr_for_indices:", run_ocr_for_indices)
    if len(pil_images) > 0:
        st.write("max CNN prob (per image):", [float(np.max(p)) for p in probs_all])


# =========================================================
# 8.6) OCR + 融合
# =========================================================
fused_results = []
ocr_progress = st.progress(0, text="OCR解析中...") if (ocr_reader and run_ocr_for_indices) else None

dummy_ocr = OCRResult(False, None, [], {}, None, 0.0, "")

for i, im in enumerate(pil_images):
    if ocr_reader and (i in run_ocr_for_indices):
        try:
            # ocr_engine が target= 未対応でも落ちないようフォールバック
            try:
                ocr_res = recognize_from_image(im, reader=ocr_reader, mode=ocr_mode, target=ocr_target_key)
            except TypeError:
                ocr_res = recognize_from_image(im, reader=ocr_reader, mode=ocr_mode)
        except Exception as e:
            st.warning(f"OCR失敗（{names[i]}）: {e}（CNNのみで続行）")
            ocr_res = dummy_ocr
    else:
        ocr_res = dummy_ocr

    if debug_ocr and ocr_reader and (i in run_ocr_for_indices):
        st.write(f"[OCR] i={i} file={names[i]}")
        st.write(f"  detections={len(ocr_res.detections)}  best={ocr_res.best_class}  score={float(ocr_res.best_score):.2f}")
        st.write("  combined_text(head):", (ocr_res.combined_text[:200] if ocr_res.combined_text else "(empty)"))
        if ocr_res.detections:
            st.write("  top detections (text/conf/variant):")
            for d in ocr_res.detections[:5]:
                st.write(f"   - '{d.text}'  conf={d.confidence:.2f}  variant={d.variant}")

    fused = fuse_cnn_and_ocr(probs_all[i], CLASS_NAMES, ocr_res)
    fused_results.append(fused)

    if ocr_progress is not None:
        ocr_progress.progress(int(100 * (i + 1) / len(pil_images)))

if ocr_progress is not None:
    ocr_progress.empty()

pred_label = [f.label for f in fused_results]
pred_conf = np.array([f.confidence for f in fused_results])

METHOD_LABELS = {
    "cnn": "画像分類のみ",
    "fusion_agree": "CNNとOCRが一致",
    "fusion_ocr": "刻印OCRを優先",
    "fusion_cnn": "画像分類を優先",
}


# =========================================================
# 9) 結果表示
# =========================================================
if input_mode == "カメラ撮影（スマホ推奨）" and len(pil_images) == 1:
    conf = float(pred_conf[0])
    label = pred_label[0]
    topk_i = np.argsort(-probs_all[0])[:topk]

    st.markdown("---")
    col_img, col_result = st.columns([1, 1])

    with col_img:
        show_frame = st.checkbox("ガイド枠を表示（OCR解析範囲）", value=True, key="show_guide_frame")
        display_im = draw_guide_overlay(pil_images[0]) if show_frame else pil_images[0]
        st.image(display_im, caption="撮影画像", width="stretch")
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
        for j in topk_i:
            bar_val = float(probs_all[0, j])
            st.write(f"{get_display_name(CLASS_NAMES[j])} : {bar_val*100:.1f}%")
            st.progress(bar_val)

        render_manual_section(
            label,
            confidence=conf,
            conf_threshold=conf_th,
            expanded=conf >= conf_th,
        )

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
        st.info("確信度がしきい値以上の画像がある場合、ここにマニュアルへのリンクが表示されます。")

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "結果を CSV でダウンロード",
        csv_bytes,
        "shimano_result.csv",
        mime="text/csv",
    )

    st.subheader("モデル別判定数（確信度しきい値以上）")
    df_ok = df[df["判定"] == "OK"]
    if df_ok.empty:
        st.warning("確信度しきい値以上の結果がありません。しきい値を下げてみてください。")
    else:
        counts = df_ok["最終判定"].value_counts()
        st.bar_chart(counts)

    st.subheader("サンプル画像（最大 12 枚）")
    show_n = min(len(pil_images), 12)
    num_cols = 4
    cols = st.columns(num_cols)

    for i in range(show_n):
        topk_i = np.argsort(-probs_all[i])[:topk]
        caption = (
            f"最終: {get_display_name(pred_label[i])} ({pred_conf[i]*100:.0f}%)\n"
            + " / ".join([f"{CLASS_NAMES[j]}: {probs_all[i, j]*100:.1f}%" for j in topk_i])
        )
        with cols[i % num_cols]:
            st.image(
                pil_images[i],
                caption=f"{Path(names[i]).name}\n{caption}",
                width="stretch",
            )
