# -*- coding: utf-8 -*-
"""撮影ガイド用：枠描画・テンプレート・撮影例画像の取得"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# OCR の center クロップと同じ比率（ocr_engine fast モード）
GUIDE_BOX = (0.12, 0.12, 0.88, 0.88)

APP_ROOT = Path(__file__).resolve().parent
EXAMPLE_DIRS = [
    APP_ROOT / "assets" / "examples",
    APP_ROOT / "dataset" / "original",
]

SHOOTING_TIPS = [
    "クランクアームの **刻印（R7100 / R8100 / R9200）** が枠の中央に入るように近づけて撮影",
    "文字にピントを合わせ、手ブレ・逆光を避ける",
    "斜めすぎない（30度以内が目安）",
    "泥や指で刻印を隠さない",
]

BAD_EXAMPLES = [
    ("❌ 遠すぎる", "型番が小さく読み取れません"),
    ("❌ ブレ・ピンボケ", "OCRが文字を認識できません"),
    ("❌ 刻印が枠外", "緑の枠内に R7100 等を収めてください"),
    ("❌ 強い逆光", "シルエットになり刻印が消えます"),
]


def _box_pixels(w: int, h: int) -> Tuple[int, int, int, int]:
    l, t, r, b = GUIDE_BOX
    return int(w * l), int(h * t), int(w * r), int(h * b)


def draw_guide_overlay(
    pil_rgb: Image.Image,
    *,
    show_label: bool = True,
    box_color: Tuple[int, int, int] = (46, 204, 113),
    corner_len: int = 0,
) -> Image.Image:
    """撮影画像に刻印ガイド枠を重ねた画像を返す。"""
    im = pil_rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    x1, y1, x2, y2 = _box_pixels(w, h)

    if corner_len <= 0:
        corner_len = max(16, min(w, h) // 12)

    lw = max(2, min(w, h) // 200)
    for (ax, ay, dx, dy) in [
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ]:
        draw.line([(ax, ay), (ax + dx * corner_len, ay)], fill=box_color, width=lw)
        draw.line([(ax, ay), (ax, ay + dy * corner_len)], fill=box_color, width=lw)

    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=max(1, lw - 1))

    if show_label:
        label = "刻印をこの枠内に（OCR解析範囲）"
        try:
            font = ImageFont.truetype("arial.ttf", max(14, h // 28))
        except OSError:
            font = ImageFont.load_default()
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        tx = max(4, (w - tw) // 2)
        ty = max(4, y1 - th - 8)
        draw.rectangle([tx - 4, ty - 2, tx + tw + 4, ty + th + 2], fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=box_color, font=font)

    return im


def make_viewfinder_template(width: int = 360, height: int = 480) -> Image.Image:
    """カメラ起動前に表示するビューファインダー風テンプレート。"""
    im = Image.new("RGB", (width, height), (28, 32, 38))
    draw = ImageDraw.Draw(im)
    x1, y1, x2, y2 = _box_pixels(width, height)

    draw.rectangle([0, 0, width, y1], fill=(18, 20, 24))
    draw.rectangle([0, y2, width, height], fill=(18, 20, 24))
    draw.rectangle([0, y1, x1, y2], fill=(18, 20, 24))
    draw.rectangle([x2, y1, width, y2], fill=(18, 20, 24))

    green = (46, 204, 113)
    cl = max(20, width // 10)
    lw = 3
    for (ax, ay, dx, dy) in [
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ]:
        draw.line([(ax, ay), (ax + dx * cl, ay)], fill=green, width=lw)
        draw.line([(ax, ay), (ax, ay + dy * cl)], fill=green, width=lw)

    draw.rectangle([x1, y1, x2, y2], outline=green, width=1)

    try:
        font_lg = ImageFont.truetype("arial.ttf", 15)
        font_sm = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font_lg = font_sm = ImageFont.load_default()

    draw.text((width // 2 - 70, y1 - 22), "刻印・型番", fill=green, font=font_lg)
    draw.text((12, height - 52), "R7100 / R8100 / R9200", fill=(200, 200, 200), font=font_sm)
    draw.text((12, height - 32), "SHIMANO ロゴ", fill=(160, 160, 160), font=font_sm)
    return im


def _pick_example_path(class_dir: str) -> Optional[Path]:
    for root in EXAMPLE_DIRS:
        folder = root / class_dir
        if not folder.is_dir():
            continue
        for name in (f"{class_dir}.jpg", f"FCR{class_dir[-4:]}.jpg"):
            p = folder / name
            if p.is_file():
                return p
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            found = sorted(folder.glob(pattern))
            if found:
                return found[0]
    return None
    
def get_shooting_examples() -> List[dict]:
    meta = [
        ("FCR7100", "105 FC-R7100",        "アーム外側の型番刻印を大きく"),
        ("FCR8100", "ULTEGRA FC-R8100",    "文字が読める距離・角度"),
        ("FCR9200", "DURA-ACE FC-R9200",   "逆光を避け、側面から光を当てる"),
    ]
    results = []
    for cls, product, tip in meta:
        # トップレベルの good_FCRxxxx.jpg を優先して探す
        path = APP_ROOT / f"good_{cls}.jpg"
        if not path.is_file():
            path = _pick_example_path(cls)
        results.append({
            "class":   cls,
            "product": product,
            "tip":     tip,
            "path":    path,
        })
    return results




def guide_overlay_html() -> str:
    """カメラ直上に表示する CSS ガイド。"""
    l, t, r, b = GUIDE_BOX
    wl, wt = l * 100, t * 100
    ww, wh = (r - l) * 100, (b - t) * 100
    bottom_pct = (1.0 - b) * 100
    right_pct = (1.0 - r) * 100
    return f"""
<style>
  .cam-guide-wrap {{
    max-width: 420px;
    margin: 0 auto 0.5rem auto;
    position: relative;
    aspect-ratio: 3/4;
    background: #1a1d21;
    border-radius: 12px;
    overflow: hidden;
  }}
  .cam-guide-wrap .dim {{ position: absolute; background: rgba(0,0,0,0.55); }}
  .cam-guide-wrap .d-top {{ left:0; right:0; top:0; height:{wt}%; }}
  .cam-guide-wrap .d-bottom {{ left:0; right:0; bottom:0; height:{bottom_pct}%; }}
  .cam-guide-wrap .d-left {{ left:0; top:{wt}%; width:{wl}%; height:{wh}%; }}
  .cam-guide-wrap .d-right {{ right:0; top:{wt}%; width:{right_pct}%; height:{wh}%; }}
  .cam-guide-frame {{
    position: absolute;
    left: {wl}%; top: {wt}%;
    width: {ww}%; height: {wh}%;
    border: 2px dashed #2ecc71;
    box-sizing: border-box;
    border-radius: 6px;
  }}
  .cam-guide-label {{
    position: absolute;
    left: 50%; top: calc({wt}% - 1.6rem);
    transform: translateX(-50%);
    color: #2ecc71;
    font-size: 0.85rem;
    font-weight: 600;
    text-shadow: 0 1px 3px #000;
  }}
  .cam-guide-hint {{
    position: absolute;
    bottom: 8px;
    left: 0; right: 0;
    text-align: center;
    color: #bbb;
    font-size: 0.75rem;
  }}
</style>
<div class="cam-guide-wrap">
  <div class="dim d-top"></div>
  <div class="dim d-bottom"></div>
  <div class="dim d-left"></div>
  <div class="dim d-right"></div>
  <div class="cam-guide-frame"></div>
  <div class="cam-guide-label">刻印をここに合わせる</div>
  <div class="cam-guide-hint">下のカメラで撮影 → 枠内に R7100 等が写ると精度UP</div>
</div>
"""
