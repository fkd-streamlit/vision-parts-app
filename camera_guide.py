# -*- coding: utf-8 -*-
"""撮影ガイド用：枠描画・撮影ヒント"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# OCR の center クロップと同じ比率
GUIDE_BOX = (0.12, 0.12, 0.88, 0.88)

APP_ROOT = Path(__file__).resolve().parent

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


def get_shooting_examples() -> List[dict]:
    """撮影例リストを返す"""
    example_dirs = [
        APP_ROOT / "assets" / "examples",
        APP_ROOT / "dataset" / "original",
        APP_ROOT,
    ]
    meta = [
        ("FCR7100", "FC-R7100（105）",         "アーム外側の型番刻印を大きく"),
        ("FCR8100", "FC-R8100（ULTEGRA）",     "文字が読める距離・角度"),
        ("FCR9200", "FC-R9200（DURA-ACE）",    "逆光を避け、側面から光を当てる"),
    ]
    results = []
    for cls, product, tip in meta:
        path = None
        # good_FCRxxxx.jpg をトップレベルから探す
        for root in [APP_ROOT]:
            p = root / f"good_{cls}.jpg"
            if p.is_file():
                path = p
                break
        # dataset/original/FCRxxxx/ から探す
        if path is None:
            for root in example_dirs:
                folder = root / cls
                if folder.is_dir():
                    for pat in ("*.jpg", "*.jpeg", "*.png"):
                        found = sorted(folder.glob(pat))
                        if found:
                            path = found[0]
                            break
                if path:
                    break
        results.append({"class": cls, "product": product, "tip": tip, "path": path})
    return results


def guide_overlay_html() -> str:
    """撮影のコツを HTML で返す"""
    tips_html = "".join(
        f"<li style='margin-bottom:6px'>{tip}</li>"
        for tip in SHOOTING_TIPS
    )
    return f"""
<div style="
    background: #1a2a1a;
    border: 1px solid #2ecc71;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 13px;
    color: #cccccc;
    line-height: 1.6;
">
  <p style="color:#2ecc71; font-weight:bold; margin:0 0 8px">📷 撮影のコツ</p>
  <ul style="margin:0; padding-left:18px">
    {tips_html}
  </ul>
</div>
"""
