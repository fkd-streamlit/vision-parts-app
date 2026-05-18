# -*- coding: utf-8 -*-
"""
画像水増しスクリプト（augment_images.py）
1クラス1枚の元画像から、各クラス200枚の学習用画像を自動生成します。

【使い方】
1. 元画像を以下のフォルダに置く
   dataset/original/
       FCR7100/  FCR7100の元画像.jpg
       FCR8100/  FCR8100の元画像.jpg
       FCR9200/  FCR9200の元画像.jpg

2. python augment_images.py を実行

3. dataset/train/ に水増し画像が生成される
"""

import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

# =========================================================
# 設定
# =========================================================
SRC_ROOT  = Path(r"dataset\original")   # 元画像フォルダ
DST_ROOT  = Path(r"dataset\train")      # 出力先（train_poc.py のDATA_ROOTと同じ）
N_AUGMENT = 200                          # 1クラスあたりの生成枚数
SEED      = 42

CLASS_NAMES = ["FCR7100", "FCR8100", "FCR9200"]

random.seed(SEED)
np.random.seed(SEED)


# =========================================================
# 個別の変換関数
# =========================================================

def random_rotate(img: Image.Image) -> Image.Image:
    """ランダム回転（±45度）"""
    angle = random.uniform(-45, 45)
    return img.rotate(angle, expand=True, fillcolor=(255, 255, 255))


def random_crop_resize(img: Image.Image) -> Image.Image:
    """ランダムクロップ後にリサイズ（ズームのバリエーション）"""
    w, h = img.size
    scale = random.uniform(0.6, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    left   = random.randint(0, w - nw)
    top    = random.randint(0, h - nh)
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.LANCZOS)


def random_flip(img: Image.Image) -> Image.Image:
    """左右・上下ランダム反転"""
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.15:
        img = ImageOps.flip(img)
    return img


def random_brightness(img: Image.Image) -> Image.Image:
    """明るさ変化（照明条件の違い）"""
    factor = random.uniform(0.4, 1.8)
    return ImageEnhance.Brightness(img).enhance(factor)


def random_contrast(img: Image.Image) -> Image.Image:
    """コントラスト変化"""
    factor = random.uniform(0.4, 2.0)
    return ImageEnhance.Contrast(img).enhance(factor)


def random_saturation(img: Image.Image) -> Image.Image:
    """彩度変化（色あせ・錆色対応）"""
    factor = random.uniform(0.2, 1.8)
    return ImageEnhance.Color(img).enhance(factor)


def random_sharpness(img: Image.Image) -> Image.Image:
    """シャープネス変化（ピンボケ対応）"""
    factor = random.uniform(0.0, 3.0)
    return ImageEnhance.Sharpness(img).enhance(factor)


def random_blur(img: Image.Image) -> Image.Image:
    """ガウスぼかし（手ブレ・フォーカスずれ）"""
    if random.random() < 0.4:
        radius = random.uniform(0.5, 3.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def random_noise(img: Image.Image) -> Image.Image:
    """ランダムノイズ（センサーノイズ・画質劣化）"""
    if random.random() < 0.4:
        arr   = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(5, 25), arr.shape)
        arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img   = Image.fromarray(arr)
    return img


def random_scratch(img: Image.Image) -> Image.Image:
    """傷・汚れの模擬（ランダム黒線を数本追加）"""
    if random.random() < 0.5:
        from PIL import ImageDraw
        draw   = ImageDraw.Draw(img)
        w, h   = img.size
        n_lines = random.randint(1, 5)
        for _ in range(n_lines):
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            x2 = x1 + random.randint(-w // 4, w // 4)
            y2 = y1 + random.randint(-h // 4, h // 4)
            color = (
                random.randint(0, 60),
                random.randint(0, 60),
                random.randint(0, 60),
            )
            width = random.randint(1, 4)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    return img


def random_mud(img: Image.Image) -> Image.Image:
    """泥・汚れの模擬（茶色半透明のブロブを追加）"""
    if random.random() < 0.4:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        w, h = img.size
        n_blobs = random.randint(1, 6)
        for _ in range(n_blobs):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            rw = random.randint(10, w // 5)
            rh = random.randint(10, h // 5)
            mud_color = (
                random.randint(60, 120),
                random.randint(40, 80),
                random.randint(10, 40),
            )
            draw.ellipse(
                [(cx - rw, cy - rh), (cx + rw, cy + rh)],
                fill=mud_color,
            )
        # ぼかして自然に見せる
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return img


def random_erasing(img: Image.Image) -> Image.Image:
    """ランダム矩形マスク（一部が隠れた状態を模擬）"""
    if random.random() < 0.35:
        arr   = np.array(img)
        h, w  = arr.shape[:2]
        rh    = random.randint(h // 10, h // 4)
        rw    = random.randint(w // 10, w // 4)
        top   = random.randint(0, h - rh)
        left  = random.randint(0, w - rw)
        fill  = random.randint(0, 255)
        arr[top:top + rh, left:left + rw] = fill
        img   = Image.fromarray(arr)
    return img


def random_perspective(img: Image.Image) -> Image.Image:
    """射影変換（斜め角度から撮影した状態を模擬）"""
    if random.random() < 0.4:
        w, h   = img.size
        margin = int(min(w, h) * 0.15)
        coeffs = [
            random.randint(0,      margin), random.randint(0,      margin),
            random.randint(w - margin, w),  random.randint(0,      margin),
            random.randint(w - margin, w),  random.randint(h - margin, h),
            random.randint(0,      margin), random.randint(h - margin, h),
        ]
        img = img.transform(
            (w, h), Image.QUAD,
            data=coeffs,
            resample=Image.BILINEAR,
        )
    return img


def random_grayscale(img: Image.Image) -> Image.Image:
    """白黒変換（錆・金属くすみ対応）"""
    if random.random() < 0.1:
        img = ImageOps.grayscale(img).convert("RGB")
    return img


# =========================================================
# 全変換をまとめて適用
# =========================================================
TRANSFORMS = [
    random_flip,
    random_rotate,
    random_crop_resize,
    random_perspective,
    random_brightness,
    random_contrast,
    random_saturation,
    random_sharpness,
    random_blur,
    random_noise,
    random_scratch,
    random_mud,
    random_erasing,
    random_grayscale,
]


def augment_one(img: Image.Image) -> Image.Image:
    """全変換をランダムに組み合わせて1枚生成する"""
    result = img.copy().convert("RGB")
    random.shuffle(TRANSFORMS)
    for fn in TRANSFORMS:
        result = fn(result)
    return result


# =========================================================
# メイン処理
# =========================================================
def main():
    print("===== 画像水増し開始 =====\n")

    for cls in CLASS_NAMES:
        src_dir = SRC_ROOT / cls
        dst_dir = DST_ROOT / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        # 元画像を収集
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        src_files = [p for p in src_dir.iterdir() if p.suffix.lower() in exts]

        if not src_files:
            print(f"[WARNING] {src_dir} に画像が見つかりません。スキップします。")
            continue

        print(f"[{cls}]  元画像: {len(src_files)} 枚  →  {N_AUGMENT} 枚生成")

        count = 0
        while count < N_AUGMENT:
            # 元画像をランダムに1枚選んで水増し
            src_path = random.choice(src_files)
            base_img = Image.open(src_path).convert("RGB")

            aug_img  = augment_one(base_img)
            out_path = dst_dir / f"aug_{count:04d}.jpg"
            aug_img.save(out_path, "JPEG", quality=92)
            count += 1

            if count % 50 == 0:
                print(f"  {count}/{N_AUGMENT} 枚完了...")

        print(f"  完了 → {dst_dir}\n")

    print("===== 水増し完了 =====")
    print(f"出力先: {DST_ROOT.resolve()}")
    print(f"各クラス {N_AUGMENT} 枚  合計 {N_AUGMENT * len(CLASS_NAMES)} 枚")


if __name__ == "__main__":
    main()
