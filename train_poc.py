# -*- coding: utf-8 -*-
"""
シマノ クランク 学習スクリプト (PoC版)
対応クラス：FCR7100 / FCR8100 / FCR9200

【実行前に】python augment_images.py で画像を水増ししてください
【実行方法】python train_poc.py
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from pathlib import Path

# =========================================================
# 1) パス設定
# =========================================================
DATA_ROOT = r"dataset\train"
SAVE_PATH = r"shimano_model_poc.ckpt"

# =========================================================
# 2) クラス定義（app_poc.py と完全一致・アルファベット順）
# =========================================================
CLASS_NAMES = sorted([
    "FCR7100",
    "FCR8100",
    "FCR9200",
])
NUM_CLASSES = len(CLASS_NAMES)

# =========================================================
# 3) 学習パラメータ
# =========================================================
BATCH_SIZE  = 16
NUM_EPOCHS  = 30
LR          = 1e-4
VAL_RATIO   = 0.2
PATIENCE    = 7
NUM_WORKERS = 0   # ★ Windows では必ず 0

# =========================================================
# 4) Transform
# =========================================================
train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# =========================================================
# ★ Windows multiprocessing 対策：学習処理を関数にまとめる
# =========================================================
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device      : {DEVICE}")
    print(f"classes     : {CLASS_NAMES}")
    print(f"num_classes : {NUM_CLASSES}")

    # --- Dataset ---
    full_dataset = datasets.ImageFolder(DATA_ROOT, transform=train_tf)

    print(f"ImageFolder detected classes: {full_dataset.classes}")
    assert full_dataset.classes == CLASS_NAMES, (
        f"クラス順が一致しません。\n"
        f"  検出: {full_dataset.classes}\n"
        f"  定義: {CLASS_NAMES}\n"
        f"フォルダ名を CLASS_NAMES と完全一致させてください。"
    )

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * VAL_RATIO))
    n_train = n_total - n_val

    if n_train <= 0 or n_val <= 0:
        raise ValueError(
            f"データが少なすぎます（total={n_total}）。"
            "augment_images.py を先に実行してください。"
        )

    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )

    print(f"train: {len(train_ds)} 枚  /  val: {len(val_ds)} 枚")

    # --- モデル ---
    try:
        weights  = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
        print("ImageNet V2 weights loaded.")
    except Exception:
        backbone = models.resnet50(pretrained=True)
        print("Legacy pretrained weights loaded.")

    backbone.fc = nn.Linear(backbone.fc.in_features, NUM_CLASSES)
    model = backbone.to(DEVICE)

    # --- 損失 / オプティマイザ / スケジューラ ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    # --- 学習ループ ---
    best_val_acc   = 0.0
    patience_count = 0
    history        = []

    print("\n===== 学習開始 =====")

    for epoch in range(1, NUM_EPOCHS + 1):

        # 学習フェーズ
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            correct  += (torch.max(out, 1)[1] == y).sum().item()
            total    += y.size(0)

        train_loss = run_loss / total
        train_acc  = correct  / total

        # 検証フェーズ
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out  = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                v_corr += (torch.max(out, 1)[1] == y).sum().item()
                v_tot  += y.size(0)

        val_loss = v_loss / v_tot
        val_acc  = v_corr / v_tot

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
        })

        print(
            f"[{epoch:>3}/{NUM_EPOCHS}] "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val   loss={val_loss:.4f} acc={val_acc:.3f} | "
            f"lr={current_lr:.2e}"
        )

        # ベストモデル保存
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(
                {
                    "state_dict":  model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "epoch":       epoch,
                    "val_acc":     val_acc,
                },
                SAVE_PATH,
            )
            print(f"  >>> best model saved (val_acc={val_acc:.4f}) → {SAVE_PATH}")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n早期終了：{PATIENCE} エポック改善なし（epoch {epoch}）")
                break

    # 学習履歴を JSON に保存
    history_path = Path(SAVE_PATH).with_suffix(".history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\n===== 学習完了 =====")
    print(f"best val_acc : {best_val_acc:.4f}")
    print(f"model saved  : {SAVE_PATH}")
    print(f"history saved: {history_path}")


# =========================================================
# ★ Windows では if __name__ == '__main__': が必須
# =========================================================
if __name__ == '__main__':
    main()
