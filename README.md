# シマノ クランク 自動特定システム

スマホカメラをかざすだけで、シマノのクランクモデル（FCR7100 / FCR8100 / FCR9200）を自動判別するWebアプリです。  
ResNet-50の転移学習を使い、Streamlit で動作します。

---

## 対応モデル

| クラス名 | 製品名 |
|----------|--------|
| FCR7100 | SHIMANO 105 FC-R7100 |
| FCR8100 | SHIMANO ULTEGRA FC-R8100 |
| FCR9200 | SHIMANO DURA-ACE FC-R9200 |

---

## ファイル構成

```
製品検出アプリ/
├── app_poc.py            # Streamlit推論アプリ
├── train_poc.py          # 学習スクリプト
├── augment_images.py     # 画像水増しスクリプト
├── requirements.txt      # 依存ライブラリ
├── shimano_model_poc.ckpt          # 学習済みモデル（学習後に生成）
├── shimano_model_poc.history.json  # 学習履歴（学習後に生成）
└── dataset/
    ├── original/         # 元画像（各クラス1枚以上）
    │   ├── FCR7100/
    │   ├── FCR8100/
    │   └── FCR9200/
    └── train/            # 水増し後の学習画像（自動生成）
        ├── FCR7100/
        ├── FCR8100/
        └── FCR9200/
```

---

## セットアップ

### 1. 仮想環境の作成

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac / Linux
```

### 2. ライブラリのインストール

```bash
pip install -r requirements.txt
```

`requirements.txt` の内容：

```
streamlit>=1.35.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
pandas>=2.0.0
numpy>=1.24.0
```

---

## 使い方

### Step 1 — 元画像を配置する

各クラスのフォルダに元画像を1枚以上置きます。

```
dataset/original/FCR7100/  ←  FCR7100の写真
dataset/original/FCR8100/  ←  FCR8100の写真
dataset/original/FCR9200/  ←  FCR9200の写真
```

### Step 2 — 画像を水増しする

1枚の元画像から200枚の学習用画像を自動生成します（傷・泥・照明変化・角度変化などを模擬）。

```bash
python augment_images.py
```

生成される変換の種類：

| 変換 | 内容 |
|------|------|
| 回転 | ±45度のランダム回転 |
| クロップ | ズーム60〜100%でランダム切り出し |
| 射影変換 | 斜め角度から撮影した状態を模擬 |
| 明るさ・コントラスト | 照明条件の違いを模擬 |
| ぼかし | 手ブレ・ピンボケを模擬 |
| ノイズ | センサーノイズを模擬 |
| 傷 | ランダム黒線を1〜5本追加 |
| 泥汚れ | 茶色ブロブをランダム配置 |
| 矩形マスク | 部分的に隠れた状態を模擬 |
| グレースケール | 錆・くすみを模擬 |

### Step 3 — 学習する

```bash
python train_poc.py
```

完了すると `shimano_model_poc.ckpt` が生成されます。  
学習の進捗はターミナルに表示されます。

```
[  1/30] train loss=1.0821 acc=0.342 | val loss=1.0512 acc=0.408 | lr=1.00e-04
[  2/30] train loss=0.9134 acc=0.521 | val loss=0.8823 acc=0.583 | lr=9.98e-05
  >>> best model saved (val_acc=0.5833) → shimano_model_poc.ckpt
...
```

#### 学習パラメータ（train_poc.py 内で変更可能）

| パラメータ | デフォルト値 | 説明 |
|------------|-------------|------|
| BATCH_SIZE | 16 | バッチサイズ（メモリ不足時は8に下げる） |
| NUM_EPOCHS | 30 | 最大エポック数 |
| LR | 1e-4 | 学習率 |
| PATIENCE | 7 | 早期終了：改善なしで停止するエポック数 |

### Step 4 — アプリを起動する

```bash
streamlit run app_poc.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。

---

## アプリの機能

### カメラ撮影モード（スマホ推奨）

スマホのブラウザでアプリのURLを開き、カメラで撮影すると即座に判定結果が表示されます。

- 判定モデル名
- 確信度（%）
- Top-k スコアのバー表示

### ファイルアップロードモード

ZIP または画像ファイル（JPG / PNG / BMP / TIFF / WebP）を複数まとめてアップロードし、一括推論できます。

- 判定結果一覧テーブル
- CSV ダウンロード
- モデル別判定数バーチャート
- サムネイル表示（最大12枚）

### サイドバー設定

| 設定項目 | 説明 |
|----------|------|
| モデルパス | 学習済み .ckpt ファイルのパス |
| モデルアップロード | .ckpt ファイルを直接アップロード |
| Top-k 表示数 | 上位何クラスのスコアを表示するか |
| 確信度しきい値 | この値未満は「判定不可」として扱う |
| バッチサイズ | 一度に処理する画像枚数 |

---

## Streamlit Cloud へのデプロイ

1. このフォルダを GitHub リポジトリにpushする
2. `shimano_model_poc.ckpt`（約100MB）は Git LFS に登録する

```bash
git lfs install
git lfs track "*.ckpt"
git add .gitattributes
git add .
git commit -m "initial commit"
git push origin main
```

3. [share.streamlit.io](https://share.streamlit.io) にアクセスし、GitHubリポジトリと `app_poc.py` を選択して「Deploy」

デプロイ後はスマホブラウザからURLにアクセスするだけでカメラが起動します。

---

## 注意事項

- 学習データが1クラスあたり200枚未満の場合、精度が低くなります。実運用では各クラス500枚以上を推奨します。
- FCR7100 / FCR8100 / FCR9200 は外観が非常に似ているため、ロゴ部分が明確に写っている画像での撮影が精度向上につながります。
- Windows 環境では `num_workers=0` が必須です（multiprocessing の制約）。

---

## 今後の改善候補

- Grad-CAM によるモデルの判断根拠の可視化
- Confusion Matrix の出力（クラス間の誤判定パターン確認）
- ONNX エクスポートによるスマホネイティブアプリ化
- 製品クラスの追加（FC-R9100、FC-R6800 など）
