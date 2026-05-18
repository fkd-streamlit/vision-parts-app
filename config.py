"""アプリ共通設定（CNNクラス・OCRパターン・表示名）"""

CLASS_NAMES = sorted([
    "FCR7100",
    "FCR8100",
    "FCR9200",
])
CNN_CLASS_NAMES = CLASS_NAMES  # 学習モデル用（OTHER はCNNに含めない）

NUM_CLASSES = len(CLASS_NAMES)

DISPLAY_NAMES = {
    "FCR7100": "FC-R7100（105）",
    "FCR8100": "FC-R8100（ULTEGRA）",
    "FCR9200": "FC-R9200（DURA-ACE）",
    "OTHER":   "判定不可（対象外）",
}

# OCR: 刻印・ロゴから拾うキーワード（強い順）
# weight が高いほど型番一致として信頼
OCR_PATTERNS = {
    "FCR7100": [
        (r"FCR\s*[-]?\s*7100", 1.0),
        (r"FC\s*[-]?\s*R\s*7100", 1.0),
        (r"\bR\s*7100\b", 0.95),
        (r"\b105\b", 0.55),
        (r"\b7100\b", 0.45),
    ],
    "FCR8100": [
        (r"FCR\s*[-]?\s*8100", 1.0),
        (r"FC\s*[-]?\s*R\s*8100", 1.0),
        (r"\bR\s*8100\b", 0.95),
        (r"ULTEGRA", 0.75),
        (r"\b8100\b", 0.45),
    ],
    "FCR9200": [
        (r"FCR\s*[-]?\s*9200", 1.0),
        (r"FC\s*[-]?\s*R\s*9200", 1.0),
        (r"\bR\s*9200\b", 0.95),
        (r"DURA[\s-]*ACE", 0.8),
        (r"\b9200\b", 0.45),
    ],
}

# このスコア以上なら OCR をCNNより優先
OCR_OVERRIDE_THRESHOLD = 0.72
# OCR と CNN が一致したときの確信度ブースト上限
FUSION_AGREE_BOOST = 0.15

# OCR速度: "fast"=撮影向け(数秒), "thorough"=精度重視(数十秒)
OCR_MODE = "fast"

REJECT_CLASSES = {"OTHER"}


def get_display_name(class_name: str) -> str:
    return DISPLAY_NAMES.get(class_name, class_name)


def is_reject(class_name: str) -> bool:
    return class_name in REJECT_CLASSES
