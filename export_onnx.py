# export_onnx.py
import torch
from torchvision import models

# 学習時のクラス（OTHERを追加する前の3クラス）
CLASS_NAMES = sorted(["FCR7100", "FCR8100", "FCR9200"])
NUM_CLASSES = len(CLASS_NAMES)

CKPT_PATH = "shimano_model_poc.ckpt"
ONNX_PATH = "shimano_model_poc.onnx"

# モデル復元
ckpt     = torch.load(CKPT_PATH, map_location="cpu")
backbone = models.resnet50(weights=None)
backbone.fc = torch.nn.Linear(backbone.fc.in_features, NUM_CLASSES)

sd = {k.replace("model.", "").replace("module.", ""): v
      for k, v in ckpt["state_dict"].items()}
backbone.load_state_dict(sd, strict=False)
backbone.eval()

# ONNX書き出し
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    backbone, dummy, ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11,
)
print(f"saved: {ONNX_PATH}")
print(f"classes: {CLASS_NAMES}")