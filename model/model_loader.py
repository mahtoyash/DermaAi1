import os
import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SkinCancerModel(nn.Module):
    def __init__(self, num_classes, hidden1, hidden2, dropout=0.5):
        super().__init__()
        self.backbone = models.densenet121(weights=None)
        in_feats = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(in_feats, hidden1),    # 0
            nn.BatchNorm1d(hidden1),          # 1
            nn.ReLU(inplace=True),            # 2
            nn.Dropout(dropout),              # 3
            nn.Linear(hidden1, hidden2),      # 4
            nn.BatchNorm1d(hidden2),          # 5
            nn.ReLU(inplace=True),            # 6
            nn.Dropout(dropout),              # 7
            nn.Linear(hidden2, num_classes),  # 8
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


# ── Singleton cache — model sirf ek baar load ho ──────────────────────────────
_cache = {"model": None, "classes": None, "val_acc": None}


def load_model(pth_path):
    if _cache["model"] is not None:
        return _cache["model"], _cache["classes"], _cache["val_acc"]

    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model file not found: {pth_path}")

    checkpoint = torch.load(pth_path, map_location=device)
    state      = checkpoint["model_state"]

    h1      = state["classifier.0.weight"].shape[0]
    h2      = state["classifier.4.weight"].shape[0]
    classes = checkpoint["classes"]
    val_acc = checkpoint.get("val_acc", 0.0)

    model = SkinCancerModel(len(classes), h1, h2)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    _cache.update({"model": model, "classes": classes, "val_acc": val_acc})
    return model, classes, val_acc


def get_model():
    return _cache["model"], _cache["classes"], _cache["val_acc"]
