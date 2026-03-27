import torch
import torch.nn.functional as F
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, UnidentifiedImageError

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Training ke exact val_transforms — same mean/std, same library
TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.7216, 0.5765, 0.5725],   # HAM10000-specific
        std =[0.1404, 0.1501, 0.1669],
    ),
    ToTensorV2(),
])


def _load_as_rgb_numpy(image_source) -> np.ndarray:
    """File path ya PIL Image — dono se RGB numpy array banao."""
    if isinstance(image_source, str):
        img = cv2.imread(image_source)
        if img is None:
            raise ValueError(f"cv2 image load fail: {image_source}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_source, Image.Image):
        img = np.array(image_source.convert("RGB"))
    elif isinstance(image_source, np.ndarray):
        img = image_source
    else:
        raise TypeError("image_source must be str path, PIL.Image, or numpy array")
    return img


def predict(model, classes, image_source):
    """
    image_source: file path (str) | PIL.Image | numpy array (RGB, HWC)

    Returns: list of dicts sorted by confidence desc
      [{"class": "Melanoma", "confidence": 87.32}, ...]
    """
    img    = _load_as_rgb_numpy(image_source)
    tensor = TRANSFORM(image=img)["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().tolist()

    if isinstance(probs, float):
        probs = [probs]

    results = [
        {"class": cls, "confidence": round(prob * 100, 2)}
        for cls, prob in zip(classes, probs)
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results
