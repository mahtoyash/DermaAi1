import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Correct mean/std — training wale (HAM10000-specific, NOT ImageNet)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.7216, 0.5765, 0.5725],
        std =[0.1404, 0.1501, 0.1669],
    ),
])

def predict(model, classes, image_source):
    if isinstance(image_source, str):
        img = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, Image.Image):
        img = image_source.convert("RGB")
    else:
        raise TypeError("str path ya PIL.Image dono chalega")

    tensor = TRANSFORM(img).unsqueeze(0).to(device)

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
