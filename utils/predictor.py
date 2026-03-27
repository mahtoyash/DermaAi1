import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict(model, classes, image_path):
    """
    Returns: list of dicts sorted by confidence desc
    [{"class": "melanoma", "confidence": 87.32}, ...]
    """
    img    = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().tolist()

    results = [
        {"class": cls, "confidence": round(prob * 100, 2)}
        for cls, prob in zip(classes, probs)
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results
