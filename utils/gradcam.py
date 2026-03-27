import numpy as np
import cv2
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


class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._hooks      = []
        # DenseNet121 ka last dense block — 7×7 spatial maps capture hote hain
        self._register(model.backbone.features.denseblock4)

    def _register(self, layer):
        def fwd(_, __, output):
            self.activations = output

        def bwd(_, __, grad_out):
            self.gradients = grad_out[0]

        self._hooks.append(layer.register_forward_hook(fwd))
        self._hooks.append(layer.register_full_backward_hook(bwd))

    def remove(self):
        for h in self._hooks:
            h.remove()

    def generate(self, image_path, class_idx):
        img      = Image.open(image_path).convert("RGB")
        original = np.array(img.resize((224, 224)))
        tensor   = TRANSFORM(img).unsqueeze(0).to(device)

        self.model.eval()
        logits = self.model(tensor)          # forward — hooks fire

        self.model.zero_grad()
        logits[0, class_idx].backward()     # backward — gradient hooks fire

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks did not capture. Check target layer.")

        grads   = self.gradients.detach().cpu()    # [1, C, H, W]
        acts    = self.activations.detach().cpu()  # [1, C, H, W]

        weights = grads.mean(dim=[2, 3], keepdim=True)          # global avg pool
        cam     = F.relu((weights * acts).sum(dim=1).squeeze())  # weighted sum + ReLU
        cam     = cam.numpy()

        # Resize to 224×224 & normalize 0-1
        cam = cv2.resize(cam, (224, 224))
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # JET heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.55 * original + 0.45 * heatmap).astype(np.uint8)
        return overlay


def generate_gradcam(model, image_path, class_idx, save_path):
    gc = GradCAM(model)
    try:
        overlay = gc.generate(image_path, class_idx)
        Image.fromarray(overlay).save(save_path)
    finally:
        gc.remove()   # hooks hamesha remove karo
    return save_path
