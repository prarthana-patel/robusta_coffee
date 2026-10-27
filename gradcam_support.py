"""On-demand gradient-weighted activation maps for supported timm CNN-like features."""
import numpy as np
from PIL import Image

def generate_cam(model, input_tensor, original_image, class_idx):
    import torch
    import torch.nn.functional as F
    # Only use spatial convolution layers; transformer token maps require specialized reshape.
    layers = [(name, layer) for name, layer in model.named_modules()
              if isinstance(layer, torch.nn.Conv2d)]
    if not layers:
        raise RuntimeError("No compatible convolutional feature layer found.")
    # Prefer a late convolution with spatial output, falling back through earlier layers.
    for name, layer in reversed(layers):
        captured = {}
        def hook(_module, _args, output):
            if torch.is_tensor(output) and output.ndim == 4 and min(output.shape[-2:]) > 1:
                captured["activation"] = output
                output.retain_grad()
        handle = layer.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                x = input_tensor.detach().clone().requires_grad_(True)
                model.zero_grad(set_to_none=True)
                logits = model(x)
                activation = captured.get("activation")
                if activation is None or logits.ndim != 2:
                    continue
                logits[0, class_idx].backward()
                grad = activation.grad
                if grad is None:
                    continue
                weights = grad.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
                cam = cam.detach().float().cpu().numpy()
                cam -= cam.min()
                if cam.max() > 0:
                    cam /= cam.max()
                # Blend a red-to-yellow activation mask with the original image.
                base = original_image.convert("RGB").resize((224, 224))
                rgb = np.asarray(base, dtype=np.float32)
                heat = np.zeros_like(rgb)
                heat[..., 0] = 255 * cam
                heat[..., 1] = 175 * cam
                alpha = (0.48 * cam)[..., None]
                blend = np.uint8(np.clip(rgb * (1 - alpha) + heat * alpha, 0, 255))
                return Image.fromarray(blend), name
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)
    raise RuntimeError("No spatial convolution layer yielded usable Grad-CAM gradients.")
