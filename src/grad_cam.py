import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[:, target_class]
        target.backward()

        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = torch.sum(weights * activations, dim=1).squeeze(0)  # [H, W]

        cam = F.relu(cam)
        cam = cam.cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cv2.resize(cam, (224, 224))
        return cam
