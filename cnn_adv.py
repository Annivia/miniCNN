import torch
import torch.nn as nn
from torchvision import models

class HybridMobileNetV2(nn.Module):
    def __init__(self, env_n, backbone, freeze_backbone=False):
        super().__init__()
        self.env_n = env_n  # boxing, skiing, lb_foraging, mario
        if self.env_n == "boxing":
            self.output_dim = 8
        elif self.env_n == "skiing":
            self.output_dim = 4
            raise NotImplementedError
        elif self.env_n == "mario":
            self.output_dim = 11
            raise NotImplementedError
        elif self.env_n == "lb_foraging":
            self.output_dim = 4
            raise NotImplementedError
        else:
            raise NotImplementedError(self.env_n)
        if backbone == "mobilenet":
            self.backbone = models.mobilenet_v2(pretrained=True).features
            in_features = 1280  # last_channel of MobileNetV2
        elif backbone == "efficientnet":
            # efficientnet = models.efficientnet_b0(pretrained=True)
            self.backbone = models.efficientnet_b0(pretrained=True).features
            # in_features = efficientnet.classifier[1].in_features
            in_features = 1280

        self.pool = nn.AdaptiveAvgPool2d(1)


        # --- MLP head ---
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.output_dim)
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False


    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        output = self.mlp(x)

        if self.env_n == "boxing":
            float_output_1 = output[:, 1:3]
            binary_output = torch.sigmoid(output[:, [0, 3, 4, 7]])  # Apply sigmoid for binary outputs [color, fist]
            float_output_2 = output[:, 5:7]
            output = torch.cat((binary_output[:, [0]], float_output_1, binary_output[:, [1, 2]], float_output_2,
                                binary_output[:, [3]]), dim=1)
            return output
        else:
            return output


