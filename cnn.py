import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectCNN(nn.Module):
    def __init__(self, env_n, num_channels, img_size_x, img_size_y):
        super(DetectCNN, self).__init__()

        self.env_n = env_n  # boxing, skiing, lb_foraging, mario
        if self.env_n == "boxing":
            self.output_dim = 8
        elif self.env_n == "skiing":
            self.output_dim = 6
        elif self.env_n == "redball":
            self.output_dim = 5
        elif self.env_n == "lb_foraging":
            self.output_dim = 4
            raise NotImplementedError
        else:
            raise NotImplementedError(self.env_n)
        if self.env_n in ["boxing", "skiing"]:
            self.cnn = nn.Sequential(
                nn.Conv2d(num_channels, 32, kernel_size=5, stride=1),  # → [32, 51, 39]
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),  # → [32, 25, 19]
                nn.Conv2d(32, 64, kernel_size=3, stride=1),  # → [64, 11, 8]
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),  # → [64, 5, 4]
                nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → [64, 3, 2]
                nn.ReLU(),
            )

        else:
            self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1),  # (32, 56, 56)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 28, 28)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 14, 14)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (128, 1, 1)
        )

        # figure out flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size_x, img_size_y)
            conv_out = self.cnn(dummy)
            flat_size = conv_out.numel()
            print("flat_size is: ", flat_size)

        # --- MLP head ---
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, self.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.cnn(x)
        output = self.mlp(z)

        if self.env_n == "boxing":
            float_output_1 = output[:, 1:3]
            binary_output = torch.sigmoid(output[:, [0, 3, 4, 7]])  # Apply sigmoid for binary outputs [color, fist]
            float_output_2 = output[:, 5:7]
            output = torch.cat((binary_output[:, [0]], float_output_1, binary_output[:, [1, 2]], float_output_2, binary_output[:, [3]]), dim=1)
            return output
        elif self.env_n == "redball":
            binary_output = torch.sigmoid(output[:, :3])  # Apply sigmoid for binary outputs [obstacle]
            float_output = output[:, 3:]
            output = torch.cat((binary_output, float_output), dim=1)
            return output
        else:
            return output

