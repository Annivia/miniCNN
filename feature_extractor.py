import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from obj_detection.cnn import DetectCNN
from obj_detection.cnn_adv import HybridMobileNetV2


class FeatureExtractor(nn.Module):
    def __init__(self, env_n, cnn_backbone, freeze_cnn, freeze_mlp=False):
        super(FeatureExtractor, self).__init__()
        self.env_n = env_n

        if self.env_n == "boxing":
            image_height, image_width = 210, 160
            single_frame_feature_size = 8
            multiple_frame_feature_size = 12
        elif self.env_n == "skiing":
            image_height, image_width = 210, 160
            single_frame_feature_size = 6
            multiple_frame_feature_size = 1
        else:
            raise NotImplementedError

        if cnn_backbone == "scratch":
            if env_n == "boxing":
                save_folder = "scratch_2025-05-06_21-16-45"
            elif env_n == "skiing":
                save_folder = "scratch_2025-05-11_00-13-54"
            else:
                raise NotImplementedError
            log_path = f"logs/obj_detection/{env_n}/single_frame"
            model_save_path = os.path.join(log_path, save_folder, 'best_detection.pth')
            self.cnn = DetectCNN(env_n, 3, image_height, image_width)
            self.cnn.load_state_dict(torch.load(model_save_path, weights_only=True))
        else:
            raise NotImplementedError

        self.mlp = nn.Sequential(
            nn.Linear(single_frame_feature_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, multiple_frame_feature_size)
        )

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        if freeze_mlp:
            for param in self.mlp.parameters():
                param.requires_grad = False

    def forward(self, frame_prev, frame_curr, train_flag=False):
        feature_prev = self.cnn(frame_prev)
        feature_curr = self.cnn(frame_curr)

        output = torch.cat([feature_prev, feature_curr], dim=1)
        output = self.mlp(output)

        if self.env_n == "boxing":
            float_output_1 = output[:, 1:3]
            binary_output = torch.sigmoid(output[:, [0, 3, 6, 9]])  # Apply sigmoid for binary outputs [color, fist]
            float_output_2 = output[:, 4:6]
            float_output_3 = output[:, 7:9]
            float_output_4 = output[:, 10:]  # 13 entries in total
            output = torch.cat((binary_output[:, [0]], float_output_1,
                                binary_output[:, [1]], float_output_2,
                                binary_output[:, [2]], float_output_3,
                                binary_output[:, [3]], float_output_4), dim=1)
            return output
        else:
            if train_flag:
                return output
            else:
                return torch.cat((feature_curr, output), dim=1)


