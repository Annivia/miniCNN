import os
import time
import pandas as pd
import cv2
import sys
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay

# Original model definition (needed for loading weights)
class TwoStageDetectCNN(nn.Module):
    def __init__(self, num_channels, img_size_x, img_size_y):
        super(TwoStageDetectCNN, self).__init__()
        
        # Base feature extractor (shared)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),  # Added batch normalization
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),  # Added batch normalization
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),  # Added batch normalization
            nn.ReLU(),
        )
        
        # Block detection pathway (for left/front/right blocked)
        self.block_detection_encoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.block_detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # Just 3 block features
        )
        
        # Ball detection pathway (for ball existence)
        self.ball_detection_encoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.ball_detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Just ball existence
        )
        
        # Position encoder - only processes features when ball exists
        self.position_encoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Position MLP that takes feature encoding + redball_exists
        self.position_mlp = nn.Sequential(
            nn.Linear(128 + 1, 256),  # 128 features + 1 for redball_exists
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # x and y position
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Improved weight initialization to avoid stuck predictions"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for output biases
        nn.init.constant_(self.block_detection_head[-1].bias, 0.5)
        nn.init.constant_(self.ball_detection_head[-1].bias, 0)
        
    def forward(self, x):
        # Feature extraction (shared base features)
        features = self.feature_extractor(x)
        
        # Block detection branch
        block_features = self.block_detection_encoder(features)
        block_output = self.block_detection_head(block_features)
        block_sigmoid = torch.sigmoid(block_output)
        
        # Ball detection branch
        ball_features = self.ball_detection_encoder(features)
        ball_output = self.ball_detection_head(ball_features)
        ball_sigmoid = torch.sigmoid(ball_output)
        
        # Position prediction (always computed for batch speed)
        position_features = self.position_encoder(features)
        position_input = torch.cat([position_features, ball_sigmoid], dim=1)
        position_output = self.position_mlp(position_input)
        
        # Gate position output by ball_exists
        position_output = position_output * ball_sigmoid
        
        # Concatenate full output: 3 binary + 2 position + 1 existence
        return torch.cat([
            block_sigmoid,             # left, front, right blocked
            position_output,           # x, y (gated)
            ball_sigmoid               # existence score
        ], dim=1)

# New Binary-only model
class BinaryDetectCNN(nn.Module):
    def __init__(self, num_channels, img_size_x, img_size_y):
        super(BinaryDetectCNN, self).__init__()
        
        # Base feature extractor (shared)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Block detection pathway (for left/front/right blocked)
        self.block_detection_encoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.block_detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # Just 3 block features
        )
        
        # Ball detection pathway (for ball existence)
        self.ball_detection_encoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.ball_detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Just ball existence
        )
        
        # Initialize bias for outputs - important to match original model behavior
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Special initialization for output biases to avoid stuck predictions"""
        # We only need to initialize the biases for the output layers, as other weights will be transferred
        nn.init.constant_(self.block_detection_head[-1].bias, 0.5)
        nn.init.constant_(self.ball_detection_head[-1].bias, 0)
        
    def forward(self, x):
        # Feature extraction (shared base features)
        features = self.feature_extractor(x)
        
        # Block detection branch
        block_features = self.block_detection_encoder(features)
        block_output = self.block_detection_head(block_features)
        block_sigmoid = torch.sigmoid(block_output)
        
        # Ball detection branch
        ball_features = self.ball_detection_encoder(features)
        ball_output = self.ball_detection_head(ball_features)
        ball_sigmoid = torch.sigmoid(ball_output)
        
        # Concatenate binary outputs only: 3 binary + 1 existence
        # Format matches original order: left, front, right, (placeholder x), (placeholder y), exists
        # The placeholders will be zeros and ignored in evaluation
        batch_size = x.shape[0]
        placeholder = torch.zeros(batch_size, 2, device=x.device)
        
        return torch.cat([
            block_sigmoid,             # left, front, right blocked
            placeholder,               # placeholder for x, y
            ball_sigmoid               # existence score
        ], dim=1)

# Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, dataset_dir, env_name, size, csv_file, transform=None):
        self.env_name = env_name
        if size == 10000:
            self.image_folder = f'redball_images/redball_train_filtered'
        else:
            self.image_folder = f'redball_images/redball_test_filtered'
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1:].values.astype(float)

        image_path = os.path.join(self.image_folder, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)  # (C, H, W)
        image = image / 255.0  # Normalize

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.float)


def evaluate_binary_model(model, data_loader, device, env_name):
    model.eval()
    feature_names = ["left_blocked", "front_blocked", "right_blocked", "ball_exists"]
    predictions_binary = []
    targets_binary = []
    
    with torch.no_grad():
        for images, labels_true in tqdm(data_loader, desc="Evaluating"):
            images, labels_true = images.to(device), labels_true.to(device)
            labels_pred = model(images)
            
            # For full model output with placeholder, extract only binary features
            binary_predictions = torch.cat([labels_pred[:, :3], labels_pred[:, 5:6]], dim=1)
            binary_targets = torch.cat([labels_true[:, :3], labels_true[:, 5:6]], dim=1)
            
            predictions_binary.append(binary_predictions.cpu().numpy())
            targets_binary.append(binary_targets.cpu().numpy())
    
    all_predictions = np.vstack(predictions_binary)
    all_targets = np.vstack(targets_binary)
    
    # Calculate accuracy for each binary feature
    results = {}
    for i, feature_name in enumerate(feature_names):
        pred_binary = (all_predictions[:, i] > 0.5).astype(int)
        target_binary = (all_targets[:, i] > 0.5).astype(int)
        
        accuracy = (pred_binary == target_binary).mean()
        
        # Compute confusion matrix
        cm = confusion_matrix(target_binary, pred_binary, labels=[0, 1])
        true_negatives = cm[0, 0]
        false_positives = cm[0, 1]
        false_negatives = cm[1, 0]
        true_positives = cm[1, 1]
        
        # Compute precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[feature_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist()
        }
        
        print(f"{feature_name}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    return results


def transfer_binary_weights(original_model, binary_model):
    """Transfer weights from original model to binary-only model"""
    # Get state dictionaries
    original_state_dict = original_model.state_dict()
    binary_state_dict = binary_model.state_dict()
    
    # Copy weights for shared layers - using state_dict keys instead of parameters
    for name in binary_state_dict:
        if name in original_state_dict:
            print(f"Transferring weights for: {name}")
            binary_state_dict[name].copy_(original_state_dict[name])
        else:
            print(f"Warning: Could not find matching weight for {name}")
    
    # Apply the updated state dict
    binary_model.load_state_dict(binary_state_dict)
    
    return binary_model


def main():
    # Configuration
    env_n = "redball"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    
    # Define output directory
    output_dir = 'binary_model_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test dataset
    dataset_dir = f'redball_data'
    test_dataset = CustomImageDataset(
        dataset_dir, env_n, size=2000,
        csv_file='redball_data/test.csv',
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input image shape
    first_image, _ = test_dataset[0]
    image_shape = first_image.shape
    num_channels = image_shape[0] if len(image_shape) == 3 else 1
    image_height = image_shape[1] if len(image_shape) == 3 else image_shape[0]
    image_width = image_shape[2] if len(image_shape) == 3 else image_shape[1]
    
    print(f"Device: {device}")
    print(f"Image shape: {image_shape}")
    
    # Initialize and load original model
    original_model = TwoStageDetectCNN(num_channels, image_height, image_width).to(device)
    original_model_path = "logs/redball/improved_two_stage_2025-05-13_03-08-03/best_detection.pth"  # Path to the best model
    
    print(f"Loading original model from {original_model_path}")
    try:
        original_model.load_state_dict(torch.load(original_model_path, map_location=device))
        print("Original model loaded successfully")
    except Exception as e:
        print(f"Error loading original model: {e}")
        return
    
    # Initialize binary model
    binary_model = BinaryDetectCNN(num_channels, image_height, image_width).to(device)
    
    # Transfer weights from original model to binary model
    print("Transferring weights to binary model")
    binary_model = transfer_binary_weights(original_model, binary_model)
    
    # Set binary model to evaluation mode
    binary_model.eval()
    
    # Verify output structure
    sample_image, _ = test_dataset[0]
    sample_image = sample_image.unsqueeze(0).to(device)
    with torch.no_grad():
        binary_output = binary_model(sample_image)
        print(f"Binary model output shape: {binary_output.shape}")
        print(f"Binary model output sample: {binary_output[0]}")
    
    # Save binary model
    binary_model_path = os.path.join(output_dir, "binary.pth")
    torch.save(binary_model.state_dict(), binary_model_path)
    print(f"Binary model saved to {binary_model_path}")
    
    # Compare model sizes
    original_params = sum(p.numel() for p in original_model.parameters())
    binary_params = sum(p.numel() for p in binary_model.parameters())
    print(f"Original model parameters: {original_params:,}")
    print(f"Binary model parameters: {binary_params:,}")
    print(f"Parameter reduction: {original_params - binary_params:,} ({(1 - binary_params/original_params)*100:.1f}%)")
    
    # Evaluate original model performance (just binary features)
    print("\nEvaluating original model (binary features only):")
    original_results = evaluate_binary_model(original_model, test_loader, device, env_n)
    
    # Evaluate binary model performance
    print("\nEvaluating binary model:")
    binary_results = evaluate_binary_model(binary_model, test_loader, device, env_n)
    
    # Plot confusion matrices for both models
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    feature_names = ["left_blocked", "front_blocked", "right_blocked", "ball_exists"]
    
    for i, feature_name in enumerate(feature_names):
        # Original model confusion matrix
        cm_original = np.array(original_results[feature_name]["confusion_matrix"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_original, display_labels=["Negative", "Positive"])
        disp.plot(cmap=plt.cm.Blues, values_format='d', ax=axes[0, i])
        axes[0, i].set_title(f"Original: {feature_name}")
        axes[0, i].grid(False)
        
        # Binary model confusion matrix
        cm_binary = np.array(binary_results[feature_name]["confusion_matrix"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_binary, display_labels=["Negative", "Positive"])
        disp.plot(cmap=plt.cm.Blues, values_format='d', ax=axes[1, i])
        axes[1, i].set_title(f"Binary: {feature_name}")
        axes[1, i].grid(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_comparison.png'))
    plt.close()
    
    # Save results to JSON
    comparison_results = {
        "original_model": original_results,
        "binary_model": binary_results,
        "model_info": {
            "original_params": original_params,
            "binary_params": binary_params,
            "reduction_percentage": (1 - binary_params/original_params)*100
        }
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    print(f"Results saved to {output_dir}/evaluation_results.json")
    print("Done!")


if __name__ == "__main__":
    main()