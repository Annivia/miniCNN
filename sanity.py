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

def evaluate_model(model, data_loader, device):
    model.eval()
    
    # Initialize lists to store predictions and ground truth
    all_preds = []
    all_targets = []
    
    # For position evaluation (only when ball exists)
    position_preds = []
    position_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            
            # Store predictions and targets
            all_preds.append(outputs)
            all_targets.append(targets)
            
            # Extract position data only when ball exists
            ball_exists_mask = targets[:, 5] > 0.5  # Using the ball_exists column
            
            if torch.any(ball_exists_mask):
                # Extract position predictions and targets only when ball exists
                position_preds.append(outputs[ball_exists_mask, 3:5])  # x, y predictions
                position_targets.append(targets[ball_exists_mask, 3:5])  # x, y targets (columns 4,5)
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
    
    # Evaluate binary classification metrics for block detection (first 3 outputs)
    block_names = ["Left Blocked", "Front Blocked", "Right Blocked"]
    
    print("\n=== Block Detection Evaluation ===")
    for i in range(3):
        # Convert sigmoid outputs to binary predictions (threshold = 0.5)
        block_preds = (all_preds[:, i] > 0.5).astype(int)
        block_targets = all_targets[:, i].astype(int)  # Columns 1,2,3 are the block labels
        
        # Calculate accuracy
        block_accuracy = (block_preds == block_targets).mean()
        print(f"\n{block_names[i]} Accuracy: {block_accuracy:.4f}")
        
        # Create confusion matrix
        cm = confusion_matrix(block_targets, block_preds)
        print(f"{block_names[i]} Confusion Matrix:")
        print(cm)
        
        # Calculate TP, FP, TN, FN
        if cm.shape == (2, 2):  # Only calculate metrics if we have both classes
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            # Display confusion matrix
            plt.figure(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Blocked", "Blocked"])
            disp.plot(cmap=plt.cm.Blues, values_format='d')
            plt.title(f"{block_names[i]} Confusion Matrix")
            plt.savefig(f"{block_names[i].lower().replace(' ', '_')}_confusion.png")
            plt.close()
    
    # Evaluate ball existence detection (output index 5)
    print("\n=== Ball Existence Evaluation ===")
    # Convert sigmoid outputs to binary predictions (threshold = 0.5)
    ball_preds = (all_preds[:, 5] > 0.5).astype(int)
    ball_targets = all_targets[:, 5].astype(int)  # Column 6 is ball_exists
    
    # Calculate accuracy
    ball_accuracy = (ball_preds == ball_targets).mean()
    print(f"Ball Existence Accuracy: {ball_accuracy:.4f}")

 
    
    # Position evaluation only when ball exists
    if position_preds:
        print("\n=== Ball Position Evaluation (Only when ball exists) ===")
        # Concatenate position predictions and targets
        position_preds = torch.cat(position_preds, dim=0).cpu().numpy()
        position_targets = torch.cat(position_targets, dim=0).cpu().numpy()
        
        # Calculate MSE for x and y separately
        mse_x = np.mean((position_preds[:, 0] - position_targets[:, 0]) ** 2)
        mse_y = np.mean((position_preds[:, 1] - position_targets[:, 1]) ** 2)
        
        # Calculate RMSE
        rmse_x = np.sqrt(mse_x)
        rmse_y = np.sqrt(mse_y)
        
        # Calculate R2 score
        r2_x = r2_score(position_targets[:, 0], position_preds[:, 0])
        r2_y = r2_score(position_targets[:, 1], position_preds[:, 1])
        
        print(f"X Position MSE: {mse_x:.4f}")
        print(f"Y Position MSE: {mse_y:.4f}")
        print(f"X Position RMSE: {rmse_x:.4f}")
        print(f"Y Position RMSE: {rmse_y:.4f}")
        print(f"X Position R2: {r2_x:.4f}")
        print(f"Y Position R2: {r2_y:.4f}")
        
        # Plot predicted vs actual positions
        plt.figure(figsize=(10, 8))
        plt.scatter(position_targets[:, 0], position_targets[:, 1], label='Actual', alpha=0.5)
        plt.scatter(position_preds[:, 0], position_preds[:, 1], label='Predicted', alpha=0.5)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Ball Position: Actual vs Predicted')
        plt.legend()
        plt.grid(True)
        plt.savefig("ball_position_scatter.png")
        plt.close()
        
        # Plot error distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(position_preds[:, 0] - position_targets[:, 0], bins=50)
        plt.xlabel('X Position Error')
        plt.ylabel('Frequency')
        plt.title(f'X Position Error Distribution\nRMSE: {rmse_x:.4f}')
        
        plt.subplot(1, 2, 2)
        plt.hist(position_preds[:, 1] - position_targets[:, 1], bins=50)
        plt.xlabel('Y Position Error')
        plt.ylabel('Frequency')
        plt.title(f'Y Position Error Distribution\nRMSE: {rmse_y:.4f}')
        
        plt.tight_layout()
        plt.savefig("position_error_distribution.png")
        plt.close()
        
        # Plot heatmap of prediction accuracy
        plt.figure(figsize=(10, 8))
        
        # Calculate Euclidean distance between predicted and actual positions
        euclidean_dist = np.sqrt(np.sum((position_preds - position_targets) ** 2, axis=1))
        
        # Create a scatter plot with color based on error
        plt.scatter(position_targets[:, 0], position_targets[:, 1], c=euclidean_dist, 
                   cmap='viridis', alpha=0.7, s=30)
        plt.colorbar(label='Position Error (Euclidean Distance)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Position Error Heatmap')
        plt.grid(True)
        plt.savefig("position_error_heatmap.png")
        plt.close()
        
        # Calculate and print mean Euclidean distance
        mean_euclidean_dist = np.mean(euclidean_dist)
        median_euclidean_dist = np.median(euclidean_dist)
        print(f"Mean Euclidean Distance: {mean_euclidean_dist:.4f}")
        print(f"Median Euclidean Distance: {median_euclidean_dist:.4f}")

def main():
    # Configuration
    env_n = "redball"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    
    # Load test dataset
    dataset_dir = f'redball_data'
    test_dataset = CustomImageDataset(
        dataset_dir, env_n, size=2000,
        csv_file='redball_data/test_filtered.csv',
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
  
    
    # Compare model sizes
    original_params = sum(p.numel() for p in original_model.parameters())
    print(f"Original model parameters: {original_params:,}")
    
    # Evaluate original model performance
    print("\nEvaluating original model:")
    evaluate_model(original_model, test_loader, device)


if __name__ == "__main__":
    main()