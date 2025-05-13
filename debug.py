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
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from sklearn.metrics import r2_score, classification_report, confusion_matrix

# -------- DEBUGGING LOGGER --------
class DebugLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "debug_log.txt")
        with open(self.log_file, 'w') as f:
            f.write(f"Debug Log Started: {datetime.now()}\n")
            f.write("-" * 80 + "\n\n")
    
    def log(self, message, print_to_console=True):
        with open(self.log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] {message}"
            f.write(log_message + "\n")
        if print_to_console:
            print(message)
    
    def log_tensor_stats(self, name, tensor, print_to_console=True):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        # Get basic statistics
        stats = {
            "shape": tensor.shape,
            "min": np.min(tensor),
            "max": np.max(tensor),
            "mean": np.mean(tensor),
            "std": np.std(tensor),
            "zeros": np.sum(tensor == 0),
            "ones": np.sum(tensor == 1) if tensor.size > 0 and np.max(tensor) <= 1 else "N/A",
            "nan_count": np.sum(np.isnan(tensor)),
            "inf_count": np.sum(np.isinf(tensor))
        }
        
        # For binary features, add class distribution
        if np.all(np.logical_or(tensor == 0, tensor == 1)) or np.all(np.logical_or(tensor >= 0, tensor <= 1)):
            value_counts = {}
            for i in range(min(10, len(np.unique(tensor)))):
                val = np.unique(tensor)[i]
                value_counts[val] = np.sum(tensor == val)
            stats["value_counts"] = value_counts
        
        log_message = f"TENSOR STATS - {name}:\n"
        for key, value in stats.items():
            log_message += f"  {key}: {value}\n"
        
        self.log(log_message, print_to_console)
        return stats

# -------- MODEL DEFINITION --------
class TwoStageDetectCNN(nn.Module):
    def __init__(self, num_channels, img_size_x, img_size_y):
        super(TwoStageDetectCNN, self).__init__()
        
        # Base feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # First stage - Detection head (obstacle detection + ball existence)
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)  # 3 blocked features + redball_exists
        )
        
        # Second stage - Position encoder (processes features only)
        self.position_encoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
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
        
    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)

        # Obstacle + ball detection
        detection_output = self.detection_head(features)
        detection_sigmoid = torch.sigmoid(detection_output)

        # Extract redball_exists score
        redball_exists = detection_sigmoid[:, 3:4]  # [B, 1]

        # Position prediction (always computed for batch speed)
        position_features = self.position_encoder(features)
        position_input = torch.cat([position_features, redball_exists], dim=1)
        position_output = self.position_mlp(position_input)

        # Gate position output by redball_exists
        position_output = position_output * redball_exists

        # Concatenate full output: 3 binary + 2 position + 1 existence
        return torch.cat([
            detection_sigmoid[:, :3],      # obstacle binary
            position_output,               # x, y (gated)
            redball_exists                 # existence score
        ], dim=1)

    def debug_forward(self, x, debug_logger):
        """Forward pass with detailed logging for debugging"""
        batch_size = x.shape[0]
        
        # Feature extraction
        debug_logger.log(f"Input shape: {x.shape}")
        debug_logger.log_tensor_stats("Input", x)
        
        features = self.feature_extractor(x)
        debug_logger.log(f"Feature extractor output shape: {features.shape}")
        debug_logger.log_tensor_stats("Features", features)

        # Obstacle + ball detection
        detection_output = self.detection_head(features)
        debug_logger.log(f"Detection head raw output shape: {detection_output.shape}")
        debug_logger.log_tensor_stats("Detection head raw output", detection_output)
        
        detection_sigmoid = torch.sigmoid(detection_output)
        debug_logger.log(f"Detection sigmoid shape: {detection_sigmoid.shape}")
        debug_logger.log_tensor_stats("Detection sigmoid", detection_sigmoid)

        # Extra debugging for binary predictions
        for i in range(3):
            debug_logger.log_tensor_stats(f"Binary feature {i} (raw)", detection_output[:, i])
            debug_logger.log_tensor_stats(f"Binary feature {i} (sigmoid)", detection_sigmoid[:, i])
        
        # Extract redball_exists score
        redball_exists = detection_sigmoid[:, 3:4]  # [B, 1]
        debug_logger.log_tensor_stats("Redball exists", redball_exists)

        # Position prediction
        position_features = self.position_encoder(features)
        debug_logger.log(f"Position encoder output shape: {position_features.shape}")
        debug_logger.log_tensor_stats("Position features", position_features)
        
        position_input = torch.cat([position_features, redball_exists], dim=1)
        debug_logger.log(f"Position MLP input shape: {position_input.shape}")
        
        position_output = self.position_mlp(position_input)
        debug_logger.log(f"Position output shape: {position_output.shape}")
        debug_logger.log_tensor_stats("Position output (before gating)", position_output)
        
        # Gate position output by redball_exists
        position_output = position_output * redball_exists
        debug_logger.log_tensor_stats("Position output (after gating)", position_output)

        # Concatenate full output
        full_output = torch.cat([
            detection_sigmoid[:, :3],      # obstacle binary
            position_output,               # x, y (gated)
            redball_exists                 # existence score
        ], dim=1)
        
        debug_logger.log(f"Final output shape: {full_output.shape}")
        debug_logger.log_tensor_stats("Final output", full_output)
        
        return full_output


class CustomImageDataset(Dataset):
    def __init__(self, dataset_dir, env_name, size, csv_file, transform=None, debug_logger=None):
        self.env_name = env_name
        self.debug_logger = debug_logger
        
        if debug_logger:
            debug_logger.log(f"Initializing dataset with CSV file: {csv_file}")
        
        print("labels are at: ", csv_file)
        if size == 10000:
            self.image_folder = f'redball_images/redball_train_filtered'
        else:
            self.image_folder = f'redball_images/redball_test_filtered'
            
        self.data = pd.read_csv(csv_file, header=None)
        
        if debug_logger:
            debug_logger.log(f"Dataset loaded with {len(self.data)} samples")
            # Debug: Print label distribution statistics
            for col in range(1, 7):  # Assuming 6 label columns (3 binary + 2 position + 1 existence)
                if col <= 3 or col == 6:  # Binary features
                    counts = self.data.iloc[:, col].value_counts()
                    debug_logger.log(f"Column {col} distribution:\n{counts}")
                else:  # Position features
                    stats = self.data.iloc[:, col].describe()
                    debug_logger.log(f"Column {col} statistics:\n{stats}")
        
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
            
        # Debug: occasionally log sample information
        if self.debug_logger and idx % 1000 == 0:
            self.debug_logger.log(f"Sample {idx}:")
            self.debug_logger.log(f"  Image: {image_name}")
            self.debug_logger.log(f"  Label: {label}")
            self.debug_logger.log(f"  Image shape: {image.shape}")
            self.debug_logger.log(f"  Image min/max: {np.min(image)}/{np.max(image)}")

        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.float)


def compute_loss(pred, target, env_name, debug_logger=None):
    """Compute the loss with detailed logging for debugging"""
    binary_loss_fn = nn.BCELoss(reduction='none')  # Use 'none' for detailed analysis
    position_loss_fn = nn.MSELoss(reduction='none')
    
    # For redball environment
    if env_name == "redball":
        # ---- First handle binary features ----
        # Extract binary features: left/front/right blocked + redball_exists
        binary_indices = [0, 1, 2, 5]  # Indices in the prediction vector
        binary_target = target[:, binary_indices]  
        binary_pred = pred[:, binary_indices]
        
        # Binary classification loss for all binary features
        binary_losses = binary_loss_fn(binary_pred, binary_target)
        
        # Debug: log per-feature binary losses
        if debug_logger:
            feature_names = ["left_blocked", "front_blocked", "right_blocked", "ball_exists"]
            for i, feature in enumerate(feature_names):
                feature_loss = binary_losses[:, i]
                mean_loss = torch.mean(feature_loss).item()
                
                # Check prediction quality
                pred_binary = (binary_pred[:, i] > 0.5).float()
                target_binary = binary_target[:, i]
                accuracy = torch.mean((pred_binary == target_binary).float()).item()
                
                # Get detailed stats on predictions and targets
                debug_logger.log(f"Binary feature '{feature}' stats:")
                debug_logger.log(f"  Mean Loss: {mean_loss:.6f}")
                debug_logger.log(f"  Accuracy: {accuracy:.6f}")
                
                # Calculate confusion matrix metrics
                true_pos = torch.sum((pred_binary == 1) & (target_binary == 1)).item()
                false_pos = torch.sum((pred_binary == 1) & (target_binary == 0)).item()
                true_neg = torch.sum((pred_binary == 0) & (target_binary == 0)).item()
                false_neg = torch.sum((pred_binary == 0) & (target_binary == 1)).item()
                
                debug_logger.log(f"  TP: {true_pos}, FP: {false_pos}, TN: {true_neg}, FN: {false_neg}")
                debug_logger.log(f"  Pred distribution: {torch.sum(pred_binary == 1).item()} ones, {torch.sum(pred_binary == 0).item()} zeros")
                debug_logger.log(f"  Target distribution: {torch.sum(target_binary == 1).item()} ones, {torch.sum(target_binary == 0).item()} zeros")
                
                # Check for prediction bias
                debug_logger.log_tensor_stats(f"{feature} predictions", binary_pred[:, i])
                
                if mean_loss < 0.01:
                    debug_logger.log(f"  WARNING: Very low loss for {feature}. Check for data imbalance or trivial solutions.")
        
        # Take mean across the batch and features for the binary loss
        loss_binary = torch.mean(binary_losses)
        
        # ---- Then handle position features ----
        # Extract position features: x, y coordinates
        position_target = target[:, 3:5]
        position_pred = pred[:, 3:5]
        
        # Get ground truth ball existence (for loss weighting)
        ball_exists = target[:, 5].unsqueeze(1)  # Shape: [batch_size, 1]
        
        # Calculate position loss for all samples
        position_losses = position_loss_fn(position_pred, position_target)
        
        # Debug position loss
        if debug_logger:
            debug_logger.log(f"Position loss stats:")
            debug_logger.log_tensor_stats("Position loss", position_losses)
            debug_logger.log_tensor_stats("Ball exists", ball_exists)
        
        # Multiply by ball_exists to zero out loss for samples where no ball exists
        weighted_position_losses = position_losses * ball_exists.expand_as(position_losses)
        
        # Compute the mean, but only over samples where ball exists
        # Add a small epsilon to avoid division by zero
        num_balls = torch.sum(ball_exists) + 1e-6
        loss_position = torch.sum(weighted_position_losses) / num_balls
        
        # Balance binary and position losses with weighting factors
        alpha = 1.0  # Weight for binary loss
        beta = 2.0   # Weight for position loss (higher to emphasize position accuracy)
        
        if debug_logger:
            debug_logger.log(f"Total binary loss: {loss_binary:.6f}")
            debug_logger.log(f"Total position loss: {loss_position:.6f}")
            debug_logger.log(f"Weighted binary loss: {alpha * loss_binary:.6f}")
            debug_logger.log(f"Weighted position loss: {beta * loss_position:.6f}")
        
        # Total loss combines both tasks
        total_loss = alpha * loss_binary + beta * loss_position
        
        return total_loss
    else:
        # For other environments
        return nn.MSELoss()(pred, target)


def format_metrics_string(val_feature_metrics):
    # Explicit correct mapping
    feature_metric_mapping = [
        ("left_blocked", "accuracy"),
        ("front_blocked", "accuracy"),
        ("right_blocked", "accuracy"),
        ("ball_exists", "accuracy"),
        ("ball_x_pos", "std_dev"),
        ("ball_y_pos", "std_dev")
    ]

    print("Detailed Validation Metrics:")
    metrics_str = ""
    for i, (feature_name, expected_type) in enumerate(feature_metric_mapping):
        if i >= len(val_feature_metrics):
            print(f"⚠️  Missing metrics for {feature_name}")
            continue
            
        metric_type, value = val_feature_metrics[i]
        if metric_type != expected_type:
            print(f"⚠️  Mismatch: {feature_name} expected {expected_type}, got {metric_type}")
        if metric_type == "accuracy":
            print(f"  {feature_name}: Accuracy = {value:.4f}")
            metrics_str += f"{feature_name}={value:.4f} "
        else:
            print(f"  {feature_name}: StdDev = {value:.4f}")
            metrics_str += f"{feature_name}={value:.4f} "
            
    return metrics_str


def evaluate_batch(pred, target, env_name, debug_logger=None):
    """Evaluate a single batch of predictions with detailed debugging"""
    predictions = pred.detach().cpu().numpy()
    targets = target.detach().cpu().numpy()
    feature_metrics = []
    
    if env_name == "redball":
        # Binary features (obstacles + redball_exists)
        binary_indices = [0, 1, 2, 5]  # left, front, right blocked + redball_exists
        binary_names = ["left_blocked", "front_blocked", "right_blocked", "ball_exists"]
        
        for i, idx in enumerate(binary_indices):
            feature_name = binary_names[i]
            pred_binary = (predictions[:, idx] > 0.5).astype(int)
            target_binary = (targets[:, idx] > 0.5).astype(int)
            accuracy = (pred_binary == target_binary).mean()
            feature_metrics.append(("accuracy", accuracy))
            
            if debug_logger:
                # Calculate confusion matrix for more detailed analysis
                tn, fp, fn, tp = confusion_matrix(target_binary, pred_binary, labels=[0, 1]).ravel()
                total = len(target_binary)
                
                debug_logger.log(f"Batch evaluation - {feature_name}:")
                debug_logger.log(f"  Accuracy: {accuracy:.4f}")
                debug_logger.log(f"  TP: {tp} ({tp/total:.2%}), FP: {fp} ({fp/total:.2%})")
                debug_logger.log(f"  TN: {tn} ({tn/total:.2%}), FN: {fn} ({fn/total:.2%})")
                
                # Calculate class balance in targets
                pos_ratio = np.mean(target_binary)
                debug_logger.log(f"  Target class balance: {pos_ratio:.2%} positive, {1-pos_ratio:.2%} negative")
                
                # Check for model bias
                pred_ratio = np.mean(pred_binary)
                debug_logger.log(f"  Prediction distribution: {pred_ratio:.2%} positive, {1-pred_ratio:.2%} negative")
                
                # Check for prediction quality
                if accuracy < 0.6:
                    debug_logger.log(f"  WARNING: Low accuracy for {feature_name}. Check for data issues or model problems.")
                
                if abs(pos_ratio - pred_ratio) > 0.2:
                    debug_logger.log(f"  WARNING: Large difference between target and prediction distributions for {feature_name}.")
                
                # Check prediction distribution before thresholding
                raw_preds = predictions[:, idx]
                debug_logger.log_tensor_stats(f"{feature_name} raw predictions", raw_preds)
                
                # Check if predictions are stuck near 0.5
                near_threshold = np.logical_and(raw_preds > 0.4, raw_preds < 0.6)
                if np.mean(near_threshold) > 0.5:
                    debug_logger.log(f"  WARNING: {np.mean(near_threshold):.2%} of predictions are near the 0.5 threshold.")
        
        # Position features (x, y)
        position_indices = [3, 4]  # x, y position
        position_names = ["ball_x_pos", "ball_y_pos"]
        
        for i, idx in enumerate(position_indices):
            feature_name = position_names[i]
            # Only evaluate on samples where redball exists
            redball_exists_mask = targets[:, 5] > 0.5  # Use ground truth for masking
            
            if np.sum(redball_exists_mask) > 0:  # Check if any redball exists
                pred_filtered = predictions[redball_exists_mask, idx]
                target_filtered = targets[redball_exists_mask, idx]
                
                diff = pred_filtered - target_filtered
                std_dev = np.std(diff)
                feature_metrics.append(("std_dev", std_dev))
                
                if debug_logger:
                    debug_logger.log(f"Batch evaluation - {feature_name} (balls only):")
                    debug_logger.log(f"  Count: {np.sum(redball_exists_mask)} samples with balls")
                    debug_logger.log(f"  StdDev of difference: {std_dev:.6f}")
                    debug_logger.log(f"  Mean absolute error: {np.mean(np.abs(diff)):.6f}")
                    debug_logger.log(f"  Max absolute error: {np.max(np.abs(diff)):.6f}")
                    debug_logger.log_tensor_stats(f"{feature_name} errors", diff)
            else:
                # If no redball exists in this batch
                feature_metrics.append(("std_dev", float('nan')))
                
                if debug_logger:
                    debug_logger.log(f"Batch evaluation - {feature_name}: No balls in batch")
    else:
        # For other environments
        n_features = predictions.shape[1]
        for i in range(n_features):
            diff = predictions[:, i] - targets[:, i]
            std_dev = np.std(diff)
            feature_metrics.append(("std_dev", std_dev))

    return feature_metrics


def evaluate_model(model, data_loader, device, env_name, debug_logger=None):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    if debug_logger:
        debug_logger.log("Starting model evaluation")

    with torch.no_grad():
        batch_count = 0
        for images, labels_true in data_loader:
            batch_count += 1
            images, labels_true = images.to(device), labels_true.to(device)
            
            # Run model with debugging on first batch
            if batch_count == 1 and debug_logger:
                labels_pred = model.debug_forward(images, debug_logger)
            else:
                labels_pred = model(images)
                
            loss = compute_loss(labels_pred, labels_true, env_name, debug_logger if batch_count == 1 else None)
            total_loss += loss.item()

            all_predictions.append(labels_pred.cpu().numpy())
            all_targets.append(labels_true.cpu().numpy())
            
            # Detailed debug for first batch
            if batch_count == 1 and debug_logger:
                debug_logger.log(f"First evaluation batch statistics:")
                batch_metrics = evaluate_batch(labels_pred, labels_true, env_name, debug_logger)
                debug_logger.log(f"First batch metrics: {batch_metrics}")

    avg_loss = total_loss / len(data_loader)
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    if debug_logger:
        debug_logger.log(f"Evaluation complete. Average loss: {avg_loss:.6f}")
        debug_logger.log(f"Collected {len(predictions)} predictions")
        
        # Log overall prediction and target distributions
        for i in range(predictions.shape[1]):
            feature_name = ["left_blocked", "front_blocked", "right_blocked", 
                           "ball_x_pos", "ball_y_pos", "ball_exists"][i]
            
            # For binary features
            if i in [0, 1, 2, 5]:
                pred_binary = (predictions[:, i] > 0.5).astype(int)
                target_binary = (targets[:, i] > 0.5).astype(int)
                
                debug_logger.log(f"Overall distribution for {feature_name}:")
                debug_logger.log(f"  Target: {np.mean(target_binary):.2%} positive, {1-np.mean(target_binary):.2%} negative")
                debug_logger.log(f"  Prediction: {np.mean(pred_binary):.2%} positive, {1-np.mean(pred_binary):.2%} negative")
                
                # Confusion matrix
                tn, fp, fn, tp = confusion_matrix(target_binary, pred_binary, labels=[0, 1]).ravel()
                total = len(target_binary)
                debug_logger.log(f"  Confusion matrix:")
                debug_logger.log(f"    TP: {tp} ({tp/total:.2%}), FP: {fp} ({fp/total:.2%})")
                debug_logger.log(f"    TN: {tn} ({tn/total:.2%}), FN: {fn} ({fn/total:.2%})")
                
                # Raw prediction distribution
                raw_preds = predictions[:, i]
                bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                hist, _ = np.histogram(raw_preds, bins=bins)
                debug_logger.log(f"  Raw prediction histogram:")
                for j in range(len(bins)-1):
                    debug_logger.log(f"    {bins[j]:.1f}-{bins[j+1]:.1f}: {hist[j]} samples ({hist[j]/len(raw_preds):.2%})")

    # Calculate per-feature metrics
    feature_metrics = []
    n_features = predictions.shape[1]

    if env_name == "redball":
        # Binary features (obstacles + redball_exists) - Use accuracy
        binary_indices = [0, 1, 2, 5]  # left, front, right blocked + redball_exists
        binary_names = ["left_blocked", "front_blocked", "right_blocked", "ball_exists"]
        
        for i, idx in enumerate(binary_indices):
            feature_name = binary_names[i]
            pred_binary = (predictions[:, idx] > 0.5).astype(int)
            target_binary = (targets[:, idx] > 0.5).astype(int)
            accuracy = (pred_binary == target_binary).mean()
            feature_metrics.append(("accuracy", accuracy))
            
            if debug_logger:
                # More detailed metrics
                report = classification_report(target_binary, pred_binary, output_dict=True)
                debug_logger.log(f"Detailed metrics for {feature_name}:")
                debug_logger.log(f"  Accuracy: {accuracy:.4f}")
                debug_logger.log(f"  Precision (class 1): {report['1']['precision']:.4f}")
                debug_logger.log(f"  Recall (class 1): {report['1']['recall']:.4f}")
                debug_logger.log(f"  F1-score (class 1): {report['1']['f1-score']:.4f}")
        
        # Position features (x, y) - Use standard deviation
        position_indices = [3, 4]  # x, y position
        position_names = ["ball_x_pos", "ball_y_pos"]
        
        for i, idx in enumerate(position_indices):
            feature_name = position_names[i]
            
            # Only evaluate on samples where redball exists (according to ground truth)
            redball_exists_mask = targets[:, 5] > 0.5
            
            if np.sum(redball_exists_mask) > 0:  # Check if any redball exists
                pred_filtered = predictions[redball_exists_mask, idx]
                target_filtered = targets[redball_exists_mask, idx]
                
                diff = pred_filtered - target_filtered
                std_dev = np.std(diff)
                feature_metrics.append(("std_dev", std_dev))
                
                if debug_logger:
                    debug_logger.log(f"Position metric for {feature_name}:")
                    debug_logger.log(f"  StdDev: {std_dev:.6f}")
                    debug_logger.log(f"  Mean Absolute Error: {np.mean(np.abs(diff)):.6f}")
                    debug_logger.log(f"  Max Error: {np.max(np.abs(diff)):.6f}")
            else:
                # If no redball exists in this batch
                feature_metrics.append(("std_dev", float('nan')))
                
                if debug_logger:
                    debug_logger.log(f"No balls found for {feature_name} evaluation")
    else:
        # For other environments, all features are continuous (unchanged)
        for i in range(n_features):
            diff = predictions[:, i] - targets[:, i]
            std_dev = np.std(diff)
            feature_metrics.append(("std_dev", std_dev))

    # Calculate MSE per feature
    mse_per_feature = np.zeros(n_features)
    
    if env_name == "redball":
        # Binary features - regular MSE
        for i in [0, 1, 2, 5]:  # Binary features
            mse_per_feature[i] = ((predictions[:, i] - targets[:, i]) ** 2).mean()
        
        # Position features - only calculate MSE when redball exists
        redball_exists_mask = targets[:, 5] > 0.5
        if np.sum(redball_exists_mask) > 0:
            for i in [3, 4]:  # Position features
                pred_filtered = predictions[redball_exists_mask, i]
                target_filtered = targets[redball_exists_mask, i]
                mse_per_feature[i] = ((pred_filtered - target_filtered) ** 2).mean()
        else:
            for i in [3, 4]:
                mse_per_feature[i] = float('nan')
    else:
        # For other environments (unchanged)
        mse_per_feature = ((predictions - targets) ** 2).mean(axis=0)

    # Calculate R^2 score per feature
    r2_scores = np.zeros(n_features)
    
    if env_name == "redball":
        # Binary features
        for i in [0, 1, 2, 5]:
            r2_scores[i] = r2_score(targets[:, i], predictions[:, i])
            
        # Position features - only when redball exists
        redball_exists_mask = targets[:, 5] > 0.5
        if np.sum(redball_exists_mask) > 0:
            for i in [3, 4]:
                pred_filtered = predictions[redball_exists_mask, i]
                target_filtered = targets[redball_exists_mask, i]
                r2_scores[i] = r2_score(target_filtered, pred_filtered)
        else:
            for i in [3, 4]:
                r2_scores[i] = float('nan')
    else:
        # For other environments (unchanged)
        for i in range(n_features):
            r2_scores[i] = r2_score(targets[:, i], predictions[:, i])

    return avg_loss, mse_per_feature, feature_metrics, r2_scores


def save_metrics(metrics, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)


def analyze_dataset_distribution(train_dataset, val_dataset, test_dataset, debug_logger):
    """Analyze dataset distributions to check for imbalances"""
    debug_logger.log("Analyzing dataset distributions...")
    
    dataset_names = ["Train", "Validation", "Test"]
    datasets = [train_dataset, val_dataset, test_dataset]
    
    # Feature names for better logging
    feature_names = ["left_blocked", "front_blocked", "right_blocked", 
                    "ball_x_pos", "ball_y_pos", "ball_exists"]
    
    for name, dataset in zip(dataset_names, datasets):
        debug_logger.log(f"\n{name} dataset analysis:")
        
        # Get all labels
        all_labels = []
        for _, label in dataset:
            all_labels.append(label.numpy())
        all_labels = np.array(all_labels)
        
        debug_logger.log(f"  Dataset size: {len(all_labels)} samples")
        
        # Analyze each feature
        for i, feature_name in enumerate(feature_names):
            feature_data = all_labels[:, i]
            
            # Different analysis for binary vs continuous features
            if i in [0, 1, 2, 5]:  # Binary features
                positive_count = np.sum(feature_data > 0.5)
                negative_count = len(feature_data) - positive_count
                pos_ratio = positive_count / len(feature_data)
                neg_ratio = negative_count / len(feature_data)
                
                debug_logger.log(f"  {feature_name}:")
                debug_logger.log(f"    Positive: {positive_count} ({pos_ratio:.2%})")
                debug_logger.log(f"    Negative: {negative_count} ({neg_ratio:.2%})")
                
                # Check for severe imbalance
                if pos_ratio < 0.05 or pos_ratio > 0.95:
                    debug_logger.log(f"    WARNING: Severe class imbalance detected for {feature_name}!")
                    
                # Store for later comparison
                if name == "Train":
                    train_ratios = {"positive": pos_ratio, "negative": neg_ratio}
                elif name == "Validation":
                    val_ratios = {"positive": pos_ratio, "negative": neg_ratio}
                
            else:  # Continuous features (position)
                # Only analyze when ball exists
                ball_exists_mask = all_labels[:, 5] > 0.5
                if np.sum(ball_exists_mask) > 0:
                    filtered_data = feature_data[ball_exists_mask]
                    stats = {
                        "mean": np.mean(filtered_data),
                        "std": np.std(filtered_data),
                        "min": np.min(filtered_data),
                        "max": np.max(filtered_data),
                    }
                    
                    debug_logger.log(f"  {feature_name} (when ball exists):")
                    debug_logger.log(f"    Mean: {stats['mean']:.4f}")
                    debug_logger.log(f"    Std: {stats['std']:.4f}")
                    debug_logger.log(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                else:
                    debug_logger.log(f"  {feature_name}: No balls in dataset")
    
    # Compare train/val distributions
    debug_logger.log("\nComparing Train vs Validation distributions:")
    for i, feature_name in enumerate(feature_names):
        if i in [0, 1, 2, 5]:  # Binary features
            train_pos = train_ratios["positive"]
            val_pos = val_ratios["positive"]
            diff = abs(train_pos - val_pos)
            
            debug_logger.log(f"  {feature_name}:")
            debug_logger.log(f"    Train positive: {train_pos:.2%}")
            debug_logger.log(f"    Val positive: {val_pos:.2%}")
            debug_logger.log(f"    Difference: {diff:.2%}")
            
            if diff > 0.05:
                debug_logger.log(f"    WARNING: Distribution shift between train and validation!")


def main():
    # Configuration
    env_n = "redball"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 300
    batch_size = 128
    learning_rate = 0.0001
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join('logs', env_n, "two_stage_debug_" + current_time)
    os.makedirs(log_path, exist_ok=True)
    
    # Initialize debug logger
    debug_logger = DebugLogger(log_path)
    debug_logger.log(f"Starting training with configuration:")
    debug_logger.log(f"  Environment: {env_n}")
    debug_logger.log(f"  Device: {device}")
    debug_logger.log(f"  Epochs: {num_epochs}")
    debug_logger.log(f"  Batch size: {batch_size}")
    debug_logger.log(f"  Learning rate: {learning_rate}")
    debug_logger.log(f"  Log path: {log_path}")

    # Dataset initialization
    dataset_dir = f'redball_data'
    debug_logger.log("Initializing datasets...")
    
    train_valid_dataset = CustomImageDataset(
        dataset_dir, env_n, size=10000,
        csv_file='redball_data/train.csv',
        debug_logger=debug_logger
    )
    test_dataset = CustomImageDataset(
        dataset_dir, env_n, size=2000,
        csv_file='redball_data/test.csv',
        debug_logger=debug_logger
    )

    train_size = int(0.8 * len(train_valid_dataset))
    train_dataset, val_dataset = random_split(
        train_valid_dataset, [train_size, len(train_valid_dataset) - train_size]
    )

    print("Size for each set: ", len(train_dataset), len(val_dataset), len(test_dataset))
    debug_logger.log(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Analyze dataset distributions
    analyze_dataset_distribution(train_dataset, val_dataset, test_dataset, debug_logger)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get input image shape
    first_image, _ = train_dataset[0]
    image_shape = first_image.shape
    num_channels = image_shape[0] if len(image_shape) == 3 else 1
    image_height = image_shape[1] if len(image_shape) == 3 else image_shape[0]
    image_width = image_shape[2] if len(image_shape) == 3 else image_shape[1]

    debug_logger.log(f"Input image shape: channels={num_channels}, height={image_height}, width={image_width}")

    # Initialize model, optimizer, and scheduler
    model = TwoStageDetectCNN(num_channels, image_height, image_width).to(device)
    
    # Print model architecture for debugging
    debug_logger.log("Model architecture:")
    debug_logger.log(str(model))
    
    # Count and log the number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    debug_logger.log(f"Total parameters: {total_params:,}")
    debug_logger.log(f"Trainable parameters: {trainable_params:,}")
    
    # Analyze parameter initialization
    debug_logger.log("\nParameter initialization analysis:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            debug_logger.log(f"  {name}: Shape={param.shape}, Min={param.min().item():.6f}, Max={param.max().item():.6f}")
    
    # Test forward pass to check model output dimensions
    with torch.no_grad():
        debug_logger.log("\nTesting forward pass with dummy input...")
        dummy_input = torch.zeros(1, num_channels, image_height, image_width, device=device)
        dummy_output = model(dummy_input)
        debug_logger.log(f"Model output shape: {dummy_output.shape}")
        debug_logger.log_tensor_stats("Dummy output", dummy_output)
       
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    # Training setup
    best_val_loss = float('inf')
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'epoch_metrics': []
    }

    writer = SummaryWriter(log_dir=log_path)
    model_save_path = os.path.join(log_path, 'best_detection.pth')
    learning_curve_save_path = os.path.join(log_path, 'learning_curve.png')
    metrics_save_path = os.path.join(log_path, 'training_metrics.json')

    print(f"Training on {device}")
    debug_logger.log(f"Training on {device}")
    print(f"Saving logs to {log_path}")

    # Define feature names for clearer output
    feature_names = [
        "left_blocked", 
        "front_blocked", 
        "right_blocked", 
        "ball_x_pos", 
        "ball_y_pos", 
        "ball_exists"
    ]

    # Training loop
    for epoch in range(num_epochs):
        debug_logger.log(f"\n{'='*20} EPOCH {epoch + 1}/{num_epochs} {'='*20}")
        model.train()
        epoch_train_loss = 0.0
        batch_metrics_history = []

        # Training phase with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for batch_idx, (images, labels_true) in enumerate(train_pbar):
            images, labels_true = images.to(device), labels_true.to(device)
            
            # First batch gets extra debugging
            if batch_idx == 0 and epoch % 10 == 0:
                debug_logger.log(f"Detailed first batch analysis (epoch {epoch+1}):")
                labels_pred = model.debug_forward(images, debug_logger)
                
                # Check for gradient vanishing/exploding
                optimizer.zero_grad()
                loss = compute_loss(labels_pred, labels_true, env_n, debug_logger)
                loss.backward()
                
                # Log gradients for all parameters
                debug_logger.log("Gradient analysis:")
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        param_norm = param.norm().item()
                        debug_logger.log(f"  {name}: grad_norm={grad_norm:.6e}, param_norm={param_norm:.6e}, ratio={grad_norm/param_norm if param_norm > 0 else 'N/A'}")
                        
                        # Check for vanishing/exploding gradients
                        if grad_norm < 1e-7:
                            debug_logger.log(f"  WARNING: Possible vanishing gradient for {name}")
                        if grad_norm > 1e3:
                            debug_logger.log(f"  WARNING: Possible exploding gradient for {name}")
                
                # Apply gradient clipping and optimizer step
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                # Normal forward pass for other batches
                labels_pred = model(images)
                loss = compute_loss(labels_pred, labels_true, env_n)
                optimizer.zero_grad()
                loss.backward()
                
                # Debug gradients for detection head occasionally to track convergence
                if batch_idx % 100 == 0:
                    for name, param in model.named_parameters():
                        if "detection_head" in name and param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            debug_logger.log(f"Batch {batch_idx}, {name} grad norm: {grad_norm:.4e}")
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_train_loss += loss.item()

            # Calculate and display batch metrics
            batch_metrics = evaluate_batch(labels_pred, labels_true, env_n)
            batch_metrics_history.append(batch_metrics)
            metrics_str = format_metrics_string(batch_metrics)
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'metrics': metrics_str
            })
            
            # Detailed logging for problematic batches
            binary_accuracies = [metric[1] for i, metric in enumerate(batch_metrics) if i < 4]
            if any(acc < 0.6 for acc in binary_accuracies) and batch_idx % 20 == 0:
                debug_logger.log(f"Warning: Low binary accuracy in batch {batch_idx}:")
                for i, (metric_type, value) in enumerate(batch_metrics[:4]):  # First 4 are binary metrics
                    debug_logger.log(f"  {feature_names[i]}: {value:.4f}")
                    
                # Debug the predictions and targets
                binary_pred = (labels_pred[:, :3].detach().cpu().numpy() > 0.5).astype(int)
                binary_true = (labels_true[:, :3].detach().cpu().numpy() > 0.5).astype(int)
                
                for i in range(3):  # 3 block features
                    feature = feature_names[i]
                    correct = (binary_pred[:, i] == binary_true[:, i]).mean()
                    pos_true = binary_true[:, i].mean()
                    pos_pred = binary_pred[:, i].mean()
                    debug_logger.log(f"  {feature}: accuracy={correct:.4f}, true_pos={pos_true:.4f}, pred_pos={pos_pred:.4f}")
                    
                    # Additional detailed analysis
                    raw_preds = labels_pred[:, i].detach().cpu().numpy()
                    debug_logger.log_tensor_stats(f"{feature} raw predictions", raw_preds)

        avg_train_loss = epoch_train_loss / len(train_loader)
        metrics['train_losses'].append(avg_train_loss)
        
        # Calculate average batch metrics
        avg_batch_metrics = []
        for i in range(len(batch_metrics_history[0])):
            metric_type = batch_metrics_history[0][i][0]
            avg_value = np.mean([batch[i][1] for batch in batch_metrics_history])
            avg_batch_metrics.append((metric_type, avg_value))
        
        debug_logger.log(f"Average training metrics for epoch {epoch+1}:")
        debug_logger.log(format_metrics_string(avg_batch_metrics))

        # Validation phase
        debug_logger.log(f"Running validation for epoch {epoch+1}...")
        val_loss, val_mse_per_feature, val_feature_metrics, val_r2_scores = evaluate_model(
            model, val_loader, device, env_n, debug_logger if epoch % 10 == 0 else None
        )
        metrics['val_losses'].append(val_loss)
        
        # Detailed analysis of validation metrics
        debug_logger.log(f"Validation metrics for epoch {epoch+1}:")
        for i, (metric_type, value) in enumerate(val_feature_metrics):
            feature_name = feature_names[i]
            debug_logger.log(f"  {feature_name}: {metric_type}={value:.6f}")
            
        # Check for stagnant block statuses
        if epoch > 0 and epoch % 5 == 0:
            debug_logger.log("Checking for stagnant metrics...")
            prev_metrics = metrics['epoch_metrics'][-5:]
            
            for i in range(3):  # First 3 are block metrics
                feature_name = feature_names[i]
                recent_vals = [em['val_feature_metrics'][i][1] for em in prev_metrics]
                variance = np.var(recent_vals) if len(recent_vals) > 1 else 0
                
                debug_logger.log(f"  {feature_name} last 5 values: {recent_vals}")
                debug_logger.log(f"  {feature_name} variance: {variance:.8f}")
                
                if variance < 1e-6 and epoch > 10:
                    debug_logger.log(f"  WARNING: {feature_name} appears STAGNANT - not improving!")
                    
                    # Check if stagnant at a meaningful level
                    avg_value = np.mean(recent_vals)
                    if 0.45 < avg_value < 0.55:
                        debug_logger.log(f"  CRITICAL: {feature_name} stuck near 0.5 - model may be predicting randomly!")
                    elif avg_value > 0.95:
                        debug_logger.log(f"  WARNING: {feature_name} stuck near 1.0 - check for data imbalance!")
                    elif avg_value < 0.05:
                        debug_logger.log(f"  WARNING: {feature_name} stuck near 0.0 - check for data imbalance!")

        # Save current epoch metrics
        epoch_metric = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'learning_rate': scheduler.get_last_lr()[0],
            'val_mse_per_feature': val_mse_per_feature.tolist(),
            'val_feature_metrics': [(metric_type, float(value)) for metric_type, value in val_feature_metrics],
            'val_r2_scores': val_r2_scores.tolist()
        }
        metrics['epoch_metrics'].append(epoch_metric)

        # Save metrics periodically
        if (epoch + 1) % 5 == 0:
            save_metrics(metrics, metrics_save_path)

        # TensorBoard logging
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        
        # Log feature-specific metrics
        for i, (metric_type, value) in enumerate(val_feature_metrics):
            feature_name = feature_names[i]
            if metric_type == "accuracy":
                writer.add_scalar(f"Accuracy/{feature_name}", value, epoch)
            else:
                writer.add_scalar(f"StdDev/{feature_name}", value, epoch)

        # Learning rate scheduling
        scheduler.step()

        # Print epoch summary with clear feature names
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print("Validation Metrics:", format_metrics_string(val_feature_metrics))
        
        # Also print individual metrics with named features for better readability
        print("\nDetailed Validation Metrics:")
        for i, (metric_type, value) in enumerate(val_feature_metrics):
            feature_name = feature_names[i]
            if metric_type == "accuracy":
                print(f"  {feature_name}: Accuracy = {value:.4f}")
            else:
                print(f"  {feature_name}: StdDev = {value:.4f}")
                
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"======= New best model saved with validation loss: {best_val_loss:.4f} =======")
            debug_logger.log(f"New best model saved with validation loss: {best_val_loss:.4f}")
            
            # If we have a new best model, also evaluate on test set and print metrics
            # This helps track progress during training
            if (epoch + 1) % 10 == 0:  # Only do this every 10 epochs to save time
                print("\nIntermediate Test Set Evaluation:")
                debug_logger.log("\nIntermediate Test Set Evaluation:")
                test_loss, test_mse, test_metrics, test_r2 = evaluate_model(model, test_loader, device, env_n, debug_logger)
                print(f"  Test Loss: {test_loss:.4f}")
                debug_logger.log(f"  Test Loss: {test_loss:.4f}")
                for i, (metric_type, value) in enumerate(test_metrics):
                    feature_name = feature_names[i]
                    if metric_type == "accuracy":
                        print(f"  {feature_name}: Accuracy = {value:.4f}")
                        debug_logger.log(f"  {feature_name}: Accuracy = {value:.4f}")
                    else:
                        print(f"  {feature_name}: StdDev = {value:.4f}")
                        debug_logger.log(f"  {feature_name}: StdDev = {value:.4f}")

        # Check for potential issues every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            debug_logger.log("\nPerforming model checkup...")
            
            # Check for overfitting
            recent_train_losses = metrics['train_losses'][-10:]
            recent_val_losses = metrics['val_losses'][-10:]
            
            train_trend = recent_train_losses[-1] - recent_train_losses[0]
            val_trend = recent_val_losses[-1] - recent_val_losses[0]
            
            debug_logger.log(f"Last 10 epochs - Train loss change: {train_trend:.6f}, Val loss change: {val_trend:.6f}")
            
            if train_trend < -0.01 and val_trend > 0.005:
                debug_logger.log("WARNING: Potential overfitting detected - train loss decreasing while val loss increasing")
                
            # Check for learning plateaus
            if abs(train_trend) < 0.001 and epoch > 20:
                debug_logger.log("WARNING: Training loss plateau detected - consider adjusting learning rate or model architecture")
            
            # Analyze detection head weights to check for potential issues
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "detection_head" in name and "weight" in name:
                        weights = param.detach().cpu().numpy()
                        debug_logger.log(f"Detection head weight analysis for {name}:")
                        debug_logger.log(f"  Shape: {weights.shape}")
                        debug_logger.log(f"  Mean: {np.mean(weights):.6f}")
                        debug_logger.log(f"  Std: {np.std(weights):.6f}")
                        debug_logger.log(f"  Min: {np.min(weights):.6f}")
                        debug_logger.log(f"  Max: {np.max(weights):.6f}")
                        
                        # Check for potential weight saturation
                        if np.max(np.abs(weights)) > 10:
                            debug_logger.log("  WARNING: Large weights detected - potential saturation!")
                            
                        # Check for potential dead neurons
                        row_norms = np.linalg.norm(weights, axis=1)
                        if np.any(row_norms < 0.01):
                            dead_count = np.sum(row_norms < 0.01)
                            debug_logger.log(f"  WARNING: {dead_count} potential dead neurons detected!")

    # Final evaluation on test set
    print("\nFinal Evaluation on Test Set:")
    debug_logger.log("\n" + "="*50)
    debug_logger.log("FINAL EVALUATION ON TEST SET:")
    
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_mse_per_feature, feature_metrics, r2_scores = evaluate_model(model, test_loader, device, env_n, debug_logger)

    print(f'Test Loss: {test_loss:.4f}')
    debug_logger.log(f'Test Loss: {test_loss:.4f}')
    
    print("\nDetailed Feature Metrics:")    
    debug_logger.log("\nDetailed Feature Metrics:")    
    for i, (metric_type, value) in enumerate(feature_metrics):
        feature_name = feature_names[i]
        if metric_type == "accuracy":
            print(f"{feature_name} - Accuracy: {value:.4f}")
            debug_logger.log(f"{feature_name} - Accuracy: {value:.4f}")
        else:
            print(f"{feature_name} - Std Dev of Difference: {value:.4f}")
            debug_logger.log(f"{feature_name} - Std Dev of Difference: {value:.4f}")

    print("\nMSE and R^2 per feature:")
    debug_logger.log("\nMSE and R^2 per feature:")
    for i, (mse, r2) in enumerate(zip(test_mse_per_feature, r2_scores)):
        feature_name = feature_names[i]
        print(f"{feature_name}: MSE = {mse:.6f}, R^2 = {r2:.6f}")
        debug_logger.log(f"{feature_name}: MSE = {mse:.6f}, R^2 = {r2:.6f}")

    # Separate evaluation for position features only when redball exists
    print("\nEvaluating red ball position prediction only for cases where the ball exists:")
    debug_logger.log("\nEvaluating red ball position prediction only for cases where the ball exists:")
    with torch.no_grad():
        all_predictions = []
        all_targets = []
        
        for images, labels_true in test_loader:
            images, labels_true = images.to(device), labels_true.to(device)
            labels_pred = model(images)
            
            # Convert to numpy for filtering
            predictions = labels_pred.cpu().numpy()
            targets = labels_true.cpu().numpy()
            
            # Only keep samples where redball exists (based on ground truth)
            redball_exists_mask = targets[:, 5] > 0.5
            if np.sum(redball_exists_mask) > 0:
                all_predictions.append(predictions[redball_exists_mask])
                all_targets.append(targets[redball_exists_mask])
    
    if all_predictions:
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # Evaluate position accuracy
        x_diff = all_predictions[:, 3] - all_targets[:, 3]
        y_diff = all_predictions[:, 4] - all_targets[:, 4]
        
        x_mse = np.mean(x_diff ** 2)
        y_mse = np.mean(y_diff ** 2)
        
        x_std = np.std(x_diff)
        y_std = np.std(y_diff)
        
        x_r2 = r2_score(all_targets[:, 3], all_predictions[:, 3])
        y_r2 = r2_score(all_targets[:, 4], all_predictions[:, 4])
        
        print(f"Red ball X position (when ball exists): MSE = {x_mse:.6f}, StdDev = {x_std:.6f}, R^2 = {x_r2:.6f}")
        print(f"Red ball Y position (when ball exists): MSE = {y_mse:.6f}, StdDev = {y_std:.6f}, R^2 = {y_r2:.6f}")
        debug_logger.log(f"Red ball X position (when ball exists): MSE = {x_mse:.6f}, StdDev = {x_std:.6f}, R^2 = {x_r2:.6f}")
        debug_logger.log(f"Red ball Y position (when ball exists): MSE = {y_mse:.6f}, StdDev = {y_std:.6f}, R^2 = {y_r2:.6f}")
        
        # Calculate distance error (Euclidean distance between predicted and actual positions)
        # This gives a more intuitive measure of position accuracy
        euclidean_distances = np.sqrt(x_diff**2 + y_diff**2)
        mean_distance = np.mean(euclidean_distances)
        median_distance = np.median(euclidean_distances)
        std_distance = np.std(euclidean_distances)
        
        print(f"Position Error (Euclidean distance): Mean = {mean_distance:.6f}, Median = {median_distance:.6f}, StdDev = {std_distance:.6f}")
        debug_logger.log(f"Position Error (Euclidean distance): Mean = {mean_distance:.6f}, Median = {median_distance:.6f}, StdDev = {std_distance:.6f}")
    else:
        print("No samples with red ball found in test set.")
        debug_logger.log("No samples with red ball found in test set.")

    # Debug: Analyze the final model's detection head to see why block statuses might be stagnant
    debug_logger.log("\nAnalyzing final model weights for block status detection:")
    with torch.no_grad():
        # Examine final layer weights for each binary feature
        final_detection_layer = model.detection_head[-1]
        weights = final_detection_layer.weight.detach().cpu().numpy()
        bias = final_detection_layer.bias.detach().cpu().numpy()
        
        binary_feature_names = ["left_blocked", "front_blocked", "right_blocked", "ball_exists"]
        for i, feature_name in enumerate(binary_feature_names):
            debug_logger.log(f"Analysis of {feature_name} detection:")
            feature_weights = weights[i]
            feature_bias = bias[i]
            
            debug_logger.log(f"  Bias: {feature_bias:.6f}")
            debug_logger.log(f"  Weight stats: Mean={np.mean(feature_weights):.6f}, Std={np.std(feature_weights):.6f}")
            debug_logger.log(f"  Weight range: [{np.min(feature_weights):.6f}, {np.max(feature_weights):.6f}]")
            
            # Check for potential issues
            if np.std(feature_weights) < 0.01:
                debug_logger.log(f"  WARNING: Very low weight variance for {feature_name}, may indicate failure to learn meaningful features!")
            
            if np.abs(feature_bias) > 5:
                debug_logger.log(f"  WARNING: Large bias for {feature_name}, may cause strong default prediction regardless of input!")
    
        # Run a forward pass on a sample batch and analyze intermediate activations
        debug_logger.log("\nAnalyzing intermediate activations on a sample batch:")
        sample_images, sample_labels = next(iter(test_loader))
        sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
        
        # Get feature representations before the final layer
        with torch.no_grad():
            # Extract features
            features = model.feature_extractor(sample_images)
            # Get pre-pooling activations
            debug_logger.log_tensor_stats("Feature extractor output", features)
            
            # Get pooled feature activations
            pooled = nn.AdaptiveAvgPool2d((1, 1))(features)
            flattened = pooled.view(pooled.size(0), -1)
            debug_logger.log_tensor_stats("Pooled features", flattened)
            
            # Get activations before the final detection layer
            hidden = model.detection_head[:-2](flattened)  # Assuming the last two layers are Dropout and Linear
            debug_logger.log_tensor_stats("Hidden layer activations", hidden)
            
            # Examine raw logits (pre-sigmoid)
            detection_output = model.detection_head(flattened)
            debug_logger.log_tensor_stats("Raw logits", detection_output)
            
            # Check if logits are strongly polarized
            for i, feature_name in enumerate(binary_feature_names):
                logits = detection_output[:, i].cpu().numpy()
                debug_logger.log(f"  {feature_name} logits: Mean={np.mean(logits):.4f}, Std={np.std(logits):.4f}")
                debug_logger.log(f"  {feature_name} logits range: [{np.min(logits):.4f}, {np.max(logits):.4f}]")
                
                # Check for potential issues
                if np.std(logits) < 0.1:
                    debug_logger.log(f"  WARNING: Very low variance in {feature_name} logits - model may be stuck!")
                
                # Check if all predictions are heavily skewed toward 0 or 1
                sigmoid_preds = torch.sigmoid(detection_output[:, i]).cpu().numpy()
                near_0 = np.mean(sigmoid_preds < 0.1)
                near_1 = np.mean(sigmoid_preds > 0.9)
                near_middle = np.mean((sigmoid_preds > 0.4) & (sigmoid_preds < 0.6))
                
                debug_logger.log(f"  {feature_name} sigmoid: {near_0:.2%} near 0, {near_middle:.2%} near 0.5, {near_1:.2%} near 1")
                
                if near_0 > 0.9 or near_1 > 0.9:
                    debug_logger.log(f"  WARNING: {feature_name} predictions strongly biased to {'0' if near_0 > 0.9 else '1'}")
                
                if near_middle > 0.8:
                    debug_logger.log(f"  WARNING: {feature_name} predictions mostly near 0.5, indicating uncertainty!")

    # Plot and save learning curves
    plt.figure(figsize=(18, 8))
    fontsize = 18

    plt.subplot(121)
    plt.plot(metrics['train_losses'][1:], '-o', label="training_loss")
    plt.title("Training Loss", fontsize=fontsize)
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)

    plt.subplot(122)
    plt.plot(metrics['val_losses'][1:], '-o', label="validation_loss")
    plt.title("Validation Loss", fontsize=fontsize)
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)

    plt.tight_layout()
    plt.savefig(learning_curve_save_path)
    debug_logger.log(f"Saved learning curve to {learning_curve_save_path}")
    plt.close()

    # Plot and save feature-specific metrics
    plt.figure(figsize=(18, 12))
    
    # Extract metrics over time for binary features
    binary_names = ["Left Blocked", "Front Blocked", "Right Blocked", "Ball Exists"]
    accuracies_over_time = {name: [] for name in binary_names}
    
    for epoch_metric in metrics['epoch_metrics']:
        for i, (metric_type, value) in enumerate(epoch_metric['val_feature_metrics']):
            if i < 4:  # Binary features
                accuracies_over_time[binary_names[i]].append(value)
    
    # Plot accuracies over time
    plt.subplot(221)
    for name in binary_names:
        plt.plot(accuracies_over_time[name], '-o', label=name)
    plt.title("Binary Feature Accuracies Over Time", fontsize=fontsize-2)
    plt.xlabel("Epochs", fontsize=fontsize-2)
    plt.ylabel("Accuracy", fontsize=fontsize-2)
    plt.legend(fontsize=fontsize-4)
    plt.grid(True, alpha=0.3)
    
    # 1. Binary features accuracy (final)
    plt.subplot(222)
    binary_indices = [0, 1, 2, 5]  # left, front, right, exists
    binary_accuracies = [metric[1] for i, metric in enumerate(feature_metrics) if i < 4]
    
    plt.bar(binary_names, binary_accuracies)
    plt.title("Binary Features Accuracy", fontsize=fontsize)
    plt.ylabel("Accuracy", fontsize=fontsize)
    plt.ylim(0, 1)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    # 2. Position features std deviation
    plt.subplot(223)
    position_indices = [3, 4]  # x, y
    position_names = ["X Position", "Y Position"]
    position_std = [metric[1] for i, metric in enumerate(feature_metrics) if 3 <= i < 5]
    
    plt.bar(position_names, position_std)
    plt.title("Position Features StdDev (Ball Exists Only)", fontsize=fontsize)
    plt.ylabel("StdDev", fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    # 3. MSE per feature
    plt.subplot(224)
    all_names = ["Left Blocked", "Front Blocked", "Right Blocked", "X Position", "Y Position", "Ball Exists"]
    plt.bar(all_names, test_mse_per_feature)
    plt.title("MSE per Feature", fontsize=fontsize)
    plt.ylabel("MSE", fontsize=fontsize)
    plt.xticks(rotation=45, fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'feature_metrics.png'))
    debug_logger.log(f"Saved feature metrics to {os.path.join(log_path, 'feature_metrics.png')}")
    plt.close()

    # Create a confusion matrix for each binary feature
    plt.figure(figsize=(20, 5))
    binary_indices = [0, 1, 2, 5]
    binary_names = ["Left Blocked", "Front Blocked", "Right Blocked", "Ball Exists"]
    
    all_predictions_binary = []
    all_targets_binary = []
    
    with torch.no_grad():
        for images, labels_true in test_loader:
            images, labels_true = images.to(device), labels_true.to(device)
            labels_pred = model(images)
            
            # Convert to numpy
            predictions = labels_pred.cpu().numpy()
            targets = labels_true.cpu().numpy()
            
            all_predictions_binary.append(predictions)
            all_targets_binary.append(targets)
    
    all_predictions_binary = np.vstack(all_predictions_binary)
    all_targets_binary = np.vstack(all_targets_binary)
    
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    for i, (idx, name) in enumerate(zip(binary_indices, binary_names)):
        plt.subplot(1, 4, i+1)
        
        pred_binary = (all_predictions_binary[:, idx] > 0.5).astype(int)
        target_binary = (all_targets_binary[:, idx] > 0.5).astype(int)
        
        cm = confusion_matrix(target_binary, pred_binary, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
        plt.title(f"{name} Confusion Matrix", fontsize=fontsize-2)
        plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'binary_confusion_matrices.png'))
    debug_logger.log(f"Saved confusion matrices to {os.path.join(log_path, 'binary_confusion_matrices.png')}")
    plt.close()
    
    # Plot raw prediction distributions for each binary feature
    plt.figure(figsize=(20, 5))
    
    for i, (idx, name) in enumerate(zip(binary_indices, binary_names)):
        plt.subplot(1, 4, i+1)
        
        # Get raw sigmoid outputs
        raw_preds = all_predictions_binary[:, idx]
        
        # Plot histogram of raw predictions
        plt.hist(raw_preds, bins=20, alpha=0.7)
        plt.title(f"{name} Prediction Distribution", fontsize=fontsize-2)
        plt.xlabel("Sigmoid Output", fontsize=fontsize-4)
        plt.ylabel("Count", fontsize=fontsize-4)
        plt.grid(True, alpha=0.3)
        
        # Add markers for 0.5 threshold and mean
        plt.axvline(0.5, color='r', linestyle='--', label="Threshold")
        plt.axvline(np.mean(raw_preds), color='g', linestyle='-', label="Mean")
        
        # Highlight potential issues
        if 0.4 < np.mean(raw_preds) < 0.6:
            plt.text(0.5, plt.ylim()[1]*0.9, "Warning: Mean near 0.5", 
                     color='red', fontsize=fontsize-4, ha='center')
            
        plt.legend(fontsize=fontsize-4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'binary_prediction_distributions.png'))
    debug_logger.log(f"Saved prediction distributions to {os.path.join(log_path, 'binary_prediction_distributions.png')}")
    plt.close()

    # Plot position error distributions
    if all_predictions is not None and len(all_predictions) > 0:
        plt.figure(figsize=(18, 6))
        
        plt.subplot(131)
        plt.hist(x_diff, bins=30, alpha=0.7)
        plt.title("X Position Error Distribution", fontsize=fontsize)
        plt.xlabel("Error", fontsize=fontsize)
        plt.ylabel("Frequency", fontsize=fontsize)
        
        plt.subplot(132)
        plt.hist(y_diff, bins=30, alpha=0.7)
        plt.title("Y Position Error Distribution", fontsize=fontsize)
        plt.xlabel("Error", fontsize=fontsize)
        
        plt.subplot(133)
        plt.hist(euclidean_distances, bins=30, alpha=0.7)
        plt.title("Euclidean Distance Error", fontsize=fontsize)
        plt.xlabel("Distance Error", fontsize=fontsize)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_path, 'position_error_distribution.png'))
        debug_logger.log(f"Saved position error distributions to {os.path.join(log_path, 'position_error_distribution.png')}")
        plt.close()
        
        # Scatter plot of predicted vs actual positions
        plt.figure(figsize=(12, 10))
        plt.scatter(all_targets[:, 3], all_targets[:, 4], alpha=0.5, label="Actual Positions")
        plt.scatter(all_predictions[:, 3], all_predictions[:, 4], alpha=0.5, label="Predicted Positions")
        plt.title("Actual vs Predicted Ball Positions", fontsize=fontsize)
        plt.xlabel("X Position", fontsize=fontsize)
        plt.ylabel("Y Position", fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(log_path, 'position_scatter.png'))
        debug_logger.log(f"Saved position scatter plot to {os.path.join(log_path, 'position_scatter.png')}")
        plt.close()

    # Create correlation matrix for features to look for dependencies
    plt.figure(figsize=(10, 8))
    feature_names_short = ["left", "front", "right", "x_pos", "y_pos", "exists"]
    
    # Compute correlation between targets and predictions
    target_pred_corr = np.zeros((len(feature_names_short), len(feature_names_short)))
    
    for i in range(len(feature_names_short)):
        for j in range(len(feature_names_short)):
            # Calculate correlation between target_i and pred_j
            corr = np.corrcoef(all_targets_binary[:, i], all_predictions_binary[:, j])[0, 1]
            target_pred_corr[i, j] = corr
    
    # Plot correlation matrix
    plt.imshow(target_pred_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    
    # Add labels
    plt.xticks(np.arange(len(feature_names_short)), [f"pred_{name}" for name in feature_names_short], rotation=45)
    plt.yticks(np.arange(len(feature_names_short)), [f"true_{name}" for name in feature_names_short])
    
    # Add correlation values
    for i in range(len(feature_names_short)):
        for j in range(len(feature_names_short)):
            plt.text(j, i, f"{target_pred_corr[i, j]:.2f}", ha="center", va="center", 
                     color="white" if abs(target_pred_corr[i, j]) > 0.5 else "black")
    
    plt.title("Correlation Matrix: Ground Truth vs Predictions", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'correlation_matrix.png'))
    debug_logger.log(f"Saved correlation matrix to {os.path.join(log_path, 'correlation_matrix.png')}")
    plt.close()

    # Save final metrics
    save_metrics(metrics, metrics_save_path)
    debug_logger.log(f"Saved final metrics to {metrics_save_path}")
    writer.close()
    
    # Final recommendation based on debugging
    debug_logger.log("\n" + "="*50)
    debug_logger.log("DEBUGGING SUMMARY AND RECOMMENDATIONS:")
    
    # Analyze the block accuracy patterns
    block_accuracies = [metric[1] for i, metric in enumerate(feature_metrics) if i < 3]
    avg_block_accuracy = np.mean(block_accuracies)
    
    debug_logger.log(f"Average block detection accuracy: {avg_block_accuracy:.4f}")
    
    # Make recommendations
    if avg_block_accuracy < 0.7:
        debug_logger.log("ISSUE: Poor block detection accuracy.")
        
        # Check for class imbalance problems
        debug_logger.log("Possible causes and solutions:")
        debug_logger.log("1. Class imbalance - Analyze training data distributions and consider:")
        debug_logger.log("   - Weighted loss functions for imbalanced classes")
        debug_logger.log("   - Data augmentation to balance classes")
        debug_logger.log("   - Adjusting thresholds based on ROC curve analysis")
        
        # Check for feature extraction problems
        debug_logger.log("2. Inadequate feature extraction - Consider:")
        debug_logger.log("   - Adding more convolutional layers or filters")
        debug_logger.log("   - Using pre-trained models and transfer learning")
        debug_logger.log("   - Adding skip connections or residual blocks")
        
        # Check for optimization problems
        debug_logger.log("3. Optimization issues - Try:")
        debug_logger.log("   - Adjusting learning rate or using learning rate schedulers")
        debug_logger.log("   - Different optimizers (e.g., Adam with different parameters)")
        debug_logger.log("   - Batch normalization or different initialization")
        
        # Check for architecture problems
        debug_logger.log("4. Model architecture limitations - Consider:")
        debug_logger.log("   - Separating block detection into a dedicated model")
        debug_logger.log("   - Task-specific feature extractors for each detection task")
        debug_logger.log("   - Attention mechanisms to focus on relevant parts of the image")
    else:
        debug_logger.log("Block detection accuracy is reasonable, focus on improving other aspects.")
    
    debug_logger.log("\nCheck the complete debug log for detailed analysis of model behavior.")
    debug_logger.log("End of debugging session.")


if __name__ == "__main__":
    main()