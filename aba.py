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
from sklearn.metrics import r2_score

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
        # Extract features
        features = self.feature_extractor(x)
        
        # First stage: detect obstacles and ball existence
        detection_output = self.detection_head(features)
        detection_sigmoid = torch.sigmoid(detection_output)
        
        # Extract redball_exists prediction
        redball_exists = detection_sigmoid[:, 3:4]  # Keep dimension for concatenation
        
        # Process features for position estimation
        position_features = self.position_encoder(features)
        
        # Concatenate position features with redball_exists
        position_input = torch.cat([position_features, redball_exists], dim=1)
        
        # Second stage: predict position based on features and redball_exists
        position_output = self.position_mlp(position_input)
        
        # For visualization/debugging: zero out positions when ball doesn't exist
        # During training, this doesn't affect gradients - it's just for better visualization
        redball_exists_mask = (redball_exists > 0.5).float()
        visualized_position = position_output * redball_exists_mask
        
        # Concatenate all outputs in the original order [left, front, right, x, y, exists]
        # The first 3 are binary obstacle detections
        # The next 2 are position coordinates 
        # The last one is redball existence
        return torch.cat([
            detection_sigmoid[:, :3],  # blocked features (binary)
            visualized_position,       # position features with optional zeroing
            detection_sigmoid[:, 3:4]  # redball_exists (binary)
        ], dim=1)


class CustomImageDataset(Dataset):
    def __init__(self, dataset_dir, env_name, size, csv_file, transform=None):
        self.env_name = env_name
        print("labels are at: ", csv_file)
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


def compute_loss(pred, target, env_name):
    binary_loss_fn = nn.BCELoss()
    position_loss_fn = nn.MSELoss(reduction='none')
    
    # For redball environment
    if env_name == "redball":
        # ---- First handle binary features ----
        # Extract binary features: left/front/right blocked + redball_exists
        binary_indices = [0, 1, 2, 5]  # Indices in the prediction vector
        binary_target = target[:, binary_indices]  
        binary_pred = pred[:, binary_indices]
        
        # Binary classification loss for all binary features
        loss_binary = binary_loss_fn(binary_pred, binary_target)
        
        # ---- Then handle position features ----
        # Extract position features: x, y coordinates
        position_target = target[:, 3:5]
        position_pred = pred[:, 3:5]
        
        # Get ground truth ball existence (for loss weighting)
        ball_exists = target[:, 5].unsqueeze(1)  # Shape: [batch_size, 1]
        
        # Two approaches to handle the conditional position loss:
        
        # Approach 1: Use the ball_exists as a weight multiplier
        # Calculate position loss for all samples
        position_losses = position_loss_fn(position_pred, position_target)
        
        # Multiply by ball_exists (0 when no ball, 1 when ball exists)
        # This zeros out the loss for samples where no ball exists
        weighted_position_losses = position_losses * ball_exists.expand_as(position_losses)
        
        # Compute the mean, but only over samples where ball exists
        # Add a small epsilon to avoid division by zero
        num_balls = torch.sum(ball_exists) + 1e-6
        loss_position = torch.sum(weighted_position_losses) / num_balls
        
        # Balance binary and position losses with weighting factors
        alpha = 1.0  # Weight for binary loss
        beta = 2.0   # Weight for position loss (higher to emphasize position accuracy)
        
        # Total loss combines both tasks
        total_loss = alpha * loss_binary + beta * loss_position
        
        return total_loss
    else:
        # For other environments
        return nn.MSELoss()(pred, target)


def format_metrics_string(feature_metrics):
    # Explicit ordering of feature names and their expected metric type
    feature_metric_mapping = [
        ("left_blocked", "accuracy"),
        ("front_blocked", "accuracy"),
        ("right_blocked", "accuracy"),
        ("ball_exists", "accuracy"),
        ("ball_x_pos", "std_dev"),
        ("ball_y_pos", "std_dev")
    ]

    metrics_str = ""
    for i, (feature_name, expected_type) in enumerate(feature_metric_mapping):
        metric_type, value = feature_metrics[i]
        if metric_type != expected_type:
            metrics_str += f"{feature_name}(!{metric_type}):{value:.3f} "  # Flag mismatch
        else:
            label = "acc" if metric_type == "accuracy" else "std"
            metrics_str += f"{feature_name}({label}):{value:.3f} "

    return metrics_str.strip()


def evaluate_batch(pred, target, env_name):
    """Evaluate a single batch of predictions"""
    predictions = pred.detach().cpu().numpy()
    targets = target.detach().cpu().numpy()
    feature_metrics = []
    
    if env_name == "redball":
        # Binary features (obstacles + redball_exists)
        binary_indices = [0, 1, 2, 5]  # left, front, right blocked + redball_exists
        for i in binary_indices:
            pred_binary = (predictions[:, i] > 0.5).astype(int)
            target_binary = (targets[:, i] > 0.5).astype(int)
            accuracy = (pred_binary == target_binary).mean()
            feature_metrics.append(("accuracy", accuracy))
        
        # Position features (x, y)
        position_indices = [3, 4]  # x, y position
        for i in position_indices:
            # Only evaluate on samples where redball exists
            redball_exists_mask = targets[:, 5] > 0.5  # Use ground truth for masking
            
            if np.sum(redball_exists_mask) > 0:  # Check if any redball exists
                pred_filtered = predictions[redball_exists_mask, i]
                target_filtered = targets[redball_exists_mask, i]
                
                diff = pred_filtered - target_filtered
                std_dev = np.std(diff)
                feature_metrics.append(("std_dev", std_dev))
            else:
                # If no redball exists in this batch
                feature_metrics.append(("std_dev", float('nan')))
    else:
        # For other environments
        n_features = predictions.shape[1]
        for i in range(n_features):
            diff = predictions[:, i] - targets[:, i]
            std_dev = np.std(diff)
            feature_metrics.append(("std_dev", std_dev))

    return feature_metrics


def evaluate_model(model, data_loader, device, env_name):
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels_true in data_loader:
            images, labels_true = images.to(device), labels_true.to(device)
            labels_pred = model(images)
            loss = compute_loss(labels_pred, labels_true, env_name)
            total_loss += loss.item()

            predictions.extend(labels_pred.cpu().numpy())
            targets.extend(labels_true.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate per-feature metrics
    feature_metrics = []
    n_features = predictions.shape[1]

    if env_name == "redball":
        # Binary features (obstacles + redball_exists) - Use accuracy
        binary_indices = [0, 1, 2, 5]  # left, front, right blocked + redball_exists
        for i in binary_indices:
            pred_binary = (predictions[:, i] > 0.5).astype(int)
            target_binary = (targets[:, i] > 0.5).astype(int)
            accuracy = (pred_binary == target_binary).mean()
            feature_metrics.append(("accuracy", accuracy))
        
        # Position features (x, y) - Use standard deviation
        position_indices = [3, 4]  # x, y position
        for i in position_indices:
            # Only evaluate on samples where redball exists (according to ground truth)
            redball_exists_mask = targets[:, 5] > 0.5
            
            if np.sum(redball_exists_mask) > 0:  # Check if any redball exists
                pred_filtered = predictions[redball_exists_mask, i]
                target_filtered = targets[redball_exists_mask, i]
                
                diff = pred_filtered - target_filtered
                std_dev = np.std(diff)
                feature_metrics.append(("std_dev", std_dev))
            else:
                # If no redball exists in this batch
                feature_metrics.append(("std_dev", float('nan')))
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


def main():
    # Configuration
    env_n = "redball"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 300
    batch_size = 128
    learning_rate = 0.0001
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join('logs', env_n, "two_stage_" + current_time)
    os.makedirs(log_path, exist_ok=True)

    # Dataset initialization
    dataset_dir = f'redball_data'
    train_valid_dataset = CustomImageDataset(
        dataset_dir, env_n, size=10000,
        csv_file='redball_data/train.csv',
    )
    test_dataset = CustomImageDataset(
        dataset_dir, env_n, size=2000,
        csv_file='redball_data/test.csv',
    )

    train_size = int(0.8 * len(train_valid_dataset))
    train_dataset, val_dataset = random_split(
        train_valid_dataset, [train_size, len(train_valid_dataset) - train_size]
    )

    print("Size for each set: ", len(train_dataset), len(val_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get input image shape
    first_image, _ = train_dataset[0]
    image_shape = first_image.shape
    num_channels = image_shape[0] if len(image_shape) == 3 else 1
    image_height = image_shape[1] if len(image_shape) == 3 else image_shape[0]
    image_width = image_shape[2] if len(image_shape) == 3 else image_shape[1]

    # Initialize model, optimizer, and scheduler
    model = TwoStageDetectCNN(num_channels, image_height, image_width).to(device)
    
    # Print model architecture for debugging
    print(model)
    
    # Test forward pass to check model output dimensions
    with torch.no_grad():
        dummy_input = torch.zeros(1, num_channels, image_height, image_width, device=device)
        dummy_output = model(dummy_input)
        print(f"Model output shape: {dummy_output.shape}")
       
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
        model.train()
        epoch_train_loss = 0.0

        # Training phase with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for images, labels_true in train_pbar:
            images, labels_true = images.to(device), labels_true.to(device)
            labels_pred = model(images)

            loss = compute_loss(labels_pred, labels_true, env_n)
            
            # Check for NaN values
            if torch.isnan(loss):
                print("NaN loss detected! Skipping batch.")
                continue

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

            # Calculate and display batch metrics
            batch_metrics = evaluate_batch(labels_pred, labels_true, env_n)
            metrics_str = format_metrics_string(batch_metrics)
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'metrics': metrics_str
            })

        avg_train_loss = epoch_train_loss / len(train_loader)
        metrics['train_losses'].append(avg_train_loss)

        # Validation phase
        val_loss, val_mse_per_feature, val_feature_metrics, val_r2_scores = evaluate_model(model, val_loader, device, env_n)
        metrics['val_losses'].append(val_loss)

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
            
            # If we have a new best model, also evaluate on test set and print metrics
            # This helps track progress during training
            if (epoch + 1) % 10 == 0:  # Only do this every 10 epochs to save time
                print("\nIntermediate Test Set Evaluation:")
                test_loss, test_mse, test_metrics, test_r2 = evaluate_model(model, test_loader, device, env_n)
                print(f"  Test Loss: {test_loss:.4f}")
                for i, (metric_type, value) in enumerate(test_metrics):
                    feature_name = feature_names[i]
                    if metric_type == "accuracy":
                        print(f"  {feature_name}: Accuracy = {value:.4f}")
                    else:
                        print(f"  {feature_name}: StdDev = {value:.4f}")

    # Final evaluation on test set
    print("\nFinal Evaluation on Test Set:")
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_mse_per_feature, feature_metrics, r2_scores = evaluate_model(model, test_loader, device, env_n)

    print(f'Test Loss: {test_loss:.4f}')
    
    print("\nDetailed Feature Metrics:")    
    for i, (metric_type, value) in enumerate(feature_metrics):
        feature_name = feature_names[i]
        if metric_type == "accuracy":
            print(f"{feature_name} - Accuracy: {value:.4f}")
        else:
            print(f"{feature_name} - Std Dev of Difference: {value:.4f}")

    print("\nMSE and R^2 per feature:")
    for i, (mse, r2) in enumerate(zip(test_mse_per_feature, r2_scores)):
        feature_name = feature_names[i]
        print(f"{feature_name}: MSE = {mse:.6f}, R^2 = {r2:.6f}")

    # Separate evaluation for position features only when redball exists
    print("\nEvaluating red ball position prediction only for cases where the ball exists:")
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
        
        # Calculate distance error (Euclidean distance between predicted and actual positions)
        # This gives a more intuitive measure of position accuracy
        euclidean_distances = np.sqrt(x_diff**2 + y_diff**2)
        mean_distance = np.mean(euclidean_distances)
        median_distance = np.median(euclidean_distances)
        std_distance = np.std(euclidean_distances)
        
        print(f"Position Error (Euclidean distance): Mean = {mean_distance:.6f}, Median = {median_distance:.6f}, StdDev = {std_distance:.6f}")
    else:
        print("No samples with red ball found in test set.")

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
    plt.close()

    # Plot and save feature-specific metrics
    plt.figure(figsize=(18, 12))
    
    # 1. Binary features accuracy
    plt.subplot(221)
    binary_indices = [0, 1, 2, 5]  # left, front, right, exists
    binary_names = ["Left Blocked", "Front Blocked", "Right Blocked", "Ball Exists"]
    binary_accuracies = []
    
    for idx in binary_indices:
        for i, (metric_type, value) in enumerate(feature_metrics):
            if i == idx and metric_type == "accuracy":
                binary_accuracies.append(value)
    
    plt.bar(binary_names, binary_accuracies)
    plt.title("Binary Features Accuracy", fontsize=fontsize)
    plt.ylabel("Accuracy", fontsize=fontsize)
    plt.ylim(0, 1)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    # 2. Position features std deviation
    plt.subplot(222)
    position_indices = [3, 4]  # x, y
    position_names = ["X Position", "Y Position"]
    position_std = []
    
    for idx in position_indices:
        for i, (metric_type, value) in enumerate(feature_metrics):
            if i == idx and metric_type == "std_dev":
                position_std.append(value)
    
    plt.bar(position_names, position_std)
    plt.title("Position Features StdDev (Ball Exists Only)", fontsize=fontsize)
    plt.ylabel("StdDev", fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    # 3. MSE per feature
    plt.subplot(223)
    all_names = ["Left Blocked", "Front Blocked", "Right Blocked", "X Position", "Y Position", "Ball Exists"]
    plt.bar(all_names, test_mse_per_feature)
    plt.title("MSE per Feature", fontsize=fontsize)
    plt.ylabel("MSE", fontsize=fontsize)
    plt.xticks(rotation=45, fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    # 4. R^2 per feature
    plt.subplot(224)
    plt.bar(all_names, r2_scores)
    plt.title("R^2 per Feature", fontsize=fontsize)
    plt.ylabel("R^2", fontsize=fontsize)
    plt.xticks(rotation=45, fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'feature_metrics.png'))
    plt.close()

    # Create a confusion matrix for ball existence prediction
    plt.figure(figsize=(10, 8))
    
    # Collect all predictions for ball existence
    all_ball_exists_pred = []
    all_ball_exists_true = []
    
    with torch.no_grad():
        for images, labels_true in test_loader:
            images, labels_true = images.to(device), labels_true.to(device)
            labels_pred = model(images)
            
            # Extract ball existence predictions and ground truth
            ball_exists_pred = (labels_pred[:, 5] > 0.5).cpu().numpy()
            ball_exists_true = (labels_true[:, 5] > 0.5).cpu().numpy()
            
            all_ball_exists_pred.extend(ball_exists_pred)
            all_ball_exists_true.extend(ball_exists_true)
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(all_ball_exists_true, all_ball_exists_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Ball", "Ball Exists"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Ball Existence Confusion Matrix", fontsize=fontsize)
    plt.savefig(os.path.join(log_path, 'ball_exists_confusion.png'))
    plt.close()

    # Plot position error distribution (when ball exists)
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
        plt.close()

    # Save final metrics
    save_metrics(metrics, metrics_save_path)
    writer.close()
                # all_targets.append(targets[redball_exists_mask])
    
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
    else:
        print("No samples with red ball found in test set.")

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
    plt.close()

    # Plot and save feature-specific metrics
    plt.figure(figsize=(18, 12))
    
    # 1. Binary features accuracy
    plt.subplot(221)
    binary_indices = [0, 1, 2, 5]  # left, front, right, exists
    binary_names = ["Left Blocked", "Front Blocked", "Right Blocked", "Ball Exists"]
    binary_accuracies = [feature_metrics[i][1] for i in range(len(feature_metrics)) if i in binary_indices]
    
    plt.bar(binary_names, binary_accuracies)
    plt.title("Binary Features Accuracy", fontsize=fontsize)
    plt.ylabel("Accuracy", fontsize=fontsize)
    plt.ylim(0, 1)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    # 2. Position features std deviation
    plt.subplot(222)
    position_indices = [3, 4]  # x, y
    position_names = ["X Position", "Y Position"]
    position_std = [feature_metrics[i][1] for i in range(len(feature_metrics)) if i-2 in position_indices]  # Adjust index
    
    plt.bar(position_names, position_std)
    plt.title("Position Features StdDev (Ball Exists Only)", fontsize=fontsize)
    plt.ylabel("StdDev", fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    # 3. MSE per feature
    plt.subplot(223)
    all_names = ["Left Blocked", "Front Blocked", "Right Blocked", "X Position", "Y Position", "Ball Exists"]
    plt.bar(all_names, test_mse_per_feature)
    plt.title("MSE per Feature", fontsize=fontsize)
    plt.ylabel("MSE", fontsize=fontsize)
    plt.xticks(rotation=45, fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    # 4. R^2 per feature
    plt.subplot(224)
    plt.bar(all_names, r2_scores)
    plt.title("R^2 per Feature", fontsize=fontsize)
    plt.ylabel("R^2", fontsize=fontsize)
    plt.xticks(rotation=45, fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'feature_metrics.png'))
    plt.close()

    # Create a confusion matrix for ball existence prediction
    plt.figure(figsize=(10, 8))
    
    # Collect all predictions for ball existence
    all_ball_exists_pred = []
    all_ball_exists_true = []
    
    with torch.no_grad():
        for images, labels_true in test_loader:
            images, labels_true = images.to(device), labels_true.to(device)
            labels_pred = model(images)
            
            # Extract ball existence predictions and ground truth
            ball_exists_pred = (labels_pred[:, 5] > 0.5).cpu().numpy()
            ball_exists_true = (labels_true[:, 5] > 0.5).cpu().numpy()
            
            all_ball_exists_pred.extend(ball_exists_pred)
            all_ball_exists_true.extend(ball_exists_true)
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(all_ball_exists_true, all_ball_exists_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Ball", "Ball Exists"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Ball Existence Confusion Matrix", fontsize=fontsize)
    plt.savefig(os.path.join(log_path, 'ball_exists_confusion.png'))
    plt.close()

    # Plot position error distribution (when ball exists)
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
        plt.close()

    # Save final metrics
    save_metrics(metrics, metrics_save_path)
    writer.close()
    
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
    else:
        print("No samples with red ball found in test set.")

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
    plt.close()

    # Plot and save feature-specific metrics
    plt.figure(figsize=(18, 12))
    
    # 1. Binary features accuracy
    plt.subplot(221)
    binary_indices = [0, 1, 2, 5]  # left, front, right, exists
    binary_names = ["Left Blocked", "Front Blocked", "Right Blocked", "Ball Exists"]
    binary_accuracies = [feature_metrics[i][1] for i in binary_indices]
    
    plt.bar(binary_names, binary_accuracies)
    plt.title("Binary Features Accuracy", fontsize=fontsize)
    plt.ylabel("Accuracy", fontsize=fontsize)
    plt.ylim(0, 1)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    # 2. Position features std deviation
    plt.subplot(222)
    position_indices = [3, 4]  # x, y
    position_names = ["X Position", "Y Position"]
    position_std = [feature_metrics[i][1] for i in position_indices]
    
    plt.bar(position_names, position_std)
    plt.title("Position Features StdDev (Ball Exists Only)", fontsize=fontsize)
    plt.ylabel("StdDev", fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    # 3. MSE per feature
    plt.subplot(223)
    all_names = ["Left Blocked", "Front Blocked", "Right Blocked", "X Position", "Y Position", "Ball Exists"]
    plt.bar(all_names, test_mse_per_feature)
    plt.title("MSE per Feature", fontsize=fontsize)
    plt.ylabel("MSE", fontsize=fontsize)
    plt.xticks(rotation=45, fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    # 4. R^2 per feature
    plt.subplot(224)
    plt.bar(all_names, r2_scores)
    plt.title("R^2 per Feature", fontsize=fontsize)
    plt.ylabel("R^2", fontsize=fontsize)
    plt.xticks(rotation=45, fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'feature_metrics.png'))
    plt.close()

    # Save final metrics
    save_metrics(metrics, metrics_save_path)
    writer.close()


if __name__ == "__main__":
    main()