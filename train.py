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

# -------- IMPROVED MODEL ARCHITECTURE --------
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
        
        # *** CHANGE: Separate pathways for block detection vs ball detection ***
        
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
        
        # Initialize weights with better approach to avoid poor initialization
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
        
        # Special initialization for output biases to avoid stuck predictions
        # Block detection - initialize biases to be more likely to predict "blocked"
        # (countering the class imbalance)
        nn.init.constant_(self.block_detection_head[-1].bias, 0.5)
        
        # Ball detection - initialize biases to balance predictions
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


# -------- IMPROVED LOSS FUNCTION --------
class RebalancedRedBallLoss(nn.Module):
    """Custom loss function that better handles class imbalance and task balancing"""
    def __init__(self, block_weight=10.0, position_weight=1.0, ball_exists_weight=1.0, device='cuda'):
        super(RebalancedRedBallLoss, self).__init__()
        self.block_weight = block_weight
        self.position_weight = position_weight
        self.ball_exists_weight = ball_exists_weight
        
        # Use focal loss parameters to address class imbalance
        self.gamma = 2.0  # Focusing parameter
        self.alpha = 0.25  # Balance parameter
        
        # Initialize class weights based on dataset statistics
        # Higher weights for minority class (blocked=1)
        self.block_pos_weight = torch.tensor([4.0, 4.0, 4.0], device=device)
        
        # Mean squared error for position
        self.position_loss_fn = nn.MSELoss(reduction='none')
        
    def focal_loss(self, pred, target):
        """Focal loss for binary classification with class imbalance"""
        # Binary cross entropy
        bce = -(target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
        
        # Apply focal scaling factor
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha balancing
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        return (focal_weight * alpha_weight * bce).mean()
    
    def forward(self, pred, target):
        # -- Extract components --
        # Block status (first 3 features)
        block_pred = pred[:, 0:3]
        block_target = target[:, 0:3]
        
        # Position (features 3 and 4)
        pos_pred = pred[:, 3:5]
        pos_target = target[:, 3:5]
        
        # Ball exists (feature 5)
        ball_exists_pred = pred[:, 5]
        ball_exists_target = target[:, 5]
        
        # -- Compute individual losses --
        # Block status loss with focal loss for each feature
        block_loss = 0
        for i in range(3):
            feature_loss = self.focal_loss(block_pred[:, i], block_target[:, i])
            block_loss += feature_loss * self.block_pos_weight[i]
        block_loss = block_loss / 3  # Average across features
        
        # Ball exists loss with focal loss
        ball_exists_loss = self.focal_loss(ball_exists_pred, ball_exists_target)
        
        # Position loss - only consider when ball exists
        # Get ground truth ball existence (for loss weighting)
        ball_exists = ball_exists_target.unsqueeze(1)  # Shape: [batch_size, 1]
        
        # Calculate position loss for all samples
        position_losses = self.position_loss_fn(pos_pred, pos_target)
        
        # Multiply by ball_exists (0 when no ball, 1 when ball exists)
        weighted_position_losses = position_losses * ball_exists.expand_as(position_losses)
        
        # Compute the mean, but only over samples where ball exists
        # Add a small epsilon to avoid division by zero
        num_balls = torch.sum(ball_exists) + 1e-6
        position_loss = torch.sum(weighted_position_losses) / num_balls
        
        # -- Combine losses with task weights --
        total_loss = (
            self.block_weight * block_loss + 
            self.position_weight * position_loss + 
            self.ball_exists_weight * ball_exists_loss
        )
        
        return total_loss, {
            'block_loss': block_loss.item(),
            'position_loss': position_loss.item(),
            'ball_exists_loss': ball_exists_loss.item(),
            'total_loss': total_loss.item()
        }


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

    metrics_str = ""
    for i, (feature_name, expected_type) in enumerate(feature_metric_mapping):
        if i >= len(val_feature_metrics):
            continue
            
        metric_type, value = val_feature_metrics[i]
        if metric_type == "accuracy":
            metrics_str += f"{feature_name}={value:.4f} "
        else:
            metrics_str += f"{feature_name}={value:.4f} "
            
    return metrics_str


def evaluate_model(model, data_loader, device, env_name):
    model.eval()
    total_loss_dict = {
        'block_loss': 0.0,
        'position_loss': 0.0,
        'ball_exists_loss': 0.0,
        'total_loss': 0.0
    }
    predictions = []
    targets = []

    # criterion = RebalancedRedBallLoss(device=device)
    # Even stronger emphasis on block detection
    criterion = RebalancedRedBallLoss(block_weight=20.0, position_weight=1.0, ball_exists_weight=2.0, device=device)

    with torch.no_grad():
        for images, labels_true in data_loader:
            images, labels_true = images.to(device), labels_true.to(device)
            labels_pred = model(images)
            loss, loss_dict = criterion(labels_pred, labels_true)
            
            for key in total_loss_dict:
                total_loss_dict[key] += loss_dict[key]

            predictions.extend(labels_pred.cpu().numpy())
            targets.extend(labels_true.cpu().numpy())

    # Average the losses
    for key in total_loss_dict:
        total_loss_dict[key] /= len(data_loader)

    avg_loss = total_loss_dict['total_loss']
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
            try:
                r2_scores[i] = r2_score(targets[:, i], predictions[:, i])
            except:
                r2_scores[i] = 0.0
            
        # Position features - only when redball exists
        redball_exists_mask = targets[:, 5] > 0.5
        if np.sum(redball_exists_mask) > 0:
            for i in [3, 4]:
                try:
                    pred_filtered = predictions[redball_exists_mask, i]
                    target_filtered = targets[redball_exists_mask, i]
                    r2_scores[i] = r2_score(target_filtered, pred_filtered)
                except:
                    r2_scores[i] = 0.0
        else:
            for i in [3, 4]:
                r2_scores[i] = float('nan')
    else:
        # For other environments (unchanged)
        for i in range(n_features):
            try:
                r2_scores[i] = r2_score(targets[:, i], predictions[:, i])
            except:
                r2_scores[i] = 0.0

    return avg_loss, mse_per_feature, feature_metrics, r2_scores, total_loss_dict


def save_metrics(metrics, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)


def main():
    # Configuration
    env_n = "redball"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 300
    batch_size = 128
    learning_rate = 0.001  # Increased learning rate
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join('logs', env_n, "improved_two_stage_" + current_time)
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get input image shape
    first_image, _ = train_dataset[0]
    image_shape = first_image.shape
    num_channels = image_shape[0] if len(image_shape) == 3 else 1
    image_height = image_shape[1] if len(image_shape) == 3 else image_shape[0]
    image_width = image_shape[2] if len(image_shape) == 3 else image_shape[1]

    # Initialize model, loss function, optimizer, and scheduler
    model = TwoStageDetectCNN(num_channels, image_height, image_width).to(device)
       
    criterion = RebalancedRedBallLoss(block_weight=10.0, position_weight=1.0, ball_exists_weight=2.0, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Added weight decay and switched to AdamW
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)  # Better scheduler

    # Training setup
    best_val_loss = float('inf')
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'epoch_metrics': []
    }

    writer = SummaryWriter(log_dir=log_path)
    model_save_path = os.path.join(log_path, 'best_detection.pth')
    last_model_save_path = os.path.join(log_path, 'last_detection.pth')
    learning_curve_save_path = os.path.join(log_path, 'learning_curve.png')
    metrics_save_path = os.path.join(log_path, 'training_metrics.json')

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
        train_loss_dict = {
            'block_loss': 0.0,
            'position_loss': 0.0,
            'ball_exists_loss': 0.0,
            'total_loss': 0.0
        }
        batch_count = 0

        # Training phase with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels_true in train_pbar:
            images, labels_true = images.to(device), labels_true.to(device)
            
            # Forward pass
            labels_pred = model(images)
            loss, loss_dict = criterion(labels_pred, labels_true)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update loss tracking
            epoch_train_loss += loss.item()
            for key in train_loss_dict:
                train_loss_dict[key] += loss_dict[key]
            batch_count += 1

            # Calculate batch metrics for display
            batch_metrics = evaluate_batch(labels_pred, labels_true, env_n)
            metrics_str = format_metrics_string(batch_metrics)
            
            # Show compact progress with loss and metrics
            loss_str = f"total={loss.item():.4f}"
            train_pbar.set_postfix({'loss': loss_str, 'metrics': metrics_str})

        # Average the train losses
        avg_train_loss = epoch_train_loss / batch_count
        for key in train_loss_dict:
            train_loss_dict[key] /= batch_count
            
        metrics['train_losses'].append(avg_train_loss)

        # Validation phase
        val_loss, val_mse_per_feature, val_feature_metrics, val_r2_scores, val_loss_dict = evaluate_model(
            model, val_loader, device, env_n
        )
        metrics['val_losses'].append(val_loss)
        
        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save current epoch metrics
        epoch_metric = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_loss_components': train_loss_dict,
            'val_loss': val_loss,
            'val_loss_components': val_loss_dict,
            'learning_rate': current_lr,
            'val_mse_per_feature': val_mse_per_feature.tolist(),
            'val_feature_metrics': [(metric_type, float(value)) for metric_type, value in val_feature_metrics],
            'val_r2_scores': val_r2_scores.tolist()
        }
        metrics['epoch_metrics'].append(epoch_metric)

        # Save metrics periodically
        if (epoch + 1) % 5 == 0:
            save_metrics(metrics, metrics_save_path)
            # Also save the current model state
            torch.save(model.state_dict(), last_model_save_path)

        # TensorBoard logging
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)
        
        # Log detailed loss components
        for key in train_loss_dict:
            writer.add_scalar(f"Loss/{key}_train", train_loss_dict[key], epoch)
            writer.add_scalar(f"Loss/{key}_val", val_loss_dict[key], epoch)
        
        # Log feature-specific metrics
        for i, (metric_type, value) in enumerate(val_feature_metrics):
            feature_name = feature_names[i]
            if metric_type == "accuracy":
                writer.add_scalar(f"Accuracy/{feature_name}", value, epoch)
            else:
                writer.add_scalar(f"StdDev/{feature_name}", value, epoch)

        # Display compact epoch summary after tqdm
        val_metrics_str = format_metrics_string(val_feature_metrics)
        print(f"E{epoch+1} - Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}, Metrics: {val_metrics_str}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved: {best_val_loss:.4f}")
            
            # Quick test set check for best model
            if (epoch + 1) % 10 == 0:
                test_loss, _, test_metrics, _, _ = evaluate_model(model, test_loader, device, env_n)
                test_metrics_str = format_metrics_string(test_metrics)
                print(f"Test: {test_loss:.4f}, Metrics: {test_metrics_str}")

        # Early stopping check
        if epoch > 50 and all(metrics['val_losses'][-5] < metrics['val_losses'][-5+i] for i in range(1, 5)):
            print("Early stopping triggered - validation loss not improving.")
            break

    # Final evaluation on test set
    print("\nFinal Evaluation on Test Set:")
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_mse_per_feature, feature_metrics, r2_scores, test_loss_dict = evaluate_model(model, test_loader, device, env_n)

    print(f'Test Loss: {test_loss:.4f}')
    
    # Display final metrics in a compact format
    test_metrics_str = format_metrics_string(feature_metrics)
    print(f"Final Test Metrics: {test_metrics_str}")

    # Separate evaluation for position features only when redball exists
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
        
        print(f"Ball position (when exists): X: StdDev={x_std:.4f}, Y: StdDev={y_std:.4f}")
        
        # Calculate distance error (Euclidean distance between predicted and actual positions)
        euclidean_distances = np.sqrt(x_diff**2 + y_diff**2)
        mean_distance = np.mean(euclidean_distances)
        median_distance = np.median(euclidean_distances)
        std_distance = np.std(euclidean_distances)
        
        print(f"Position Error: Mean={mean_distance:.4f}, Median={median_distance:.4f}")
    else:
        print("No samples with red ball found in test set.")

    # Plot and save learning curves
    plt.figure(figsize=(18, 8))
    fontsize = 18

    plt.subplot(121)
    plt.plot(metrics['train_losses'][1:], '-o', label="Training")
    plt.plot(metrics['val_losses'][1:], '-o', label="Validation")
    plt.title("Loss Curves", fontsize=fontsize)
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(122)
    # Extract accuracy over time for all features
    epochs = range(1, len(metrics['epoch_metrics']) + 1)
    block_accs = {feature: [] for feature in feature_names[:3]}  # Just block features
    ball_exists_acc = []
    
    for epoch_data in metrics['epoch_metrics']:
        val_metrics = epoch_data['val_feature_metrics']
        for i, (metric_type, value) in enumerate(val_metrics):
            if i < 3:  # Block features
                block_accs[feature_names[i]].append(value)
            elif i == 3:  # Ball exists
                ball_exists_acc.append(value)
    
    for feature, accs in block_accs.items():
        plt.plot(epochs, accs, '-o', label=feature)
    plt.plot(epochs, ball_exists_acc, '-o', label='ball_exists')
    plt.title("Binary Features Accuracy", fontsize=fontsize)
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.ylabel("Accuracy", fontsize=fontsize)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(learning_curve_save_path)
    plt.close()
    
    # Create confusion matrices for all binary features
    plt.figure(figsize=(20, 5))
    
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
    
    binary_features = [(0, "Left Blocked"), (1, "Front Blocked"), (2, "Right Blocked"), (5, "Ball Exists")]
    
    for i, (idx, name) in enumerate(binary_features):
        plt.subplot(1, 4, i+1)
        
        pred_binary = (all_predictions_binary[:, idx] > 0.5).astype(int)
        target_binary = (all_targets_binary[:, idx] > 0.5).astype(int)
        
        cm = confusion_matrix(target_binary, pred_binary, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
        plt.title(f"{name}", fontsize=fontsize-2)
        plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'binary_confusion_matrices.png'))
    plt.close()

    # Save final metrics
    save_metrics(metrics, metrics_save_path)
    writer.close()

    print(f"\nTraining complete! Best model saved to: {model_save_path}")


if __name__ == "__main__":
    main()