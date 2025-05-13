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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class DetectCNN(nn.Module):
    def __init__(self, env_n, num_channels, img_size_x, img_size_y):
        super(DetectCNN, self).__init__()

        self.env_n = env_n  # boxing, skiing, lb_foraging, mario
        if self.env_n == "redball":
            self.output_dim = 6
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
    binary_loss = nn.BCELoss()
    float_loss = nn.MSELoss()
    categorical_loss = nn.CrossEntropyLoss()

    if env_name == "redball":
        binary_target = torch.clamp(target[:, :3], 0, 1)
        loss_binary = binary_loss(pred[:, :3], binary_target)
        loss_float = float_loss(pred[:, 3:], target[:, 3:])
        # loss_float = categorical_loss(pred[:, 3:], target[:, 3:])
        total_loss = loss_binary + loss_float

    else:
        total_loss = float_loss(target, pred)
    return total_loss


def format_metrics_string(feature_metrics):
    metrics_str = ""
    for i, (metric_type, value) in enumerate(feature_metrics):
        if metric_type == "accuracy":
            metrics_str += f"F{i + 1}(acc):{value:.3f} "
        else:
            metrics_str += f"F{i + 1}(σ):{value:.3f} "
    return metrics_str.strip()


def evaluate_batch(pred, target, env_name):
    """Evaluate a single batch of predictions"""
    predictions = pred.detach().cpu().numpy()
    targets = target.detach().cpu().numpy()
    feature_metrics = []
    n_features = predictions.shape[1]

    if env_name == "redball":
        for i in range(n_features):
            if i in [0, 1, 2]:
                pred_binary = (predictions[:, i] > 0.5).astype(int)
                target_binary = (targets[:, i] > 0.5).astype(int)
                accuracy = (pred_binary == target_binary).mean()
                feature_metrics.append(("accuracy", accuracy))
            else:
                # diff = predictions[:, i] - targets[:, i]
                # mean, std_dev = np.mean(diff), np.std(diff)
                # feature_metrics.append(("std_dev", std_dev))

                # preds = predictions.round().astype(int)
                # accuracy = (preds == targets).float().mean().item()
                # feature_metrics.append(("accuracy", accuracy))
                preds = torch.from_numpy(predictions[:, i])      \
                   .to(device)                   \
                   .round()                      \
                   .long()
                # targets = targets.long()
                targ = torch.from_numpy(targets[:, i]) \
                    .to(device) \
                    .round()    \
                    .long()
                accuracy = (preds == targ).float().mean()
                feature_metrics.append(("accuracy", accuracy))

    else:
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
        for i in range(n_features):
            if i in [0, 1, 2]:
                pred_binary = (predictions[:, i] > 0.5).astype(int)
                target_binary = (targets[:, i] > 0.5).astype(int)
                accuracy = (pred_binary == target_binary).mean()
                feature_metrics.append(("accuracy", accuracy))
            else:
                # diff = predictions[:, i] - targets[:, i]
                # mean, std_dev = np.mean(diff), np.std(diff)
                # feature_metrics.append(("std_dev", std_dev))

                # preds = predictions.round().long()  # or .int()
                # accuracy = (preds == targets).float().mean().item()
                # feature_metrics.append(("accuracy", accuracy))
                preds = torch.from_numpy(predictions[:, i]) \
                    .to(device) \
                    .round() \
                    .long()
                # targets = targets.long()
                targ = torch.from_numpy(targets[:, i]) \
                    .to(device) \
                    .round() \
                    .long()
                accuracy = (preds == targ).float().mean()
                feature_metrics.append(("accuracy", accuracy))

    else:
        # For other environments, all features are continuous
        for i in range(n_features):
            diff = predictions[:, i] - targets[:, i]
            std_dev = np.std(diff)
            feature_metrics.append(("std_dev", std_dev))

    # Also calculate MSE per feature for backward compatibility
    mse_per_feature = ((predictions - targets) ** 2).mean(axis=0)

    # Calculate R^2 score per feature
    r2_scores = []
    for i in range(n_features):
        r2 = r2_score(targets[:, i], predictions[:, i])
        r2_scores.append(r2)

    return avg_loss, mse_per_feature, feature_metrics, r2_scores


def save_metrics(metrics, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)


env_n = "redball"  # redball, skiing, lb_foraging, mario
backbone = "scratch"  # mobilenet, efficientnet, scratch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 300
batch_size = 128
learning_rate = 0.0001
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join('logs', env_n, backbone + "_" + current_time)
os.makedirs(log_path, exist_ok=True)

if __name__ == "__main__":
    # Dataset initialization
    dataset_dir = f'redball_data'
    train_valid_dataset = CustomImageDataset(dataset_dir, env_n, size=10000,
        csv_file='redball_data/train.csv',
        # image_file=f'{dataset_dir}/{env_n}_{model_n}_{str(10000)}.npy'
    )
    test_dataset = CustomImageDataset(dataset_dir, env_n, size=2000,
        csv_file='redball_data/test.csv',
        # image_file=f'{dataset_dir}/{env_n}_{model_n}_{str(2000)}.npy'
    )

    train_size = int(0.8 * len(train_valid_dataset))
    train_dataset, val_dataset = random_split(
        train_valid_dataset, [train_size, len(train_valid_dataset) - train_size]
    )

    print("Size for each set: ", len(train_dataset), len(val_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    first_image, _ = train_dataset[0]
    image_shape = first_image.shape  # (10000, 3, 210, 160)
    num_channels = image_shape[0] if len(image_shape) == 3 else 1
    image_height = image_shape[1] if len(image_shape) == 3 else image_shape[0]
    image_width = image_shape[2] if len(image_shape) == 3 else image_shape[1]

    model = DetectCNN(env_n, num_channels, image_height, image_width).to(device)
       
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

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

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        epoch_train_loss = 0.0

        if epoch == 20 and backbone != "scratch":
            for param in model.backbone.parameters():
                param.requires_grad = True

            optimizer = optim.Adam([
                {"params": model.backbone.parameters(), "lr": 1e-5},
                {"params": model.mlp.parameters(), "lr": 1e-4}  # still higher LR for head
            ])
            # Reinitialize scheduler with the new optimizer
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        # Training phase with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for images, labels_true in train_pbar:
            images, labels_true = images.to(device), labels_true.to(device)
            labels_pred = model(images)

            loss = compute_loss(labels_pred, labels_true, env_n)
            # print(np.any(np.isnan(labels_pred.cpu().detach().numpy())), np.any(np.isnan(labels_true.cpu().detach().numpy())))
            if np.any(np.isnan(labels_true.cpu().detach().numpy())):
                raise ValueError
            if np.any(np.isnan(labels_pred.cpu().detach().numpy())):
                raise ValueError

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

            # Calculate and display batch metrics
            batch_metrics = evaluate_batch(labels_pred, labels_true, env_n)
            metrics_str = format_metrics_string(batch_metrics)
            train_pbar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
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
            'val_feature_metrics': [(metric_type, float(value)) for metric_type, value in val_feature_metrics]
        }
        metrics['epoch_metrics'].append(epoch_metric)

        # Save metrics periodically
        if (epoch + 1) % 5 == 0:
            save_metrics(metrics, metrics_save_path)

        # TensorBoard logging
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)

        # Learning rate scheduling
        scheduler.step()

        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print("Validation Metrics:", format_metrics_string(val_feature_metrics))
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"======= New best model saved with validation loss: {best_val_loss:.4f} =======")

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    test_loss, test_mse_per_feature, feature_metrics, r2_scores = evaluate_model(model, test_loader, device, env_n)

    print(f'\nTest Loss: {test_loss:.4f}')
    print("\nDetailed Feature Metrics:")
    for i, (metric_type, value) in enumerate(feature_metrics):
        if metric_type == "accuracy":
            print(f"Feature {i + 1} (Binary) - Accuracy: {value:.4f}")
        else:
            print(f"Feature {i + 1} (Continuous) - Std Dev of Difference: {value:.4f}")

    print("\nMSE amd R^2 per feature:")
    # for i, mse in enumerate(test_mse_per_feature):
    #     print(f"Feature {i + 1}: {mse:.4f}")
    # for i, r2 in enumerate(r2_scores):
    #     print(f"Feature {i + 1}: R^2 = {r2:.4f}")
    for i, (mse, r2) in enumerate(zip(test_mse_per_feature, r2_scores)):
        print(f"Feature {i + 1}: MSE = {mse:.6f}, R^2 = {r2:.6f}")

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

    # Save final metrics
    save_metrics(metrics, metrics_save_path)
    writer.close()
