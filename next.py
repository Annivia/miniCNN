import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define adjustable prefixes
prefix = "upper"# "lower" 

# Define paths
TRAIN_CSV = f"highway_data/train_{prefix}.csv"  # or use upper_prefix
TEST_CSV = f"highway_data/test_{prefix}.csv"    # or use upper_prefix
TRAIN_IMG_DIR = "/home/xzhang3205/full_dataset/highway_train"
TEST_IMG_DIR = "/home/xzhang3205/full_dataset/highway_test"

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# For debugging data issues
def analyze_data_distribution(dataset, name):
    """Analyze the distribution of target values to help diagnose issues"""
    targets = []
    for i in range(len(dataset)):
        _, target, _ = dataset[i]
        targets.append(target.item())
    
    targets = np.array(targets)
    print(f"\n{name} Dataset Statistics:")
    print(f"Number of samples: {len(targets)}")
    print(f"Min value: {np.min(targets):.4f}")
    print(f"Max value: {np.max(targets):.4f}")
    print(f"Mean: {np.mean(targets):.4f}")
    print(f"Median: {np.median(targets):.4f}")
    print(f"Std dev: {np.std(targets):.4f}")
    
    # Count NaN or infinite values
    nan_count = np.isnan(targets).sum()
    inf_count = np.isinf(targets).sum()
    print(f"NaN values: {nan_count}")
    print(f"Infinite values: {inf_count}")
    
    # Count how many are exactly zero (potential issues)
    zero_count = (targets == 0.0).sum()
    print(f"Exactly zero values: {zero_count} ({zero_count/len(targets)*100:.2f}%)")
    
    # Check for extreme outliers (beyond 3 std devs)
    mean, std = np.mean(targets), np.std(targets)
    outliers = np.abs(targets - mean) > 3 * std
    print(f"Outliers (beyond 3 std devs): {outliers.sum()} ({outliers.sum()/len(targets)*100:.2f}%)")
    
    return targets

# Define the dataset class
class HighwayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_prefix="lower", filter_outliers=True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_prefix = target_prefix
        self.filter_outliers = filter_outliers
        
        # Skip the first row if it's a header
        if 'file_name' in str(self.data.iloc[0, 0]):
            print("Header row detected in CSV, skipping first row")
            self.data = self.data.iloc[1:].reset_index(drop=True)
        
        # Preprocess targets - filter out invalid values
        self.preprocess_targets()
            
    def preprocess_targets(self):
        col_idx = 3 if self.target_prefix == "lower" else 4
        # Convert to numeric, forcing errors to NaN
        self.data.iloc[:, col_idx] = pd.to_numeric(self.data.iloc[:, col_idx], errors='coerce')
        
        # Remove rows with NaN targets
        valid_mask = ~self.data.iloc[:, col_idx].isna()
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            print(f"Removed {invalid_count} samples with invalid {self.target_prefix} position values")
            self.data = self.data[valid_mask].reset_index(drop=True)
        
        # Filter outliers if requested (using IQR method)
        if self.filter_outliers:
            positions = self.data.iloc[:, col_idx]
            Q1 = positions.quantile(0.25)
            Q3 = positions.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = ~((positions < (Q1 - 1.5 * IQR)) | (positions > (Q3 + 1.5 * IQR)))
            outlier_count = (~outlier_mask).sum()
            if outlier_count > 0:
                print(f"Filtered {outlier_count} outliers in {self.target_prefix} position values")
                self.data = self.data[outlier_mask].reset_index(drop=True)
        
        # Normalize target values to [0, 1] range for better training
        col_data = self.data.iloc[:, col_idx]
        self.target_min = col_data.min()
        self.target_max = col_data.max()
        self.target_range = self.target_max - self.target_min
        print(f"{self.target_prefix} position range: {self.target_min:.4f} to {self.target_max:.4f}")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get file name from first column
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a blank image as fallback
            image = torch.zeros((3, 224, 224), dtype=torch.float32)
        
        # Get the target value based on the selected prefix
        # vehicle_adjacent_lower_lane_0_x_position is column 3
        # vehicle_adjacent_upper_lane_0_x_position is column 4
        col_idx = 3 if self.target_prefix == "lower" else 4
        target = float(self.data.iloc[idx, col_idx])
        
        # Normalize target to [0, 1] range for better training
        normalized_target = (target - self.target_min) / self.target_range
        target = torch.tensor(normalized_target, dtype=torch.float32)
        
        return image, target, idx
    
    def denormalize_target(self, normalized_value):
        """Convert normalized [0,1] prediction back to original scale"""
        return normalized_value * self.target_range + self.target_min

# Define CNN model using a pretrained backbone for better feature extraction
class HighwayCNN(nn.Module):
    def __init__(self, use_pretrained=True):
        super(HighwayCNN, self).__init__()
        
        # Use ResNet18 as a pretrained backbone for better feature extraction
        self.backbone = models.resnet18(weights='DEFAULT' if use_pretrained else None)
        
        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        
        # Remove the original FC layer and add our custom regressor
        self.backbone.fc = nn.Identity()
        
        # Custom regression head
        self.regressor = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        
        # Initially freeze the backbone layers to speed up training
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze the last few layers for fine-tuning
        for child in list(self.backbone.children())[-3:]:
            for param in child.parameters():
                param.requires_grad = True
        
    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features).squeeze(1)
    
    def unfreeze(self):
        """Unfreeze all backbone layers for full fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

# Data augmentation for training (more aggressive to prevent overfitting)
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# No augmentation for validation/testing
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Choose which adjacent lane to predict (lower or upper)
target_to_predict = prefix  # Change to upper_prefix if needed

# Create datasets with preprocessing
train_dataset = HighwayDataset(
    csv_file=TRAIN_CSV,
    img_dir=TRAIN_IMG_DIR,
    transform=transform_train,
    target_prefix=target_to_predict,
    filter_outliers=True
)

test_dataset = HighwayDataset(
    csv_file=TEST_CSV,
    img_dir=TEST_IMG_DIR,
    transform=transform_val,
    target_prefix=target_to_predict,
    filter_outliers=True
)

# Analyze data distributions
train_targets = analyze_data_distribution(train_dataset, "Training")
test_targets = analyze_data_distribution(test_dataset, "Test")

# Split train dataset into train and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# Initialize model, loss function, and optimizer
model = HighwayCNN(use_pretrained=True).to(device)
criterion = nn.MSELoss()

# Use a smaller learning rate for pretrained model
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training function with early stopping and mixed precision
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50):
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    best_val_loss = float('inf')
    best_model_dict = None
    patience = 10  # for early stopping
    counter = 0    # early stopping counter
    train_losses = []
    val_losses = []
    val_errors = []
    
    print(f"Training model to predict vehicle_adjacent_{target_to_predict}_lane_0_x_position")
    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, targets, _ in pbar:
            images, targets = images.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Use mixed precision for faster training
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                # Scale the gradients and call backward()
                scaler.scale(loss).backward()
                
                # Unscale the gradients before clipping
                scaler.unscale_(optimizer)
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Step the optimizer and update the scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # Update progress bar with current loss
            pbar.set_postfix({"train_loss": loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validate
        model.eval()
        val_loss, mean_error, std_error = validate(model, val_loader, criterion, train_dataset, device)
        val_losses.append(val_loss)
        val_errors.append(mean_error)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print statistics
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Mean Error: {mean_error:.4f} (Raw: {mean_error * train_dataset.target_range:.4f}), "
              f"Std Error: {std_error:.4f}")
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_dict = model.state_dict().copy()
            counter = 0  # Reset early stopping counter
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'mean_error': mean_error,
                'std_error': std_error,
                'target_min': train_dataset.target_min,
                'target_max': train_dataset.target_max,
                'target_range': train_dataset.target_range
            }, f"best_model_{target_to_predict}.pth")
            print(f"New best model saved!")
        else:
            counter += 1
            
        # Unfreeze backbone after a few epochs of training the head
        if epoch == 5:
            print("Unfreezing backbone for fine-tuning...")
            model.unfreeze()
            # Reduce learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
        
        # Early stopping
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_dict:
        model.load_state_dict(best_model_dict)
    
    return model, train_losses, val_losses, val_errors

# Validation function
def validate(model, dataloader, criterion, train_dataset, device):
    model.eval()
    running_loss = 0.0
    all_errors = []
    
    with torch.no_grad():
        for images, targets, _ in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            
            # Calculate absolute errors
            errors = torch.abs(outputs - targets).cpu().numpy()
            all_errors.extend(errors)
    
    val_loss = running_loss / len(dataloader.dataset)
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    
    return val_loss, mean_error, std_error

# Test function
def test(model, dataloader, criterion, train_dataset, device):
    model.eval()
    running_loss = 0.0
    all_errors = []
    predictions = []
    targets_list = []
    
    with torch.no_grad():
        for images, targets, idx in tqdm(dataloader, desc="Testing"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            
            # Calculate absolute errors (normalized)
            errors = torch.abs(outputs - targets).cpu().numpy()
            all_errors.extend(errors)
            
            # Store predictions and targets for later analysis
            predictions.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
    
    test_loss = running_loss / len(dataloader.dataset)
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    
    # Convert errors back to original scale
    raw_mean_error = mean_error * train_dataset.target_range
    
    # Calculate R-squared
    targets_array = np.array(targets_list)
    predictions_array = np.array(predictions)
    
    # Convert to unnormalized values
    unnorm_targets = targets_array * train_dataset.target_range + train_dataset.target_min
    unnorm_predictions = predictions_array * train_dataset.target_range + train_dataset.target_min
    
    # Calculate R-squared (coefficient of determination)
    ss_total = np.sum((unnorm_targets - np.mean(unnorm_targets)) ** 2)
    ss_residual = np.sum((unnorm_targets - unnorm_predictions) ** 2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean Absolute Error (normalized): {mean_error:.4f}")
    print(f"Mean Absolute Error (original units): {raw_mean_error:.4f}")
    print(f"Std Deviation: {std_error:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    
    return test_loss, mean_error, raw_mean_error, std_error, r_squared, unnorm_predictions, unnorm_targets

# Main execution
if __name__ == "__main__":
    # Train the model with improved techniques
    model, train_losses, val_losses, val_errors = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50
    )
    
    # Print summary
    print("\nTraining complete!")
    print(f"Best validation loss: {min(val_losses):.4f}")
    print(f"Best validation error: {min(val_errors):.4f}")
    
    # Load and evaluate the best model
    checkpoint = torch.load(f"best_model_{target_to_predict}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nEvaluating best model on test set...")
    test_loss, mean_error, raw_mean_error, std_error, r_squared, predictions, targets = test(
        model, test_loader, criterion, train_dataset, device
    )
    
    print(f"\nFinal Results for {target_to_predict} lane vehicle position prediction:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean Absolute Error: {mean_error:.4f} (normalized), {raw_mean_error:.4f} (original units)")
    print(f"R-squared: {r_squared:.4f}")