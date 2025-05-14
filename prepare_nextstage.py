import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Reusing the model and dataset definitions from your code
# Define paths
TRAIN_CSV = "highway_data/test.csv"
TEST_CSV = "highway_data/test.csv"
TRAIN_IMG_DIR = "/home/xzhang3205/full_dataset/highway_train"
TEST_IMG_DIR = "/home/xzhang3205/full_dataset/highway_test"
MODEL_PATH = "logs/highway_use/best_model.pth"
OUTPUT_DIR = "highway_data"  # Same parent directory as train.csv and test.csv

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the dataset class (same as before)
class HighwayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Skip the first row if it's a header
        if 'file_name' in str(self.data.iloc[0, 0]):
            print("Header row detected in CSV, skipping first row")
            self.data = self.data.iloc[1:].reset_index(drop=True)
            
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
            image = torch.zeros((3, 224, 224), dtype=torch.float32)
        
        # Binary targets - lane existence (columns 1 and 2)
        try:
            lower_lane_exists = float(self.data.iloc[idx, 1])
            upper_lane_exists = float(self.data.iloc[idx, 2])
            binary_targets = torch.tensor([lower_lane_exists, upper_lane_exists], dtype=torch.float32)
        except Exception as e:
            print(f"Error processing binary targets for index {idx}: {e}")
            binary_targets = torch.zeros(2, dtype=torch.float32)
        
        # Continuous targets - positions (columns 5, 6, 7)
        try:
            vehicle_ahead = float(self.data.iloc[idx, 5])  # vehicle_ahead_same_lane_0_x_position
            agent_x = float(self.data.iloc[idx, 6])        # agent_0_x_position
            agent_y = float(self.data.iloc[idx, 7])        # agent_0_y_position
            continuous_targets = torch.tensor([vehicle_ahead, agent_x, agent_y], dtype=torch.float32)
        except Exception as e:
            print(f"Error processing continuous targets for index {idx}: {e}")
            continuous_targets = torch.zeros(3, dtype=torch.float32)
        
        return image, binary_targets, continuous_targets, idx

# Define CNN model (same as before)
class HighwayCNN(nn.Module):
    def __init__(self, binary_outputs=2, continuous_outputs=3):
        super(HighwayCNN, self).__init__()
        
        # Feature extraction backbone
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling to ensure consistent feature size regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Calculate the size of flattened features
        self.flat_features = 256 * 7 * 7
        
        # Common fully connected layers
        self.fc_common = nn.Sequential(
            nn.Linear(self.flat_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # Task-specific layers
        # Binary classification head for lane existence
        self.binary_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, binary_outputs),
            nn.Sigmoid()  # Sigmoid for binary outputs
        )
        
        # Regression head for positions
        self.continuous_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, continuous_outputs)
        )
        
    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Common processing
        x = self.fc_common(x)
        
        # Task-specific outputs
        binary_out = self.binary_head(x)
        continuous_out = self.continuous_head(x)
        
        return binary_out, continuous_out

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = HighwayDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform=transform)
test_dataset = HighwayDataset(TEST_CSV, TEST_IMG_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# Load the model
model = HighwayCNN().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully")

# Function to run inference and save filtered CSVs
def run_inference_and_save(data_loader, original_csv, prefix):
    all_predictions = []
    all_ground_truth = []
    indices = []
    
    with torch.no_grad():
        for images, binary_targets, continuous_targets, idx in data_loader:
            images = images.to(device)
            
            # Forward pass
            binary_outputs, continuous_outputs = model(images)
            
            # Convert to predictions (threshold = 0.5)
            binary_preds = (binary_outputs > 0.5).float()
            
            # Store results
            all_predictions.extend(binary_preds.cpu().numpy())
            all_ground_truth.extend(binary_targets.cpu().numpy())
            indices.extend(idx.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    indices = np.array(indices)
    
    # Load original CSV
    df = pd.read_csv(original_csv)
    
    # Create a copy of the dataframe with predictions
    pred_df = df.copy()
    if 'file_name' in str(pred_df.iloc[0, 0]):  # If header row exists
        pred_df = pred_df.iloc[1:].reset_index(drop=True)
    
    # Create filtered dataframes
    lower_mask = all_predictions[:, 0] == 1
    upper_mask = all_predictions[:, 1] == 1
    
    # Get the actual indices in the original dataframe
    lower_indices = indices[lower_mask]
    upper_indices = indices[upper_mask]
    
    # Create filtered dataframes
    lower_df = pred_df.iloc[lower_indices].copy()
    upper_df = pred_df.iloc[upper_indices].copy()
    
    # Save filtered CSVs
    lower_df.to_csv(os.path.join(OUTPUT_DIR, f"{prefix}_lower.csv"), index=False)
    upper_df.to_csv(os.path.join(OUTPUT_DIR, f"{prefix}_upper.csv"), index=False)
    
    print(f"Saved {prefix}_lower.csv with {len(lower_df)} rows")
    print(f"Saved {prefix}_upper.csv with {len(upper_df)} rows")
    
    # Evaluate model
    print(f"\nEvaluation for {prefix} dataset:")
    
    # Binary classification metrics
    binary_accuracy = accuracy_score(all_ground_truth.flatten(), all_predictions.flatten())
    binary_precision = precision_score(all_ground_truth.flatten(), all_predictions.flatten(), zero_division=0)
    binary_recall = recall_score(all_ground_truth.flatten(), all_predictions.flatten(), zero_division=0)
    binary_f1 = f1_score(all_ground_truth.flatten(), all_predictions.flatten(), zero_division=0)
    
    print(f"Lane Existence - Accuracy: {binary_accuracy:.4f}, Precision: {binary_precision:.4f}, Recall: {binary_recall:.4f}, F1: {binary_f1:.4f}")
    
    # Individual lane metrics
    for i, lane in enumerate(["Lower Lane", "Upper Lane"]):
        acc = accuracy_score(all_ground_truth[:, i], all_predictions[:, i])
        prec = precision_score(all_ground_truth[:, i], all_predictions[:, i], zero_division=0)
        rec = recall_score(all_ground_truth[:, i], all_predictions[:, i], zero_division=0)
        f1 = f1_score(all_ground_truth[:, i], all_predictions[:, i], zero_division=0)
        print(f"{lane} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    return all_predictions, all_ground_truth

# Run inference and save CSVs for train and test datasets
print("Processing training dataset...")
train_preds, train_gt = run_inference_and_save(train_loader, TEST_CSV, "test")

print("\nProcessing testing dataset...")
test_preds, test_gt = run_inference_and_save(test_loader, TEST_CSV, "test")

print("\nProcessing complete!")