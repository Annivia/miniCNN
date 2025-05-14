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
from torchvision import transforms, models

instance = 'lower' # Could be lower or upper

# Reusing the model and dataset definitions from your code
# Define paths
TRAIN_CSV = f"highway_data/train_{instance}.csv"
TEST_CSV = f"highway_data/test_{instance}.csv"
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
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Binary targets - lane existence (columns 1 and 2)
        lower_lane_exists = float(self.data.iloc[idx, 1])
        upper_lane_exists = float(self.data.iloc[idx, 2])
        binary_targets = torch.tensor([lower_lane_exists, upper_lane_exists], dtype=torch.float32)
        
        # Continuous targets - positions (columns 5, 6, 7)
        vehicle_ahead = float(self.data.iloc[idx, 5])  # vehicle_ahead_same_lane_0_x_position
        agent_x = float(self.data.iloc[idx, 6])        # agent_0_x_position
        agent_y = float(self.data.iloc[idx, 7])        # agent_0_y_position
        continuous_targets = torch.tensor([vehicle_ahead, agent_x, agent_y], dtype=torch.float32)

        
        return image, binary_targets, continuous_targets, idx


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
    
class HighwayCNNSingle(nn.Module):
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
print("Main Model loaded successfully")

model_lower = HighwayCNNSingle().to(device)
checkpoint = torch.load('best_model_lower.pth', map_location=device)
model_lower.load_state_dict(checkpoint['model_state_dict'])
model_lower.eval()
print("Lower Model loaded successfully")

model_upper = HighwayCNNSingle().to(device)
checkpoint = torch.load('best_model_upper.pth', map_location=device)
model_upper.load_state_dict(checkpoint['model_state_dict'])
model_upper.eval()
print("Lower Model loaded successfully")

model_front = HighwayCNNSingle().to(device)
checkpoint = torch.load('best_model_front.pth', map_location=device)
model_front.load_state_dict(checkpoint['model_state_dict'])
model_front.eval()
print("Lower Model loaded successfully")