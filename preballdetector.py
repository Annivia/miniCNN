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
    

class SingleImageProcessor:
    def __init__(self, image_folder):
        self.image_folder = image_folder
    
    def process_image(self, image_name):
        """Process a single image and return the tensor"""
        image_path = os.path.join(self.image_folder, image_name)
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} does not exist")
            return None
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)  # (C, H, W)
        image = image / 255.0  # Normalize
        
        return torch.tensor(image, dtype=torch.float).unsqueeze(0)  # Add batch dimension


def preprocess():
    # Configuration parameters
    model_path = 'binary_model_output/binary.pth'  # Change this to your model path
    train_csv = 'redball_data/test.csv'        # Change this to your training CSV
    output_csv = 'redball_data/test_filtered.csv'      # Output CSV filename
    image_folder = 'redball_images/redball_test_filtered'  # Image folder
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    print("Loading model...")
    model = BinaryDetectCNN(num_channels=3, img_size_x=128, img_size_y=128)  # Adjust dimensions as needed
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Load training data
    print(f"Loading training data from {train_csv}")
    try:
        df = pd.read_csv(train_csv, header=None)
        print(f"Loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Initialize image processor
    processor = SingleImageProcessor(image_folder)
    
    # Process each image and filter based on ball existence prediction
    filtered_rows = []
    threshold = 0.5  # Threshold for binary classification
    
    print("Processing images...")
    with torch.no_grad():  # Disable gradient computation for inference
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image_name = row[0]
            
            # Process the image
            image_tensor = processor.process_image(image_name)
            if image_tensor is None:
                continue
                
            # Move to device and get prediction
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            
            # Check if ball exists (index 5 in output)
            ball_exists = output[0, 5].item() > threshold
            
            if ball_exists:
                filtered_rows.append(row)
    
    # Create filtered dataframe
    filtered_df = pd.DataFrame(filtered_rows)
    
    # Save to CSV
    print(f"Saving {len(filtered_df)} filtered rows to {output_csv}")
    filtered_df.to_csv(output_csv, header=False, index=False)
    print("Done!")

if __name__ == "__main__":
    preprocess()