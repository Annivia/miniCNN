import os
import time
import pandas as pd
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import r2_score, confusion_matrix
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


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

class BallDetectionCNN(nn.Module):
    def __init__(self):
        super(BallDetectionCNN, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers for regression
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # Output 2 values for x, y coordinates
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.fc_layers(x)
        return x
    

@torch.no_grad()

def feature_extraction(image_path, binary_model_path = "binary/binary.pth", ball_model_path="/home/xzhang3205/miniCNN/logs/ball_detection.pth"):
    """
    Extract binary features from an image using the BinaryDetectCNN model.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        list: Binary features [left_blocked, front_blocked, right_blocked, ball_exists]
    """

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process the image exactly as in CustomImageDataset.__getitem__
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)  # (C, H, W)
    image = image / 255.0  # Normalize
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.tensor(image, dtype=torch.float).unsqueeze(0).to(device)
    
    # Create and load the model
    num_channels = image_tensor.shape[1]
    image_height = image_tensor.shape[2]
    image_width = image_tensor.shape[3]
    assert(image_height==448)
    assert(image_width==448)
    
    binary_model = BinaryDetectCNN(num_channels, image_height, image_width).to(device)
    binary_model.load_state_dict(torch.load(binary_model_path, map_location=device))
    binary_model.eval()

    ball_model = BallDetectionCNN().to(device)
    ball_model.load_state_dict(torch.load(ball_model_path, map_location=device))
    ball_model.eval()
    
    # For BallDetectionCNN
    ball_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    # Get predictions
    with torch.no_grad():
        output = binary_model(image_tensor)
    
    # Extract only the binary features (indices 0, 1, 2, 5)
    left_blocked = output[0, 0].item()
    front_blocked = output[0, 1].item()
    right_blocked = output[0, 2].item()
    ball_exists = output[0, 5].item()
    
    # Return the four binary features
    # return [left_blocked, front_blocked, right_blocked, ball_exists, 0, 0]

    # Default position values (will be overwritten if ball exists)
    ball_x, ball_y = 0.0, -5.0
    
    # Only run ball detection if inverted ball_exists value is positive
    if ball_exists > 0.5:
        # Load and preprocess the image for BallDetectionCNN
        ball_image = Image.open(image_path).convert('RGB')
        ball_input = ball_transform(ball_image).unsqueeze(0).to(device)
        
        # Get ball position
        with torch.no_grad():
            ball_output = ball_model(ball_input)
            ball_x = ball_output[0, 0].item()
            ball_y = ball_output[0, 1].item()
    
    # Create combined feature vector
    feature_vector = torch.tensor([
        left_blocked, 
        front_blocked, 
        right_blocked, 
        ball_exists, 
        ball_x, 
        ball_y
    ])
    
    return feature_vector