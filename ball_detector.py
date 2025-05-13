train_data = "redball_data/train_filtered.csv"
test_data = "redball_data/test_filtered.csv"
train_image_folder = "redball_images/redball_train_filtered"
test_image_folder = "redball_images/redball_test_filtered"

## Data format like this
## image_3.png,0,0,0,1.0,1.015625,1
## image_path(inside the image folder), leftblock, frontblock, rightblock, ball_x, ball_y, ballexists)

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define paths
train_data_path = "redball_data/train_filtered.csv"
test_data_path = "redball_data/test_filtered.csv"
train_image_folder = "redball_images/redball_train_filtered"
test_image_folder = "redball_images/redball_test_filtered"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Custom dataset class
class RedBallDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # Get ball coordinates (x, y)
        ball_x = float(self.data.iloc[idx, 4])
        ball_y = float(self.data.iloc[idx, 5])
        coordinates = torch.tensor([ball_x, ball_y], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, coordinates

# Define CNN model
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

# Data transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to standard size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = RedBallDataset(
    csv_file=train_data_path,
    img_dir=train_image_folder,
    transform=transform
)

test_dataset = RedBallDataset(
    csv_file=test_data_path,
    img_dir=test_image_folder,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = BallDetectionCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    history = {'train_loss': [], 'val_loss': [], 'best_val_loss': float('inf')}
    best_model_path = os.path.join(log_dir, "ball_detection.pth")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        
        for images, coordinates in train_loader:
            images = images.to(device)
            coordinates = coordinates.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, coordinates)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        
        # Validate the model
        val_loss = validate_model(model, test_loader, criterion)
        history['val_loss'].append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save the best model
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved to {best_model_path}")
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")
    
    return history

# Validation function
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for images, coordinates in val_loader:
            images = images.to(device)
            coordinates = coordinates.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, coordinates)
            
            val_loss += loss.item() * images.size(0)
    
    return val_loss / len(val_loader.dataset)

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for images, coordinates in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Move predictions and ground truth to CPU for evaluation
            pred_coords = outputs.cpu().numpy()
            true_coords = coordinates.numpy()
            
            predictions.append(pred_coords)
            ground_truth.append(true_coords)
    
    # Concatenate all batches
    predictions = np.vstack(predictions)
    ground_truth = np.vstack(ground_truth)
    
    # Calculate MSE for x and y coordinates
    mse_x = mean_squared_error(ground_truth[:, 0], predictions[:, 0])
    mse_y = mean_squared_error(ground_truth[:, 1], predictions[:, 1])
    mse_total = mean_squared_error(ground_truth, predictions)
    
    print(f"MSE X: {mse_x:.6f}")
    print(f"MSE Y: {mse_y:.6f}")
    print(f"Total MSE: {mse_total:.6f}")
    
    # Calculate Euclidean distance error
    euclidean_dist = np.sqrt(np.sum((predictions - ground_truth)**2, axis=1))
    mean_dist = np.mean(euclidean_dist)
    median_dist = np.median(euclidean_dist)
    
    print(f"Mean Euclidean Distance: {mean_dist:.6f}")
    print(f"Median Euclidean Distance: {median_dist:.6f}")
    
    return predictions, ground_truth, euclidean_dist

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'training_history.png'))
    plt.close()

# Visualize predictions
def visualize_predictions(test_loader, model, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    
    # Get a batch from the test loader
    dataiter = iter(test_loader)
    images, coordinates = next(dataiter)
    
    with torch.no_grad():
        for i in range(min(num_samples, len(images))):
            # Get the image and move to device
            image = images[i:i+1].to(device)
            true_coord = coordinates[i].numpy()
            
            # Make prediction
            pred_coord = model(image).cpu().numpy()[0]
            
            # Convert normalized tensor to image for display
            img_np = images[i].permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            # Display original image with true coordinates
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"True: ({true_coord[0]:.3f}, {true_coord[1]:.3f})")
            axes[i, 0].scatter(true_coord[0] * 64, true_coord[1] * 64, c='g', marker='o')
            axes[i, 0].axis('off')
            
            # Display original image with predicted coordinates
            axes[i, 1].imshow(img_np)
            axes[i, 1].set_title(f"Pred: ({pred_coord[0]:.3f}, {pred_coord[1]:.3f})")
            axes[i, 1].scatter(pred_coord[0] * 64, pred_coord[1] * 64, c='r', marker='x')
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'prediction_visualization.png'))
    plt.close()

# Main function to run training and evaluation
def main():
    # Train the model
    print("Starting training...")
    history = train_model(model, train_loader, criterion, optimizer, num_epochs=50)
    
    # Plot training history
    plot_training_history(history)
    
    # Load the best model
    best_model_path = os.path.join(log_dir, "ball_detection.pth")
    model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions, ground_truth, euclidean_dist = evaluate_model(model, test_loader)
    
    # Visualize some predictions
    visualize_predictions(test_loader, model)
    
    print(f"\nTraining and evaluation complete. Best model saved to {best_model_path}")
    
    # Return the model and evaluation metrics
    return model, history, (predictions, ground_truth, euclidean_dist)

if __name__ == "__main__":
    main()