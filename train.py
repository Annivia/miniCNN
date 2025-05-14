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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define paths
TRAIN_CSV = "highway_data/train.csv"
TEST_CSV = "highway_data/test.csv"
TRAIN_IMG_DIR = "/home/xzhang3205/full_dataset/highway_train"
TEST_IMG_DIR = "/home/xzhang3205/full_dataset/highway_test"
LOG_DIR = "logs/highway"

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define custom dataset
class HighwayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Read CSV with header (since it looks like the CSV has a header row)
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Skip the first row if it's a header with column names
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
            # Create a blank image as fallback
            image = torch.zeros((3, 224, 224), dtype=torch.float32)
        
        # Binary targets - lane existence (columns 1 and 2)
        try:
            lower_lane_exists = float(self.data.iloc[idx, 1])
            upper_lane_exists = float(self.data.iloc[idx, 2])
            
            binary_targets = torch.tensor([lower_lane_exists, upper_lane_exists], 
                                          dtype=torch.float32)
        except Exception as e:
            print(f"Error processing binary targets for index {idx}: {e}")
            print(f"Data: {self.data.iloc[idx, 1]}, {self.data.iloc[idx, 2]}")
            # Fallback values
            binary_targets = torch.zeros(2, dtype=torch.float32)
        
        # Continuous targets - positions (columns 5, 6, 7)
        try:
            vehicle_ahead = float(self.data.iloc[idx, 5])  # vehicle_ahead_same_lane_0_x_position
            agent_x = float(self.data.iloc[idx, 6])        # agent_0_x_position
            agent_y = float(self.data.iloc[idx, 7])        # agent_0_y_position
            
            continuous_targets = torch.tensor([vehicle_ahead, agent_x, agent_y], 
                                               dtype=torch.float32)
        except Exception as e:
            print(f"Error processing continuous targets for index {idx}: {e}")
            # Fallback values
            continuous_targets = torch.zeros(3, dtype=torch.float32)
        
        return image, binary_targets, continuous_targets

# Define CNN model
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

# Define image transformations with error handling
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets with error handling - proper handling of CSV format
try:
    # First, let's check the structure of the CSV files
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    print(f"Training CSV shape: {train_df.shape}")
    print(f"Testing CSV shape: {test_df.shape}")
    print(f"Training CSV columns: {train_df.columns.tolist()}")
    
    # Now create the datasets
    train_dataset = HighwayDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform=transform)
    test_dataset = HighwayDataset(TEST_CSV, TEST_IMG_DIR, transform=transform)
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
except Exception as e:
    print(f"Error creating datasets: {e}")
    import traceback
    traceback.print_exc()

# Create data loaders with error handling and fewer workers to diagnose issues
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# Initialize model, loss functions, and optimizer
model = HighwayCNN().to(device)
print(model)

binary_criterion = nn.BCELoss()
continuous_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Function to evaluate model
def evaluate(model, data_loader, device):
    model.eval()
    binary_total_loss = 0.0
    continuous_total_loss = 0.0
    all_binary_preds = []
    all_binary_targets = []
    continuous_errors = {
        'vehicle_ahead': [],
        'agent_x': [],
        'agent_y': []
    }
    
    with torch.no_grad():
        for images, binary_targets, continuous_targets in data_loader:
            images = images.to(device)
            binary_targets = binary_targets.to(device)
            continuous_targets = continuous_targets.to(device)
            
            # Forward pass
            binary_outputs, continuous_outputs = model(images)
            
            # Calculate binary loss
            binary_loss = binary_criterion(binary_outputs, binary_targets)
            binary_total_loss += binary_loss.item() * images.size(0)
            
            # Calculate continuous loss, ignoring -1.0 values
            mask = continuous_targets != -1.0
            if mask.sum() > 0:
                continuous_loss = continuous_criterion(
                    continuous_outputs[mask], 
                    continuous_targets[mask]
                )
                continuous_total_loss += continuous_loss.item() * images.size(0)
            
            # Convert binary outputs to predictions (threshold = 0.5)
            binary_preds = (binary_outputs > 0.5).float()
            all_binary_preds.extend(binary_preds.cpu().numpy())
            all_binary_targets.extend(binary_targets.cpu().numpy())
            
            # Calculate errors for continuous predictions
            # Only compute errors where target is not -1.0
            for i in range(continuous_targets.size(0)):
                # For vehicle ahead position
                if continuous_targets[i, 0] != -1.0:
                    continuous_errors['vehicle_ahead'].append(
                        abs(continuous_outputs[i, 0].item() - continuous_targets[i, 0].item())
                    )
                
                # Agent x position (should always be valid)
                continuous_errors['agent_x'].append(
                    abs(continuous_outputs[i, 1].item() - continuous_targets[i, 1].item())
                )
                
                # Agent y position (should always be valid)
                continuous_errors['agent_y'].append(
                    abs(continuous_outputs[i, 2].item() - continuous_targets[i, 2].item())
                )
    
    # Calculate average losses
    binary_avg_loss = binary_total_loss / len(data_loader.dataset)
    continuous_avg_loss = continuous_total_loss / len(data_loader.dataset)
    
    # Calculate total loss (used for model selection)
    val_total_loss = binary_avg_loss + continuous_avg_loss
    
    # Calculate accuracy for binary predictions
    accuracy = accuracy_score(np.array(all_binary_targets).flatten(), 
                              np.array(all_binary_preds).flatten())
    
    # Calculate mean and std of continuous errors
    continuous_stats = {
        'vehicle_ahead': {
            'mean': np.mean(continuous_errors['vehicle_ahead']) if continuous_errors['vehicle_ahead'] else 0,
            'std': np.std(continuous_errors['vehicle_ahead']) if continuous_errors['vehicle_ahead'] else 0
        },
        'agent_x': {
            'mean': np.mean(continuous_errors['agent_x']),
            'std': np.std(continuous_errors['agent_x'])
        },
        'agent_y': {
            'mean': np.mean(continuous_errors['agent_y']),
            'std': np.std(continuous_errors['agent_y'])
        }
    }
    
    return binary_avg_loss, continuous_avg_loss, val_total_loss, accuracy, continuous_stats

# Training function
def train_model(model, train_loader, test_loader, num_epochs=100):
    # Initialize lists to store metrics
    train_binary_losses = []
    train_continuous_losses = []
    val_binary_losses = []
    val_continuous_losses = []
    val_total_losses = []
    val_accuracies = []
    
    # For tracking best model - specifically using validation loss as the criterion
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Initialize training log file
    log_file = os.path.join(LOG_DIR, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write("Epoch,Train_Binary_Loss,Train_Continuous_Loss,Val_Binary_Loss,Val_Continuous_Loss,Val_Total_Loss,Val_Accuracy,Vehicle_Ahead_Error_Mean,Vehicle_Ahead_Error_Std,Agent_X_Error_Mean,Agent_X_Error_Std,Agent_Y_Error_Mean,Agent_Y_Error_Std\n")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        binary_running_loss = 0.0
        continuous_running_loss = 0.0
        
        # Use tqdm for the training loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, binary_targets, continuous_targets) in enumerate(progress_bar):
            images = images.to(device)
            binary_targets = binary_targets.to(device)
            continuous_targets = continuous_targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            binary_outputs, continuous_outputs = model(images)
            
            # Calculate binary loss
            binary_loss = binary_criterion(binary_outputs, binary_targets)
            
            # Calculate continuous loss, ignoring -1.0 values
            mask = continuous_targets != -1.0
            if mask.sum() > 0:
                continuous_loss = continuous_criterion(
                    continuous_outputs[mask], 
                    continuous_targets[mask]
                )
            else:
                continuous_loss = torch.tensor(0.0, device=device)
            
            # Combined loss (can adjust weights if needed)
            loss = binary_loss + continuous_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running losses
            binary_running_loss += binary_loss.item()
            continuous_running_loss += continuous_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'binary_loss': binary_loss.item(),
                'continuous_loss': continuous_loss.item()
            })
            
            # Print batch info for debugging early batches
            if epoch == 0 and batch_idx < 3:
                print(f"\nBatch {batch_idx}: Binary Loss: {binary_loss.item():.4f}, Continuous Loss: {continuous_loss.item():.4f}")
                print(f"Binary targets shape: {binary_targets.shape}, Continuous targets shape: {continuous_targets.shape}")
        
        # Calculate average training losses
        avg_binary_loss = binary_running_loss / len(train_loader)
        avg_continuous_loss = continuous_running_loss / len(train_loader)
        
        # Evaluate on test set
        val_binary_loss, val_continuous_loss, val_total_loss, val_accuracy, continuous_stats = evaluate(model, test_loader, device)
        
        # Update learning rate based on validation loss
        scheduler.step(val_total_loss)
        
        # Store metrics
        train_binary_losses.append(avg_binary_loss)
        train_continuous_losses.append(avg_continuous_loss)
        val_binary_losses.append(val_binary_loss)
        val_continuous_losses.append(val_continuous_loss)
        val_total_losses.append(val_total_loss)
        val_accuracies.append(val_accuracy)
        
        # Check if this is the best model based on validation loss
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_epoch = epoch
            
            # Save the best model to the specified directory
            model_save_path = os.path.join(LOG_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_binary_loss': val_binary_loss,
                'val_continuous_loss': val_continuous_loss,
                'val_total_loss': val_total_loss,
                'val_accuracy': val_accuracy,
                'continuous_stats': continuous_stats
            }, model_save_path)
            print(f"New best model saved at epoch {epoch+1} with validation loss: {val_total_loss:.4f}!")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(LOG_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        # Save metrics to log file
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{avg_binary_loss},{avg_continuous_loss},{val_binary_loss},{val_continuous_loss},{val_total_loss},{val_accuracy},"
                    f"{continuous_stats['vehicle_ahead']['mean']},{continuous_stats['vehicle_ahead']['std']},"
                    f"{continuous_stats['agent_x']['mean']},{continuous_stats['agent_x']['std']},"
                    f"{continuous_stats['agent_y']['mean']},{continuous_stats['agent_y']['std']}\n")
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Train - Binary Loss: {avg_binary_loss:.4f}, Continuous Loss: {avg_continuous_loss:.4f}")
        print(f"Test - Binary Loss: {val_binary_loss:.4f}, Continuous Loss: {val_continuous_loss:.4f}, Total Loss: {val_total_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"Position Error Stats:")
        print(f"  Vehicle Ahead - Mean: {continuous_stats['vehicle_ahead']['mean']:.4f}, Std: {continuous_stats['vehicle_ahead']['std']:.4f}")
        print(f"  Agent X - Mean: {continuous_stats['agent_x']['mean']:.4f}, Std: {continuous_stats['agent_x']['std']:.4f}")
        print(f"  Agent Y - Mean: {continuous_stats['agent_y']['mean']:.4f}, Std: {continuous_stats['agent_y']['std']:.4f}")
    
    # Save final model
    final_model_path = os.path.join(LOG_DIR, 'final_model.pth')
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, final_model_path)
    print(f"Final model saved after {num_epochs} epochs")
    
    # Save training history
    history_path = os.path.join(LOG_DIR, 'training_history.npy')
    np.save(history_path, {
        'train_binary_losses': train_binary_losses,
        'train_continuous_losses': train_continuous_losses,
        'val_binary_losses': val_binary_losses,
        'val_continuous_losses': val_continuous_losses,
        'val_total_losses': val_total_losses,
        'val_accuracies': val_accuracies,
        'best_epoch': best_epoch
    })
    
    # Return training history and best epoch
    return {
        'train_binary_losses': train_binary_losses,
        'train_continuous_losses': train_continuous_losses,
        'val_binary_losses': val_binary_losses,
        'val_continuous_losses': val_continuous_losses,
        'val_total_losses': val_total_losses,
        'val_accuracies': val_accuracies,
        'best_epoch': best_epoch
    }

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(15, 15))
    
    # Plot losses
    plt.subplot(3, 1, 1)
    plt.plot(history['train_binary_losses'], label='Train Binary Loss')
    plt.plot(history['val_binary_losses'], label='Val Binary Loss')
    plt.axvline(x=history['best_epoch'], color='r', linestyle='--', label='Best Model')
    plt.title('Binary Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot continuous losses
    plt.subplot(3, 1, 2)
    plt.plot(history['train_continuous_losses'], label='Train Continuous Loss')
    plt.plot(history['val_continuous_losses'], label='Val Continuous Loss')
    plt.axvline(x=history['best_epoch'], color='r', linestyle='--', label='Best Model')
    plt.title('Continuous Regression Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(3, 1, 3)
    plt.plot(history['val_accuracies'], label='Validation Accuracy', color='green')
    plt.axvline(x=history['best_epoch'], color='r', linestyle='--', label='Best Model')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'training_history.png'))
    plt.close()

# Test model and visualize predictions
def visualize_predictions(model, test_loader, num_samples=5):
    model.eval()
    images_shown = 0
    
    plt.figure(figsize=(15, 12))
    
    with torch.no_grad():
        for images, binary_targets, continuous_targets in test_loader:
            if images_shown >= num_samples:
                break
                
            images = images.to(device)
            binary_outputs, continuous_outputs = model(images)
            
            binary_preds = (binary_outputs > 0.5).float().cpu().numpy()
            binary_targets = binary_targets.cpu().numpy()
            continuous_outputs = continuous_outputs.cpu().numpy()
            continuous_targets = continuous_targets.cpu().numpy()
            
            # Denormalize images for visualization
            denorm = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
            
            for i in range(min(images.size(0), num_samples - images_shown)):
                img = denorm(images[i].cpu()).permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                
                plt.subplot(num_samples, 2, 2*images_shown + 1)
                plt.imshow(img)
                plt.title("Input Image")
                plt.axis('off')
                
                plt.subplot(num_samples, 2, 2*images_shown + 2)
                plt.imshow(img)
                plt.title("Predictions")
                
                # Display predictions
                pred_text = f"Lower Lane: {binary_preds[i][0]:.0f} (True: {binary_targets[i][0]:.0f})\n"
                pred_text += f"Upper Lane: {binary_preds[i][1]:.0f} (True: {binary_targets[i][1]:.0f})\n"
                pred_text += f"Vehicle Ahead: {continuous_outputs[i][0]:.3f} (True: {continuous_targets[i][0]:.3f})\n"
                pred_text += f"Agent X: {continuous_outputs[i][1]:.3f} (True: {continuous_targets[i][1]:.3f})\n"
                pred_text += f"Agent Y: {continuous_outputs[i][2]:.3f} (True: {continuous_targets[i][2]:.3f})"
                
                plt.xlabel(pred_text, fontsize=9)
                plt.axis('off')
                
                images_shown += 1
                if images_shown >= num_samples:
                    break
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'prediction_visualization.png'))
    plt.close()

# Debug function to help identify potential issues
def debug_dataset(dataset, num_samples=5):
    print("Debugging dataset...")
    for i in range(min(num_samples, len(dataset))):
        try:
            image, binary_targets, continuous_targets = dataset[i]
            print(f"Sample {i}:")
            print(f"  Image shape: {image.shape}")
            print(f"  Binary targets: {binary_targets}")
            print(f"  Continuous targets: {continuous_targets}")
        except Exception as e:
            print(f"Error processing sample {i}: {e}")

# Main execution
if __name__ == "__main__":
    # Debug dataset to identify issues
    print("Debugging train dataset:")
    debug_dataset(train_dataset)
    
    print("Debugging test dataset:")
    debug_dataset(test_dataset)
    
    # Print dataset sizes
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Print model summary
    print(model)
    
    # Train model
    try:
        history = train_model(model, train_loader, test_loader, num_epochs=100)
        
        # Plot training history
        plot_training_history(history)
        
        # Load best model for evaluation
        best_model_path = os.path.join(LOG_DIR, 'best_model.pth')
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
        
        # Evaluate final model
        val_binary_loss, val_continuous_loss, val_total_loss, val_accuracy, continuous_stats = evaluate(model, test_loader, device)
        print("\nFinal Evaluation Results (Best Model):")
        print(f"Binary Loss: {val_binary_loss:.4f}")
        print(f"Continuous Loss: {val_continuous_loss:.4f}")
        print(f"Total Loss: {val_total_loss:.4f}")
        print(f"Binary Accuracy: {val_accuracy:.4f}")
        print(f"Position Error Stats:")
        print(f"  Vehicle Ahead - Mean: {continuous_stats['vehicle_ahead']['mean']:.4f}, Std: {continuous_stats['vehicle_ahead']['std']:.4f}")
        print(f"  Agent X - Mean: {continuous_stats['agent_x']['mean']:.4f}, Std: {continuous_stats['agent_x']['std']:.4f}")
        print(f"  Agent Y - Mean: {continuous_stats['agent_y']['mean']:.4f}, Std: {continuous_stats['agent_y']['std']:.4f}")
        
        # Save final evaluation results
        with open(os.path.join(LOG_DIR, 'final_evaluation.txt'), 'w') as f:
            f.write(f"Best model from epoch: {checkpoint['epoch']+1}\n")
            f.write(f"Binary Loss: {val_binary_loss:.4f}\n")
            f.write(f"Continuous Loss: {val_continuous_loss:.4f}\n")
            f.write(f"Total Loss: {val_total_loss:.4f}\n")
            f.write(f"Binary Accuracy: {val_accuracy:.4f}\n")
            f.write(f"Position Error Stats:\n")
            f.write(f"  Vehicle Ahead - Mean: {continuous_stats['vehicle_ahead']['mean']:.4f}, Std: {continuous_stats['vehicle_ahead']['std']:.4f}\n")
            f.write(f"  Agent X - Mean: {continuous_stats['agent_x']['mean']:.4f}, Std: {continuous_stats['agent_x']['std']:.4f}\n")
            f.write(f"  Agent Y - Mean: {continuous_stats['agent_y']['mean']:.4f}, Std: {continuous_stats['agent_y']['std']:.4f}\n")
        
        # Visualize some predictions
        visualize_predictions(model, test_loader)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()