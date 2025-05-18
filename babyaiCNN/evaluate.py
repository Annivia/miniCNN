import pandas as pd
import torch
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from feature_extraction import feature_extraction

# Assuming BinaryDetectCNN and BallDetectionCNN classes are available
# and feature_extraction function is defined as provided

def evaluate_model_performance(test_csv_path, image_folder):
    """
    Evaluate the performance of feature extraction on redball test dataset
    
    Args:
        test_csv_path: Path to the test CSV file
        image_folder: Path to the folder containing test images
        
    Returns:
        Dictionary with performance metrics
    """
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    
    # Initialize metrics
    metrics = {
        'immediate_obstacle_0_is_blocked': {'correct': 0, 'total': 0},
        'immediate_obstacle_1_is_blocked': {'correct': 0, 'total': 0},
        'immediate_obstacle_2_is_blocked': {'correct': 0, 'total': 0},
        'red_ball_exists': {'correct': 0, 'total': 0},
        'red_ball_0_x_relative_position': {'mse': [], 'mae': []},
        'red_ball_0_y_relative_position': {'mse': [], 'mae': []}
    }
    
    # Process each image in the test set
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        # Get the image path
        img_name = row['file_name']
        img_path = os.path.join(image_folder, img_name)
        
        # Get ground truth values
        gt_left_blocked = bool(row['immediate_obstacle_0_is_blocked'])
        gt_front_blocked = bool(row['immediate_obstacle_1_is_blocked'])
        gt_right_blocked = bool(row['immediate_obstacle_2_is_blocked'])
        gt_ball_exists = bool(row['red_ball_exists'])
        gt_ball_x = row['red_ball_0_x_relative_position']
        gt_ball_y = row['red_ball_0_y_relative_position']
        
        try:
            # Extract features
            features = feature_extraction(img_path)
            
            # The ordering in the feature vector based on your extraction function:
            # [0] - left_blocked
            # [1] - front_blocked
            # [2] - right_blocked
            # [3] - ball_exists
            # [4] - ball_x
            # [5] - ball_y
            
            # Convert to numpy for easier handling
            feature_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else np.array(features)
            
            # Binary classification for obstacles and ball existence
            pred_left_blocked = feature_np[0] > 0.5
            pred_front_blocked = feature_np[1] > 0.5
            pred_right_blocked = feature_np[2] > 0.5
            pred_ball_exists = feature_np[3] > 0.5
            
            # Update obstacle metrics
            metrics['immediate_obstacle_0_is_blocked']['total'] += 1
            if pred_left_blocked == gt_left_blocked:
                metrics['immediate_obstacle_0_is_blocked']['correct'] += 1
                
            metrics['immediate_obstacle_1_is_blocked']['total'] += 1
            if pred_front_blocked == gt_front_blocked:
                metrics['immediate_obstacle_1_is_blocked']['correct'] += 1
                
            metrics['immediate_obstacle_2_is_blocked']['total'] += 1
            if pred_right_blocked == gt_right_blocked:
                metrics['immediate_obstacle_2_is_blocked']['correct'] += 1
            
            # Update ball exists metric
            metrics['red_ball_exists']['total'] += 1
            if pred_ball_exists == gt_ball_exists:
                metrics['red_ball_exists']['correct'] += 1
                
            # Only evaluate ball position metrics if the ball exists in ground truth
            if gt_ball_exists:
                # Ball X position
                mse = (feature_np[4] - gt_ball_x) ** 2
                mae = abs(feature_np[4] - gt_ball_x)
                metrics['red_ball_0_x_relative_position']['mse'].append(mse)
                metrics['red_ball_0_x_relative_position']['mae'].append(mae)
                
                # Ball Y position
                mse = (feature_np[5] - gt_ball_y) ** 2
                mae = abs(feature_np[5] - gt_ball_y)
                metrics['red_ball_0_y_relative_position']['mse'].append(mse)
                metrics['red_ball_0_y_relative_position']['mae'].append(mae)
                
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    
    # Calculate final metrics
    results = {}
    
    # Accuracy for binary classifications
    for binary_key in ['immediate_obstacle_0_is_blocked', 
                       'immediate_obstacle_1_is_blocked', 
                       'immediate_obstacle_2_is_blocked',
                       'red_ball_exists']:
        if metrics[binary_key]['total'] > 0:
            accuracy = metrics[binary_key]['correct'] / metrics[binary_key]['total']
            results[f"{binary_key}_accuracy"] = accuracy
    
    # MSE and MAE for position metrics
    for pos_key in ['red_ball_0_x_relative_position', 'red_ball_0_y_relative_position']:
        if len(metrics[pos_key]['mse']) > 0:
            results[f"{pos_key}_mse"] = np.mean(metrics[pos_key]['mse'])
            results[f"{pos_key}_mae"] = np.mean(metrics[pos_key]['mae'])
    
    return results

def main():
    test_csv_path = '/home/xzhang3205/miniCNN/redball_data/test.csv'
    image_folder = '/home/xzhang3205/full_dataset/redball_test'
    
    # Run evaluation
    results = evaluate_model_performance(test_csv_path, image_folder)
    
    # Print results
    print("\nEvaluation Results:")
    print("-------------------")
    
    # Print binary classification accuracy
    for key in sorted(results.keys()):
        if 'accuracy' in key:
            print(f"{key}: {results[key]:.4f}")
    
    # Print position metrics (only if ball exists)
    for key in sorted(results.keys()):
        if 'mse' in key:
            base_key = key.replace('_mse', '')
            print(f"{base_key}:")
            print(f"  MSE: {results[key]:.4f}")
            print(f"  MAE: {results[base_key + '_mae']:.4f}")

if __name__ == "__main__":
    main()