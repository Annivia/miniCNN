import pandas as pd
import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from load import feature_extraction

def evaluate_model_performance(test_csv_path, image_folder):
    """
    Evaluate the performance of feature extraction on test dataset
    
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
        'lane_existence_lower_lane_exists': {'correct': 0, 'total': 0},
        'lane_existence_upper_lane_exists': {'correct': 0, 'total': 0},
        'vehicle_adjacent_lower_lane_0_x_position': {'mse': [], 'mae': []},
        'vehicle_adjacent_upper_lane_0_x_position': {'mse': [], 'mae': []},
        'vehicle_ahead_same_lane_0_x_position': {'mse': [], 'mae': []},
        'agent_0_x_position': {'mse': [], 'mae': []},
        'agent_0_y_position': {'mse': [], 'mae': []}
    }
    
    # Process each image in the test set
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        # Get the image path
        img_name = row['file_name']
        img_path = os.path.join(image_folder, img_name)
        
        # Get ground truth values
        gt_lower_lane_exists = bool(row['lane_existence_lower_lane_exists'])
        gt_upper_lane_exists = bool(row['lane_existence_upper_lane_exists'])
        gt_vehicle_adjacent_lower_lane_0_x_position = row['vehicle_adjacent_lower_lane_0_x_position']
        gt_vehicle_adjacent_upper_lane_0_x_position = row['vehicle_adjacent_upper_lane_0_x_position']
        gt_vehicle_ahead_same_lane_0_x_position = row['vehicle_ahead_same_lane_0_x_position']
        gt_agent_0_x_position = row['agent_0_x_position']
        gt_agent_0_y_position = row['agent_0_y_position']
        
        try:
            # Extract features
            features = feature_extraction(img_path)
            
            # The ordering in the feature vector based on your extraction function:
            # [0] - lane_existence_lower_lane_exists
            # [1] - lane_existence_upper_lane_exists
            # [2] - agent_0_x_position
            # [3] - agent_0_y_position
            # [4] - vehicle_adjacent_lower_lane_0_x_position (flattened)
            # [5] - vehicle_adjacent_upper_lane_0_x_position (flattened)
            # [6] - vehicle_ahead_same_lane_0_x_position (flattened)
            
            # Ensure we're working with a numpy array
            feature_np = features.cpu().numpy()
            
            # Binary classification for lane existence
            pred_lower_lane_exists = feature_np[0] > 0.5
            pred_upper_lane_exists = feature_np[1] > 0.5
            
            # Update lane existence metrics
            metrics['lane_existence_lower_lane_exists']['total'] += 1
            if pred_lower_lane_exists == gt_lower_lane_exists:
                metrics['lane_existence_lower_lane_exists']['correct'] += 1
                
            metrics['lane_existence_upper_lane_exists']['total'] += 1
            if pred_upper_lane_exists == gt_upper_lane_exists:
                metrics['lane_existence_upper_lane_exists']['correct'] += 1
                
            # Only evaluate position metrics if they're not -1 (absent) in ground truth
            # and if the corresponding lane exists
            
            # Agent positions (always evaluate these)
            if gt_agent_0_x_position != -1:
                mse = (feature_np[2] - gt_agent_0_x_position) ** 2
                mae = abs(feature_np[2] - gt_agent_0_x_position)
                metrics['agent_0_x_position']['mse'].append(mse)
                metrics['agent_0_x_position']['mae'].append(mae)
                
            if gt_agent_0_y_position != -1:
                mse = (feature_np[3] - gt_agent_0_y_position) ** 2
                mae = abs(feature_np[3] - gt_agent_0_y_position)
                metrics['agent_0_y_position']['mse'].append(mse)
                metrics['agent_0_y_position']['mae'].append(mae)
                
            # Vehicle in lower lane - only if it exists and lane exists
            if gt_vehicle_adjacent_lower_lane_0_x_position != -1 and gt_lower_lane_exists:
                mse = (feature_np[4] - gt_vehicle_adjacent_lower_lane_0_x_position) ** 2
                mae = abs(feature_np[4] - gt_vehicle_adjacent_lower_lane_0_x_position)
                metrics['vehicle_adjacent_lower_lane_0_x_position']['mse'].append(mse)
                metrics['vehicle_adjacent_lower_lane_0_x_position']['mae'].append(mae)
                
            # Vehicle in upper lane - only if it exists and lane exists
            if gt_vehicle_adjacent_upper_lane_0_x_position != -1 and gt_upper_lane_exists:
                mse = (feature_np[5] - gt_vehicle_adjacent_upper_lane_0_x_position) ** 2
                mae = abs(feature_np[5] - gt_vehicle_adjacent_upper_lane_0_x_position)
                metrics['vehicle_adjacent_upper_lane_0_x_position']['mse'].append(mse)
                metrics['vehicle_adjacent_upper_lane_0_x_position']['mae'].append(mae)
                
            # Vehicle ahead in same lane - only if it exists
            if gt_vehicle_ahead_same_lane_0_x_position != -1:
                mse = (feature_np[6] - gt_vehicle_ahead_same_lane_0_x_position) ** 2
                mae = abs(feature_np[6] - gt_vehicle_ahead_same_lane_0_x_position)
                metrics['vehicle_ahead_same_lane_0_x_position']['mse'].append(mse)
                metrics['vehicle_ahead_same_lane_0_x_position']['mae'].append(mae)
                
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    
    # Calculate final metrics
    results = {}
    
    # Accuracy for lane existence
    for lane_key in ['lane_existence_lower_lane_exists', 'lane_existence_upper_lane_exists']:
        if metrics[lane_key]['total'] > 0:
            accuracy = metrics[lane_key]['correct'] / metrics[lane_key]['total']
            results[f"{lane_key}_accuracy"] = accuracy
    
    # MSE and MAE for position metrics
    for pos_key in ['vehicle_adjacent_lower_lane_0_x_position', 
                   'vehicle_adjacent_upper_lane_0_x_position',
                   'vehicle_ahead_same_lane_0_x_position',
                   'agent_0_x_position', 
                   'agent_0_y_position']:
        if len(metrics[pos_key]['mse']) > 0:
            results[f"{pos_key}_mse"] = np.mean(metrics[pos_key]['mse'])
            results[f"{pos_key}_mae"] = np.mean(metrics[pos_key]['mae'])
    
    return results

def main():
    test_csv_path = '/home/xzhang3205/miniCNN/highway_data/test.csv'
    image_folder = '/home/xzhang3205/full_dataset/highway_test'
    
    # Run evaluation
    results = evaluate_model_performance(test_csv_path, image_folder)
    
    # Print results
    print("\nEvaluation Results:")
    print("-------------------")
    
    # Print lane existence accuracy
    for key in sorted(results.keys()):
        if 'accuracy' in key:
            print(f"{key}: {results[key]:.4f}")
    
    # Print position metrics
    for key in sorted(results.keys()):
        if 'mse' in key:
            base_key = key.replace('_mse', '')
            print(f"{base_key}:")
            print(f"  MSE: {results[key]:.4f}")
            print(f"  MAE: {results[base_key + '_mae']:.4f}")

if __name__ == "__main__":
    main()