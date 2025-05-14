import pandas as pd
import numpy as np
import os

# Define offsets for None values (easily adjustable)
LOWER_LANE_X_OFFSET = -1.0  # Offset for vehicle_adjacent_lower_lane_0_x_position
UPPER_LANE_X_OFFSET = -1.0  # Offset for vehicle_adjacent_upper_lane_0_x_position
SAME_LANE_X_OFFSET = -1.0   # Offset for vehicle_ahead_same_lane_0_x_position

def process_csv(input_path, output_path):
    print(f"Processing {input_path}...")
    
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Get initial row count
    initial_row_count = len(df)
    print(f"Initial row count: {initial_row_count}")
    
    # Remove rows where agent positions are None
    df = df.dropna(subset=['agent_0_x_position', 'agent_0_y_position'])
    
    # Count deleted rows
    deleted_rows = initial_row_count - len(df)
    print(f"Deleted {deleted_rows} rows with None agent positions")
    
    # Validate lane existence columns (must be True or False)
    for col in ['lane_existence_lower_lane_exists', 'lane_existence_upper_lane_exists']:
        # Check if any non-NaN values are not True or False
        invalid_values = df[col].dropna().apply(lambda x: x not in [True, False])
        if invalid_values.any():
            raise ValueError(f"Column {col} contains values other than True or False")
    
    # Convert True/False to 1/0 for lane existence columns
    df['lane_existence_lower_lane_exists'] = df['lane_existence_lower_lane_exists'].map({True: 1, False: 0})
    df['lane_existence_upper_lane_exists'] = df['lane_existence_upper_lane_exists'].map({True: 1, False: 0})
    
    # Fill NaN values with appropriate offsets
    df['vehicle_adjacent_lower_lane_0_x_position'] = df['vehicle_adjacent_lower_lane_0_x_position'].fillna(LOWER_LANE_X_OFFSET)
    df['vehicle_adjacent_upper_lane_0_x_position'] = df['vehicle_adjacent_upper_lane_0_x_position'].fillna(UPPER_LANE_X_OFFSET)
    df['vehicle_ahead_same_lane_0_x_position'] = df['vehicle_ahead_same_lane_0_x_position'].fillna(SAME_LANE_X_OFFSET)
    
    # Reorder columns
    column_order = [
        'file_name',
        'lane_existence_lower_lane_exists',
        'lane_existence_upper_lane_exists',
        'vehicle_adjacent_lower_lane_0_x_position',
        'vehicle_adjacent_upper_lane_0_x_position',
        'vehicle_ahead_same_lane_0_x_position',
        'agent_0_x_position',
        'agent_0_y_position'
    ]
    
    df = df[column_order]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed data to new CSV
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Final row count: {len(df)}")
    
    return deleted_rows

def main():
    # Process training data
    train_deleted = process_csv('highway_data/highway_train_single_frame.csv', 'highway_data/train.csv')
    
    # Process test data
    test_deleted = process_csv('highway_data/highway_test_single_frame.csv', 'highway_data/test.csv')
    
    # Total deleted rows
    total_deleted = train_deleted + test_deleted
    print(f"\nTotal rows deleted: {total_deleted}")

if __name__ == "__main__":
    main()