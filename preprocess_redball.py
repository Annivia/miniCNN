import pandas as pd
import os
import sys

# def process_csv(file_path, output_path):
#     # Load CSV
#     df = pd.read_csv(file_path)

#     # Drop direction columns
#     direction_cols = [col for col in df.columns if 'direction' in col]
#     df.drop(columns=direction_cols, inplace=True)

#     # Convert boolean-like strings to 0/1
#     for col in df.columns:
#         if 'is_blocked' in col:
#             df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0})

#     df.to_csv(output_path, index=False)
#     print(f"Saved processed file to {output_path}")

def process_csv(file_path, output_path):
    # Load CSV
    df = pd.read_csv(file_path)

    # Drop direction columns
    direction_cols = [col for col in df.columns if 'direction' in col]
    df.drop(columns=direction_cols, inplace=True)

    # Convert boolean-like strings to 0/1
    for col in df.columns:
        if 'is_blocked' in col:
            df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0})

    # Add red_ball_exists column
    red_ball_x = 'red_ball_0_x_relative_position'
    red_ball_y = 'red_ball_0_y_relative_position'

    # Check for presence: if both x and y are not NaN, then red ball exists
    df['red_ball_exists'] = (~df[red_ball_x].isna() & ~df[red_ball_y].isna()).astype(int)

    # Fill missing red ball positions if not present
    df[red_ball_x].fillna(0, inplace=True)
    df[red_ball_y].fillna(-5, inplace=True)

    # Save processed file
    df.to_csv(output_path, index=False)
    print(f"Saved processed file to {output_path}")

if __name__ == '__main__':
    file_path_train = "redball_data/redball_train_filtered_single_frame.csv"
    file_path_test = "redball_data/redball_test_filtered_single_frame.csv"

    process_csv(file_path_train, "redball_data/train.csv")
    process_csv(file_path_test, "redball_data/test.csv")

