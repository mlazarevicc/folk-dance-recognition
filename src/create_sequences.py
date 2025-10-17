import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json

# --- Adjustable parameters ---
SEQUENCE_LENGTH = 30
STEP = 2
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

def create_sequences_for_videos(df, video_ids, sequence_length, step):
    """Creates sequences only for the given list of video IDs."""
    
    df_subset = df[df['video_id'].isin(video_ids)]
    
    sequences = []
    labels = []
    
    for _, group in df_subset.groupby('video_id'):
        feature_data = group.drop(columns=['video_id', 'class', 'class_encoded']).values
        label = group['class_encoded'].iloc[0]
        
        num_frames = len(feature_data)
        for i in range(0, num_frames - sequence_length + 1, step):
            sequences.append(feature_data[i:i + sequence_length])
            labels.append(label)
            
    return np.array(sequences), np.array(labels)

def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_data_folder = project_root / 'processed_data'
    input_csv = processed_data_folder / 'landmarks_augmented.csv'
    
    print("Loading data...")
    df = pd.read_csv(input_csv)
    
    # 1. Converting labels to numbers
    print("Label coding...")
    label_encoder = LabelEncoder()
    df['class_encoded'] = label_encoder.fit_transform(df['class'])
    
    label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(processed_data_folder / 'label_map.json', 'w') as f:
        json.dump(label_map, f, indent=4)
    print(f"Label map saved: {label_map}")

    # 2. We split the data by original video IDs
    print("Video-based data sharing...")
    
    # We remove the '_aug_X' suffix to get the original IDs
    original_video_ids = df['video_id'].str.replace(r'_aug_\d+', '', regex=True).unique()
    labels_for_videos = [df[df['video_id'].str.startswith(vid)]['class'].iloc[0] for vid in original_video_ids]

    # We share a list of original video IDs
    train_val_ids, test_ids = train_test_split(
        original_video_ids, test_size=TEST_SIZE, stratify=labels_for_videos, random_state=42
    )
    
    # We create a new stratification for the validation set
    labels_for_train_val = [df[df['video_id'].str.startswith(vid)]['class'].iloc[0] for vid in train_val_ids]
    relative_val_size = VALIDATION_SIZE / (1 - TEST_SIZE)
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=relative_val_size, stratify=labels_for_train_val, random_state=42
    )
    
    print(f"Number of original training videos: {len(train_ids)}")
    print(f"Number of original videos to validate: {len(val_ids)}")
    print(f"Number of original videos for the test: {len(test_ids)}")

    # 3. Creating sequences for each set specificaly
    print("\nCreating sequences...")
    
    # We take all video IDs (original + augmented) belonging to each set
    all_video_ids = df['video_id'].unique()
    
    train_video_ids_full = [vid for vid in all_video_ids if any(vid.startswith(orig_id) for orig_id in train_ids)]
    val_video_ids_full = [vid for vid in all_video_ids if any(vid.startswith(orig_id) for orig_id in val_ids)]
    test_video_ids_full = [vid for vid in all_video_ids if any(vid.startswith(orig_id) for orig_id in test_ids)]
    
    X_train, y_train = create_sequences_for_videos(df, train_video_ids_full, SEQUENCE_LENGTH, STEP)
    X_val, y_val = create_sequences_for_videos(df, val_video_ids_full, SEQUENCE_LENGTH, STEP)
    X_test, y_test = create_sequences_for_videos(df, test_video_ids_full, SEQUENCE_LENGTH, STEP)
    
    print(f"Size of training set (sequence): {len(X_train)}") 
    print(f"Size of validation set (sequence): {len(X_val)}") 
    print(f"Size of test set (sequence): {len(X_test)}")
    
    # 4. Saving data
    output_npz = processed_data_folder / 'sequences.npz'
    print(f"Saving data in {output_npz}...")
    np.savez_compressed(
        output_npz,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    
    print("\n ✅ Creation of sequences and splits completed successfully.")
    
if __name__ == '__main__':
    main()