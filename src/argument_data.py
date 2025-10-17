import pandas as pd
import numpy as np
from pathlib import Path

# --- Adjustable parameters ---
AUGMENTATION_FACTOR = 5       # How many times we want to increase the dataset for classes that have little data.
NOISE_LEVEL = 0.015           # How much "noise" are we adding. Smaller value = more subtle changes.
MIN_SAMPLES_THRESHOLD = 500   # Minimum number of frames per class below which augmentation will be performed

def augment_data(input_csv, output_csv):
    """ 
    Loads keypoint data, applies augmentation to classes 
    with little data and saves a new, enlarged dataset. 
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Extracting columns with coordinates
    feature_columns = [col for col in df.columns if '_x' in col or '_y' in col or '_z' in col]
    
    augmented_dfs = [df] # A list that will contain the original and augmented parts
    
    class_counts = df['class'].value_counts()
    print("\nNumber of frames per class (before augmentation):")
    print(class_counts)
    
    print("\nStarting augmentation...")
    for class_name, count in class_counts.items():
        if count < MIN_SAMPLES_THRESHOLD:
            print(f" -> Augmenting class '{class_name}' which has {count} frames...")
            
            # Only extract rows for the current class
            class_df = df[df['class'] == class_name].copy()
            
            # Create new, augmented rows
            for i in range(AUGMENTATION_FACTOR - 1):
                augmented_class_df = class_df.copy()
                
                # Create "noise" and add noise to the original coordinates
                noise = np.random.normal(0, NOISE_LEVEL, size=(len(class_df), len(feature_columns)))
                augmented_class_df[feature_columns] += noise
                
                # Create a new, unique video_id for each augmentation
                augmented_class_df['video_id'] = augmented_class_df['video_id'] + f'_aug_{i+1}'
                
                augmented_dfs.append(augmented_class_df)
        else:
            print(f" -> Skipping class '{class_name}' (enough data: {count})")
            
    print("\nMerging original and augmented data...")
    final_df = pd.concat(augmented_dfs, ignore_index=True)
    
    print("\nNumber of frames per class (after augmentation):")
    print(final_df['class'].value_counts())
    
    # Saving results
    final_df.to_csv(output_csv, index=False)
    print(f"\n✅ Augmented data successfully saved to: {output_csv}")

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    processed_data_folder = project_root / 'processed_data'
    
    INPUT_CSV = processed_data_folder / 'landmarks.csv'
    OUTPUT_CSV = processed_data_folder / 'landmarks_augmented.csv'
    
    augment_data(INPUT_CSV, OUTPUT_CSV)