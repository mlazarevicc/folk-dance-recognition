import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# --- Adjustable parameters ---
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10 # Number of epochs without improvement before stopping

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Generates and saves an image of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('The Confusion Matrix') 
    plt.ylabel('Actual Class') 
    plt.xlabel('Predicted Class') 
    plt.savefig(output_path) 
    plt.close() 
    print(f"Confusion matrix saved to: {output_path}")

def main():
    # Loading paths
    project_root = Path(__file__).resolve().parents[1]
    processed_data_folder = project_root / 'processed_data'
    models_folder = project_root / 'models'
    models_folder.mkdir(exist_ok=True)
    
    # Loading data
    data_path = processed_data_folder / 'sequences.npz'
    print("Učitavanje podataka...")
    data = np.load(data_path, allow_pickle=True)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    # Loading the label map
    with open(processed_data_folder / 'label_map.json', 'r') as f:
        label_map = json.load(f)
    class_names = [label_map[str(i)] for i in range(len(label_map))]
    
    # Data format check
    input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, attribute_number)
    num_classes = len(class_names)
    print(f"Input shape: {input_shape}") 
    print(f"Number of classes: {num_classes}") 

    # 1. Building the model 
    print("Building model...")
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.4),
        LSTM(256, return_sequences=False),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax') # Softmax for multiclass classification
    ])
    
    model.summary()
    
    # 2. Compiling the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', # Because labels are integers
                  metrics=['accuracy'])
    
    # Callbacks to improve training
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    # ReduceLROnPlateau: Reduces the learning rate if the loss levels off
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # 3. Model training
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )
    
    # 4. Model evaluation 
    print("\nModel evaluation on test data...") 
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0) 
    print(f"Test accuracy (Accuracy): {accuracy:.4f}") 
    print(f"Test loss (Loss): {loss:.4f}") 

    # More detailed analysis
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names, labels=range(num_classes)))
    
   # Saving the confusion matrix 
    cm_path = models_folder / 'confusion_matrix.png' 
    plot_confusion_matrix(y_test, y_pred, class_names, cm_path) 

    # 5. Model storage 
    model_path = models_folder / 'step_classifier.h5' 
    model.save(model_path) 
    print(f"\n ✅ Model successfully saved to: {model_path}")

if __name__ == '__main__':
    main()