# Automatic Recognition of Folk Dance Steps

## Introduction
🕺 This project represents a machine learning and computer vision-based system designed for **automatic classification of Serbian folk dance steps** from video recordings.  

By utilizing advanced techniques such as **pose detection** and **recurrent neural networks (LSTM)**, the system learns to recognize subtle temporal patterns in lower-body movements and assign them to the corresponding dance step category.  

The entire project is implemented in **Python**, using modern libraries for deep learning and data processing.

---

## Key Features
- **Keypoint Extraction:** Automatic detection and extraction of 3D body landmarks from video material using the **Google MediaPipe** library.  
- **Data Normalization:** Ensures that the model remains independent of the subject’s distance from the camera or body proportions.  
- **Data Augmentation:** Artificial dataset expansion by adding controlled noise to improve model robustness and generalization.  
- **Sequence Classification:** Application of **Long Short-Term Memory (LSTM)** networks specialized for learning sequential movement patterns.  
- **Real-Time Evaluation:** Ability to test the trained model live using a webcam for instant dance step recognition.

---

## Technologies Used

| Technology        | Purpose                                                                  |
|-------------------|--------------------------------------------------------------------------|
| **Python**        | Main programming language of the project.                                |
| **TensorFlow / Keras** | Building, training, and evaluating the LSTM model.                  |
| **OpenCV**        | Video file processing and webcam input handling.                         |
| **MediaPipe**     | Pose detection and landmark extraction.                                  |
| **Scikit-learn**  | Dataset splitting and model evaluation (confusion matrix, reports).      |
| **NumPy**         | Efficient numerical operations and array handling.                       |
| **Pandas**        | Loading, manipulating, and saving tabular data (CSV).                    |
| **Matplotlib / Seaborn** | Visualization of results, especially confusion matrices.          |

---

## Project Structure

```bash
folk-dance-ai-recognition/
│
├── data/                   # Folder containing video samples sorted by class
│   ├── cucanj/
│   ├── trokorak/
│   └── …
│
├── models/                 # Trained models and evaluation results
│   ├── step_classifier.h5
│   └── confusion_matrix.png
│
├── processed_data/          # Intermediate processed files
│   ├── label_map.json
│   ├── landmarks.csv
│   ├── landmarks_augmented.csv
│   └── sequences.npz
│
├── src/                    # All source Python scripts
│   ├── argument_data.py
│   ├── create_sequences.py
│   ├── train_model.py
│   ├── process_data.py
│   └── real_time_test.py
│
├── presentation.ipynb   # Jupyter Notebook presentation
├── requirements.txt     # List of required dependencies
└── README.md            # This file
```

---

## Setup and Installation

To run this project locally, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repository.git
cd folk-dance-ai-recognition
```
### 2. Create and activate a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install required libraries
Make sure to install all dependencies listed in requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

The project is executed by running the Python scripts from the `src/` directory **in the following order**:

---

### Step 0: Data Collection
Before running the scripts, fill the `data/` folder with video recordings.  
Each dance step class should have its own subfolder (e.g., `data/trokorak/`).

**Recording Guidelines:**
- **Duration:** Each video should last at least 5–7 seconds with continuous repetition of the same step.  
- **Diversity:** Record multiple videos per class from slightly different angles and speeds to help the model generalize better.

---

### Step 1: Video Processing & Landmark Extraction
This script processes all videos, detects poses, and saves normalized coordinates to `processed_data/landmarks.csv`.

```bash
python src/process_data.py
```

### Step 2: Data Augmentation
This script artificially increases the number of samples for underrepresented classes and saves the result in processed_data/landmarks_augmented.csv.

```bash
python src/argument_data.py
```

### Step 3: Sequence Creation
Takes the augmented data, creates overlapping sequences, and splits them into training, validation, and test sets.
Results are saved to processed_data/sequences.npz.

```bash
python src/create_sequences.py
```

### Step 4: Model Training

Starts the LSTM training process.
The trained model is stored as models/step_classifier.h5, and evaluation results (confusion matrix) are saved in the models/ folder.

```bash
python src/train_model.py
```

### Step 5: Real-Time Testing

For live demonstration, run the following script.
It will activate your webcam and display real-time step predictions:

```bash
python src/real_time_test.py
```

---

## Project Presentation

For a clear and concise walkthrough of the entire process — from data processing to final results — see the **presentation.ipynb** notebook.

---

### Author
Milan Lazarević SV4/2022
