<a name="readme-top"></a>
<br />
<h1 align="center">Coin Classification System</h1>

  <p align="center">
    A computer vision-based system for automated coin detection and classification using Support Vector Machines
  </p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#system-components">System Components</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
This project implements a coin classification system using computer vision and machine learning techniques. The system can detect coins in images and classify them based on their visual features. It uses a combination of image processing techniques for coin detection and Support Vector Machine (SVM) classification for determining coin types.

Key features include:
- Automated coin detection using contour analysis
- Feature extraction including contour area, color analysis, and pixel data
- SVM-based classification with optimized hyperparameters
- Data augmentation through image flipping
- Robust preprocessing including Otsu's binarization and morphological operations

### Built With
* Python
* OpenCV
* scikit-learn
* NumPy
* Pillow (PIL)
* joblib

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
- Python 3.x
- pip package manager

### Installation
1. Clone the repository
2. Install required packages:
```sh
pip install opencv-python numpy scikit-learn pillow joblib
```
3. Create directories for your dataset:
```sh
mkdir images labels
```

<!-- USAGE -->
## Usage

1. **Data Preparation**
   ```sh
   python script.py
   ```
   This will augment your training data by creating flipped versions of your images.

2. **Model Training**
   ```sh
   python cv_modeltraining_project3.py
   ```
   This will:
   - Load and process the training dataset
   - Perform grid search for optimal SVM parameters
   - Train the model and save it along with the scaler

3. **Coin Classification**
   ```sh
   python project3.py
   ```
   Input an image file when prompted, and the system will output:
   - Number of coins detected
   - Location and classification for each coin

<!-- SYSTEM COMPONENTS -->
## System Components

### 1. Data Augmentation (`script.py`)
- Creates horizontal and vertical flips of training images
- Maintains corresponding label files
- Triples the size of the training dataset

### 2. Model Training (`cv_modeltraining_project3.py`)
- Feature extraction:
  - Contour area calculation
  - Center color analysis
  - Grayscale pixel values
- GridSearchCV for hyperparameter optimization
- Model evaluation with classification reports and confusion matrix
- Model and scaler persistence using joblib

### 3. Coin Detection and Classification (`project3.py`)
- Image preprocessing:
  - Otsu's binarization
  - Morphological operations
- Coin detection using contour analysis
- Feature extraction from detected regions
- Classification using the trained SVM model

The system uses predefined area thresholds (MIN_COIN_AREA: 23000, MAX_COIN_AREA: 58000) for initial coin detection and extracts features from 250x250 pixel regions around detected coin centers.
