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
- Automated coin detection using the watershed algorithm
- Feature extraction including RGB values and coin radius
- SVM-based classification with high accuracy (94-98%)
- Efficient preprocessing using morphological operations
- Modular design for ease of use and extension

### Built With
* Python
* OpenCV
* scikit-learn
* NumPy
* joblib

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
- Python 3.x
- pip package manager

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo-name.git
   ```
2. Install required packages:
   ```sh
   pip install opencv-python numpy scikit-learn joblib
   ```
3. Add your dataset:
   - Save cropped coin images in the `images/` folder.
   - Create a `labels/` folder and include the associated labels in text format.

<!-- USAGE -->
## Usage

1. **Model Training**
   ```sh
   python ./training/cv_modeltraining_project3.py
   ```
   This will:
   - Load and process the training dataset
   - Train the SVM model using RGB and radius features
   - Save the trained model and scaler for later use

2. **Coin Detection and Classification**
   ```sh
   python project3.py
   ```
   Input an image file when prompted, and the system will:
   - Detect coin locations
   - Extract features for each coin
   - Classify the coins and output the results

<!-- SYSTEM COMPONENTS -->
## System Components

### 1. Model Training (`cv_modeltraining_project3.py`)
- **Feature Extraction**: 
  - Extracts RGB color values and coin radius
  - Processes 250x250 pixel regions of coin images
- **SVM Model Training**: 
  - Implements a linear SVM classifier using scikit-learn
  - Supports feature weighting for RGB values
- **Performance**:
  - Accuracy of 94-98% on test datasets
  - Saves the trained model and scaler using joblib for future use

### 2. Coin Detection and Classification (`project3.py`)
- **Detection Techniques**:
  - Uses the watershed algorithm for coin detection (inspired by [OpenCV Watershed Documentation](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html))
  - Refines detection using morphological operations
- **Feature Extraction**:
  - Extracts RGB values and calculates the coin radius for detected regions
  - Prepares features for classification
- **Classification**:
  - Loads the trained SVM model and scaler
  - Outputs the number of detected coins and their classifications
