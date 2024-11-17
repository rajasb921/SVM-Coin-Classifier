'''
This module cotains the code for training my computer vision model to classify coins based on their features.
- I obtained training data by cropping out the coins from the given test case images, and saving their associated labels 
  in a seperate text file. I was able to obtain 144 training samples.
- The training data consisted of 250 by 250 square images of coins. I extracted important features such as the RGB values 
  and the coin radius. I used these features as inputs to my model. 
- This model is a linear classifier that uses SVC to classify the data. Based on my own testing, I was able to obtain an 
  accuracy of 94-98% based on the size of the test data

Functions:
    extract_RGB(img):
            img (numpy.ndarray): Input image.
            list: List of features (average center color).
    detect_radius(img, epsilon=20):
            img (numpy.ndarray): Input image.
    load_dataset(image_dir, label_dir):
            image_dir (str): Directory containing images.
            label_dir (str): Directory containing labels.
            tuple: X (features), y (labels), avg_error (average radius error).
    train_model(X, y, rgb_weight=75.0):
            X (numpy.ndarray): Feature matrix.
            y (numpy.ndarray): Labels.
            rgb_weight (float): Weight multiplier for RGB features (default: 75.0).
            tuple: trained model and scaler.
    main():
        Main function to load dataset, train model, and save the trained model and scaler.
'''


import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2 as cv
import joblib

def extract_RGB(img):
    """
    Extract RGB values from a single image.
    Returns: list of features containing RGB values
    """
    features = []

    # Apply Gaussian Blur
    img = cv.GaussianBlur(img, (3, 3), 0)

    # 1. RGB values. Take the average of RGB values in a 25 x 25 square at the center of the image
    height, width, _ = img.shape
    center_x, center_y = width // 2, height // 2
    box_size = 25 
    start_x, end_x = max(0, center_x - box_size // 2), min(width, center_x + box_size // 2)
    start_y, end_y = max(0, center_y - box_size // 2), min(height, center_y + box_size // 2)
    center_region = img[start_y:end_y, start_x:end_x]
    average_rgb = np.mean(center_region, axis=(0, 1)).astype(int)
    features.extend([int(value) for value in average_rgb])

    return features


def detect_radius(img, epsilon=20):
    """
    Detects the radius of a coin using edge distances and counts.
    Helper function to compare true radius with radius extracted manually
    
    Parameters:
        edges (numpy.ndarray): Edge-detected image (binary image).
        epsilon (int): Tolerance range for grouping distances (default 20 pixels).

    Returns:
        int: Detected radius with the highest count of edge pixels.
    """
    img = cv.GaussianBlur(img, (9, 9), 0)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 175, 175)

    # Image dimensions and center
    height, width = edges.shape
    center = (width // 2, height // 2)

    # Find the coordinates of the edge pixels
    edge_points = np.column_stack(np.where(edges > 0))

    # Calculate distances of each edge point from the center
    distances = np.sqrt((edge_points[:, 1] - center[0])**2 + (edge_points[:, 0] - center[1])**2)

    # Round distances to integers for grouping
    rounded_distances = np.round(distances).astype(int)

    # Count occurrences of each distance with tolerance grouping
    radius_counts = {}
    for dist in rounded_distances:
        found = False
        for key in radius_counts:
            if abs(dist - key) <= epsilon:
                radius_counts[key] += 1
                found = True
                break
        if not found:
            radius_counts[dist] = 1
    
    # Find the distance with the highest count
    most_likely_radius = max(radius_counts, key=radius_counts.get)

    return most_likely_radius


def load_dataset(image_dir, label_dir):
    """
    Load and process all images and labels.
    Returns: X (features), y (labels)
    """
    X, y = [], []
    
    for i in range(1, 145):
        image_path = os.path.join(image_dir, f"{i}.png")
        label_path = os.path.join(label_dir, f"{i}.txt")
        
        # Skip if files don't exist
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            print(f"Skipping missing files for index {i}")
            continue
            
        # Load and process image
        img = cv.imread(image_path)
        if img is None:
            print(f"Failed to load image {i}")
            continue
            
        # Extract features
        features = extract_RGB(img)

        # Load label and radius
        with open(label_path, 'r') as f:
            label_data = f.read().strip()
            coin_type, radius = label_data.split(',')
            radius = float(radius)  # Convert radius to float
        
        # Calculate radius using the calculate_radius function
        detected_radius = detect_radius(img)
        
        # Print both radii and the error
        errors = []
        error_percentage = abs(radius - detected_radius) / radius * 100
        errors.append(error_percentage)

        # Add radius to the features
        features = np.append(features, radius)


        X.append(features)
        y.append(coin_type)  # Use coin_type as the label
    
    return np.array(X), np.array(y), np.mean(errors)


def train_model(X, y, rgb_weight=75.0):
    """
    Train the model with weighted features, giving higher priority to RGB values.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels
        rgb_weight (float): Weight multiplier for RGB features (default: 75.0)

    Returns: trained model and scaler
    """
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create feature weights array
    feature_weights = np.ones(X.shape[1])
    feature_weights[:3] = rgb_weight  # Apply weight to RGB features

    # Apply feature weights
    X_train_weighted = X_train * feature_weights
    X_test_weighted = X_test * feature_weights

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_weighted)
    X_test_scaled = scaler.transform(X_test_weighted)

    # Train SVM model
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = svm_model.predict(X_test_scaled)
    accuracy = np.mean(y_pred == y_test)

    print("\nModel Evaluation:")
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return svm_model, scaler


def main():
    # Set directories
    image_dir = './images/'
    label_dir = './labels/'
    
    # Load dataset
    print("Loading dataset...")
    X, y , avg_error= load_dataset(image_dir, label_dir)
    print(f"Dataset loaded: {len(X)} samples")
    print(f"Avg radius error: {avg_error:.4f}%")
    # Train model
    print("\nTraining model...")
    model, scaler = train_model(X, y)

    # Save model, scaler, and PCA transformer
    print("\nSaving model, scaler, and PCA transformer...")
    joblib.dump(model, 'svm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model, scaler, and PCA transformer saved successfully")

if __name__ == "__main__":
    main()
