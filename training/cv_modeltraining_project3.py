import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import joblib

def extract_features(img):
    """
    Extract features from a single image.
    Returns: list of features (area, color, pixels)
    """
    features = []
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Contour area
    blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    _, thresh = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
    else:
        area = 0
    features.append(area)
    
    # 2. Center color (3x3 kernel)
    center_y, center_x = 125, 125
    kernel = img[center_y-1:center_y+2, center_x-1:center_x+2]
    avg_color = np.mean(kernel, axis=(0, 1))
    features.extend(avg_color)
    
    # 3. Pixel values (flattened grayscale image)
    # pixel_values = gray_img.flatten()
    # features.extend(pixel_values)
    
    return features

def load_dataset(image_dir, label_dir):
    """
    Load and process all images and labels.
    Returns: X (features), y (labels)
    """
    X, y = [], []
    
    for i in range(1, 301):
        image_path = os.path.join(image_dir, f"{i}.png")
        label_path = os.path.join(label_dir, f"{i}.txt")
        
        # Skip if files don't exist
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            print(f"Skipping missing files for index {i}")
            continue
            
        # Load and process image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image {i}")
            continue
            
        # Extract features
        features = extract_features(img)
        X.append(features)
        
        # Load label
        with open(label_path, 'r') as f:
            label = f.read().strip()
            y.append(label)
    
    return np.array(X), np.array(y)

def train_model(X, y, perform_gridsearch=True):
    """
    Train the SVM model with optional GridSearch for hyperparameter tuning.
    Returns: trained model, scaler
    """
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if perform_gridsearch:
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'poly']
        }
        
        # Perform GridSearch
        grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        print("Best parameters:", grid_search.best_params_)
        svm = grid_search.best_estimator_
    else:
        # Use default parameters
        svm = SVC(kernel='rbf', C=1, gamma='auto')
        svm.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = svm.predict(X_test_scaled)
    accuracy = np.mean(y_pred == y_test)
    
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return svm, scaler

def main():
    # Set directories
    image_dir = './images/'
    label_dir = './labels/'
    
    # Load dataset
    print("Loading dataset...")
    X, y = load_dataset(image_dir, label_dir)
    print(f"Dataset loaded: {len(X)} samples")
    
    # Train model
    print("\nTraining model...")
    model, scaler = train_model(X, y, perform_gridsearch=True)
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, 'svm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved successfully")

if __name__ == "__main__":
    main()