'''
Coin Detection and Classification Module

- This module provides functions to detect and classify coins in an image using edge detection, 
  morphological operations, and a pre-trained Support Vector Machine (SVM) model. 
- The module includes functions to detect the radius of a coin, extract features from a coin, 
  and process an image to identify and classify coins.
- Coin detection is done via watershedding (inspired by OpenCV implementation https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)
- After detecting coin centers, the image is again processed to be input into the SVM model. Details about how the SVM model works are included
  in './training/cv_modeltraining_project3.py'

Functions:
    detect_radius(img, x, y, epsilon=20):
    extract_coin_features(img, center_x, center_y, rgb_weight=75.0):
        Extracts a 250x250 region around the coin center and computes weighted features.
    read_ground_truth(filename):
        Extracts true coin centers and classifications to show difference
    main():
        Main function to load the trained model and scaler, read the input image, process the 
        image to detect coins, extract features, classify coins, and output the results.

'''

import cv2 as cv
import numpy as np
import joblib
import matplotlib.pyplot as plt

def detect_radius(img, x, y, epsilon=20):
    """
    Detects the radius of a coin using edge distances and counts.
    
    Parameters:
        edges (numpy.ndarray): Edge-detected image (binary image).
        epsilon (int): Tolerance range for grouping distances (default 20 pixels).

    Returns:
        int: Detected radius with the highest count of edge pixels.
    """
    # Define the boundaries for the 250x250 region
    half_size = 125  # Half of 250
    start_x = max(0, x - half_size)
    end_x = min(img.shape[1], x + half_size)
    start_y = max(0, y - half_size)
    end_y = min(img.shape[0], y + half_size)

    # Crop the 250x250 region
    region = img[start_y:end_y, start_x:end_x]

    gray = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7,7), 0)
    edges = cv.Canny(gray, 100, 100)

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

def read_ground_truth(filename):
    """
    Read ground truth data from a text file.
    Format: First line = number of coins
           Following lines = x y label radius
    Returns: List of tuples (x, y, label, radius)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        num_coins = int(lines[0].strip())
        ground_truth = []
        for line in lines[1:]:
            x, y, label, radius = line.strip().split()
            ground_truth.append((int(x), int(y), label, float(radius)))
    return ground_truth

def extract_coin_features(img, center_x, center_y, rgb_weight=75.0):
    """
    Extract a 250x250 region around the coin center and compute weighted features.
    Arguments:
        img: Input image (assumed to be a larger image)
        center_x: x-coordinate of the coin center
        center_y: y-coordinate of the coin center
        rgb_weight: Weight multiplier for RGB features (default: 2.0)
    Returns:
        np.array: Weighted feature vector [average RGB values, area, average_radius]
    """
    # Define the boundaries for the 250x250 region
    half_size = 125  # Half of 250
    start_x = max(0, center_x - half_size)
    end_x = min(img.shape[1], center_x + half_size)
    start_y = max(0, center_y - half_size)
    end_y = min(img.shape[0], center_y + half_size)

    # Crop the 250x250 region
    region = img[start_y:end_y, start_x:end_x]

    # Apply Gaussian Blur
    region = cv.GaussianBlur(region, (3, 3), 0)

    # Convert to grayscale for radius detection
    gray = cv.cvtColor(region, cv.COLOR_BGR2GRAY)

    features = []

    # 1. Center color (25x25 box in the center)
    region_height, region_width, _ = region.shape
    region_center_x, region_center_y = region_width // 2, region_height // 2
    box_size = 25

    # Extract RGB values in the center box
    box_start_x = max(0, region_center_x - box_size // 2)
    box_end_x = min(region_width, region_center_x + box_size // 2)
    box_start_y = max(0, region_center_y - box_size // 2)
    box_end_y = min(region_height, region_center_y + box_size // 2)

    # Compute average RGB values in the center box
    center_box = region[box_start_y:box_end_y, box_start_x:box_end_x]
    average_rgb = np.mean(center_box, axis=(0, 1)).astype(int)
    
    # Apply RGB weighting
    weighted_rgb = [int(value * rgb_weight) for value in average_rgb]
    features.extend(weighted_rgb)

    # 2. Radius time
    radius = detect_radius(img, center_x, center_y)
    features.append(radius)

    return np.array(features)

# Load the trained model and scaler
model = joblib.load('./training/svm_model.pkl')
scaler = joblib.load('./training/scaler.pkl')  # Make sure to load the scaler too


# Read the image filename from stdin
image_filename = input("Enter image filename: ").strip()
truth_filename = input("Enter ground truth filename: ").strip()
img_og = cv.imread(image_filename, cv.IMREAD_COLOR)
img = cv.GaussianBlur(img_og, (9, 9), 0)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

kernel1 = np.ones((15, 15), np.uint8)
kernel2 = np.ones((25, 25), np.uint8)
kernel3 = np.ones((35, 35), np.uint8)
kernel4 = np.ones((45, 45), np.uint8)
kernel5 = np.ones((55, 55), np.uint8)
kernel6 = np.ones((65, 65), np.uint8)
kernel7 = np.ones((75, 75), np.uint8)


closing2 = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel2)
#cv2_imshow(closing2)


opening1 = cv.morphologyEx(closing2, cv.MORPH_OPEN, kernel1)
#cv2_imshow(opening1)
surebg1 = cv.dilate(opening1, kernel1, iterations=1)
surebg2 = opening1

dist_transform1 = cv.distanceTransform(opening1, cv.DIST_L2, 5)
ret1, sure_fg1 = cv.threshold(dist_transform1, 0.625 * dist_transform1.max(), 255, 0)


sure_fg = np.uint8(sure_fg1)
unknown = cv.subtract(surebg1, sure_fg)


# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
 
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
 
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

# Convert markers to a format suitable for coloring
markers_display = np.uint8(markers * (255 / markers.max()))  # Scale to 0-255

# Apply the jet colormap
jetmap = cv.applyColorMap(markers_display, cv.COLORMAP_JET)

# Initialize an empty list to store the center information
coins = []

# Find the centers of each component in markers and store in coins
unique_markers = np.unique(markers)

min_area = 20
max_area = 20000
for marker in unique_markers:
    if marker == 0:  # Skip the background
        continue
    # Get coordinates of all pixels in this marker region
    region_coords = np.column_stack(np.where(markers == marker))

    # Calculate the area of the component (number of pixels in the region)
    area = len(region_coords)

    # Calculate the centroid by averaging the coordinates
    if area > min_area and area < max_area:
        center = np.mean(region_coords, axis=0).astype(int)
        x, y = center[1], center[0]  # x and y coordinates
        # Draw a dot at the center on the jetmap
        cv.circle(jetmap, (x, y), 5, (0, 0, 0), -1)  # Black dot

        # Example usage
        features = extract_coin_features(img_og, x, y)
        features_scaled = scaler.transform(features.reshape(1, -1))
        coin_type = model.predict(features_scaled)[0]

        coins.append((x, y, coin_type))

# Output results
print(len(coins))
for x, y, coin_type in coins:
    print(f"{x} {y} {coin_type}")

# Read ground truth data
ground_truth = read_ground_truth(truth_filename)

# Create a copy of the original image for visualization
comparison_img = img.copy()

# Define visualization parameters
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_thickness = 2
prediction_color = (0, 255, 0)  # Green for predictions
truth_color = (0, 0, 255)      # Red for ground truth
match_color = (255, 255, 0)    # Yellow for matches

# Function to draw label
def draw_label(image, x, y, label, color, offset_y=20):
    text_position = (x - 20, y - offset_y)
    (text_width, text_height), _ = cv.getTextSize(label, font, font_scale, font_thickness)
    cv.rectangle(image, 
                (text_position[0] - 2, text_position[1] - text_height - 2),
                (text_position[0] + text_width + 2, text_position[1] + 2),
                (0, 0, 0),
                -1)
    cv.putText(image, 
               label, 
               text_position,
               font,
               font_scale,
               color,
               font_thickness)

# Plot predictions and ground truth
for x, y, pred_label in coins:
    # Draw prediction (green)
    cv.circle(comparison_img, (x, y), 5, prediction_color, -1)
    draw_label(comparison_img, x, y, f"P:{pred_label}", prediction_color)

for x, y, true_label, radius in ground_truth:
    # Draw ground truth (red)
    cv.circle(comparison_img, (x, y), 5, truth_color, -1)
    draw_label(comparison_img, x, y, f"T:{true_label}", truth_color, offset_y=40)
    # Draw circle with the given radius
    cv.circle(comparison_img, (x, y), int(radius), truth_color, 2)

# Calculate and display metrics
def calculate_matches(predictions, ground_truth, distance_threshold=30):
    """
    Calculate matching predictions and ground truth points based on distance threshold
    """
    matches = 0
    correct_labels = 0
    pred_points = [(x, y, label) for x, y, label in predictions]
    truth_points = [(x, y, label) for x, y, label, _ in ground_truth]
    
    for tx, ty, tlabel, _ in ground_truth:
        min_dist = float('inf')
        closest_pred = None
        for px, py, plabel in pred_points:
            dist = np.sqrt((tx-px)**2 + (ty-py)**2)
            if dist < min_dist:
                min_dist = dist
                closest_pred = (px, py, plabel)
        
        if min_dist <= distance_threshold:
            matches += 1
            if closest_pred[2] == tlabel:
                correct_labels += 1
                # Draw match connection in yellow
                cv.line(comparison_img, (tx, ty), (closest_pred[0], closest_pred[1]), match_color, 1)

    return matches, correct_labels

matches, correct_labels = calculate_matches(coins, ground_truth)

# Create a larger figure with three subplots
plt.figure(figsize=(20, 7))

# Show segmentation map
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(jetmap, cv.COLOR_BGR2RGB))
plt.title('Segmentation Map')
plt.axis('off')

# Show predictions
plt.subplot(1, 3, 2)
pred_img = img.copy()
for x, y, pred_label in coins:
    cv.circle(pred_img, (x, y), 5, prediction_color, -1)
    draw_label(pred_img, x, y, pred_label, prediction_color)
plt.imshow(cv.cvtColor(pred_img, cv.COLOR_BGR2RGB))
plt.title('Model Predictions')
plt.axis('off')

# Show comparison
plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(comparison_img, cv.COLOR_BGR2RGB))
plt.title(f'Comparison\nMatches: {matches}/{len(ground_truth)} ({matches/len(ground_truth)*100:.1f}%)\n'
          f'Correct Labels: {correct_labels}/{len(ground_truth)} ({correct_labels/len(ground_truth)*100:.1f}%)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Print metrics
print(f"\nMetrics:")
print(f"Total ground truth coins: {len(ground_truth)}")
print(f"Total detected coins: {len(coins)}")
print(f"Matching detections: {matches} ({matches/len(ground_truth)*100:.1f}%)")
print(f"Correct labels: {correct_labels} ({correct_labels/len(ground_truth)*100:.1f}%)")