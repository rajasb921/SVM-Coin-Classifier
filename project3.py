import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

def extract_coin_features(img, center_x, center_y):
    """
    Extract a 250x250 region around the coin center and compute features.
    """
    # Calculate boundaries for 250x250 square
    half_size = 125  # 250//2
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Calculate extraction boundaries with padding
    start_y = max(0, center_y - half_size)
    end_y = min(height, center_y + half_size)
    start_x = max(0, center_x - half_size)
    end_x = min(width, center_x + half_size)
    
    # Extract region and resize to 250x250 if necessary
    coin_region = img[start_y:end_y, start_x:end_x]
    if coin_region.shape[:2] != (250, 250):
        coin_region = cv2.resize(coin_region, (250, 250))
    
    # Extract features (matching training features)
    features = []
    
    # 1. Get grayscale for contour
    gray_region = cv2.cvtColor(coin_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_region, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contour area
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
    else:
        area = 0
    features.append(area)
    
    # 2. Get center color (3x3 kernel at exact center)
    center_kernel = coin_region[123:126, 123:126]  # 3x3 at center of 250x250
    avg_color = np.mean(center_kernel, axis=(0, 1))
    features.extend(avg_color)
    
    # 3. Add pixel values
    features.extend(gray_region.flatten())
    
    return np.array(features)

# Load the trained model and scaler
svm = joblib.load('./training/svm_model.pkl')
scaler = joblib.load('./training/scaler.pkl')  # Make sure to load the scaler too

MIN_COIN_AREA = 23000
MAX_COIN_AREA = 58000

# Read the image filename from stdin
filename = input().strip()
img = cv2.imread(filename, cv2.IMREAD_COLOR)
img_normalized = img.astype(np.float32) / 255.0

# Perform Otsu's binarization
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
_, binary_img_8bit = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphology
kernel = np.ones((3, 3), dtype=np.uint8)
final_binary = cv2.dilate(binary_img_8bit, kernel)
final_binary = cv2.erode(final_binary, kernel)

# Coin detection
contours, _ = cv2.findContours(final_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
coin_info = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > MIN_COIN_AREA and area < MAX_COIN_AREA:
        # Find centroid
        sum_x = 0
        sum_y = 0
        num_points = len(cnt)

        for point in cnt:
            sum_x += point[0][0]
            sum_y += point[0][1]

        if num_points != 0:
            center_x = int(sum_x / num_points)
            center_y = int(sum_y / num_points)
            
            # Extract features for this coin
            features = extract_coin_features(img, center_x, center_y)
            
            # Scale features
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Predict coin type
            coin_type = svm.predict(features_scaled)[0]
            
            coin_info.append(((center_x, center_y), coin_type, area))

# Output results
print(len(coin_info))
for center, coin_type, area in coin_info:
    print(f"{center[0]} {center[1]} {coin_type}")