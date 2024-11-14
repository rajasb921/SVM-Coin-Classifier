import cv2
import os

# List of file numbers to process
file_numbers = [1, 6, 11, 16, 21, 26, 31, 36, 41]

# Init a counter for file_num
fileNum = 1

# Output directory
img_dir = "../training/images/"
label_dir = "../training/labels/"

# Loop over each file number
for num in file_numbers:
    # Define paths for the image and text file
    img_path = f"{num}.png"
    txt_path = f"{num}.txt"
    
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image {img_path} not found.")
        continue
    
    # Read the centers from the text file
    with open(txt_path, "r") as f:
        lines = f.readlines()[1:]
    
    # Loop over each line in the text file
    for i, line in enumerate(lines):
        # Extract the x and y center coordinates
        x_center, y_center, type = map(int, line.split()[:3])
        
        # Define cropping bounds
        x_start = max(0, x_center - 125)
        y_start = max(0, y_center - 125)
        x_end = x_start + 250
        y_end = y_start + 250

        # Crop the image to a 250x250 area around the center
        cropped_img = img[y_start:y_end, x_start:x_end]
        
        # Save the cropped image
        output_path = img_dir + f"{fileNum}.png"
        cv2.imwrite(output_path, cropped_img)
        # Save the type to a text file
        with open(label_dir + f"{fileNum}.txt", "w") as type_file:
            type_file.write(str(type))
        
        # Increment the file number counter
        fileNum += 1
        print(f"Saved cropped image {output_path}")

print("Cropping completed.")
