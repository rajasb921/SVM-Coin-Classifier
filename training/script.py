import os
from PIL import Image

# Directories
image_dir = "./images/"
label_dir = "./labels/"

# Get list of image and label files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

# Ensure the number of images and labels match
assert len(image_files) == len(label_files), "Number of images and labels do not match"

# Process each image and label pair
for i, (image_file, label_file) in enumerate(zip(image_files, label_files), start=1):
    # Open image
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)

    # Flip image horizontally and save
    flipped_h = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_h.save(os.path.join(image_dir, f"{i + 144}.png"))

    # Flip image vertically and save
    flipped_v = image.transpose(Image.FLIP_TOP_BOTTOM)
    flipped_v.save(os.path.join(image_dir, f"{i + 288}.png"))

    # Copy label file for both flipped images
    label_path = os.path.join(label_dir, label_file)
    with open(label_path, 'r') as label:
        label_content = label.read()
    
    with open(os.path.join(label_dir, f"{i + 144}.txt"), 'w') as new_label:
        new_label.write(label_content)
    
    with open(os.path.join(label_dir, f"{i + 288}.txt"), 'w') as new_label:
        new_label.write(label_content)

print("Image flipping and labeling completed.")