import cv2
import numpy as np

# Function to preprocess an image
def preprocess_image(image_path):
    # Read the image in grayscale
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply thresholding to segment the meat portion
    _, binary_img = cv2.threshold(gray_img, 107, 255, cv2.THRESH_BINARY)
    
    # Invert the binary image
    binary_img = cv2.bitwise_not(binary_img)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask for the largest contour
    mask = np.zeros_like(gray_img)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Apply the mask to the original grayscale image to extract the meat portion
    meat_portion = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    
    return meat_portion

# Function to extract features from an image
def extract_features(image):
    # Define thresholds for fat and meat
    fat_threshold = 100
    meat_threshold = 200
    
    # Calculate the total number of pixels in the image
    total_pixels = image.shape[0] * image.shape[1]
    
    # Initialize counters for fat and meat pixels
    fat_pixels = 0
    meat_pixels = 0
    
    # Iterate through each pixel in the image
    for row in image:
        for pixel in row:
            if pixel < fat_threshold:
                fat_pixels += 1
            elif fat_threshold <= pixel < meat_threshold:
                meat_pixels += 1
    
    # Calculate the percentage of fat and meat pixels
    fat_percentage = (fat_pixels / total_pixels) * 100
    meat_percentage = (meat_pixels / total_pixels) * 100
    
    return fat_percentage, meat_percentage


# Load an example image
image_path = './Path1 Challenge Images for Validation/00001007-2.tif'

# Preprocess the image
processed_image = preprocess_image(image_path)

# Extract features from the processed image
features = extract_features(processed_image)

# Display the processed image and extracted features
cv2.imshow('Processed Image', processed_image)
print('Percentage of Gray Fat Content:', features)
cv2.waitKey(0)
cv2.destroyAllWindows()
