import cv2
import numpy as np

# Define the preprocess_image function
def preprocess_image(image):
    # Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find contours to identify the central region
    contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the central region
    central_region = gray_img[y:y+h, x:x+w]
    
    return central_region

# Define the extract_features function
def extract_features(image):
    # Define the threshold for classifying pixels
    threshold = 150
    
    # Threshold the image
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    # Calculate the total number of pixels in the central region
    total_pixels = image.shape[0] * image.shape[1]
    
    # Calculate the percentage of white pixels (fat content) in the central region
    white_pixels = np.sum(binary_image == 255)
    white_percentage = (white_pixels / total_pixels) * 100
    
    # Calculate the percentage of fat content (inverse of white percentage)
    fat_percentage = 100 - white_percentage
    
    return fat_percentage

# Load an example image
image_path = './Path1 Challenge Training Images/00000001-1.tif'
image = cv2.imread(image_path)

# Preprocess the image
processed_image = preprocess_image(image)

# Ensure the processed image has three channels
processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

# Extract features from the processed image
features = extract_features(processed_image)

# Display the processed image and extracted features
cv2.imshow('Processed Image', processed_image)
print('Percentage of Fat Content:', features)
cv2.waitKey(0)
cv2.destroyAllWindows()
