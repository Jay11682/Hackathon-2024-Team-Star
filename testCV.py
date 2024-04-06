# import cv2
# import numpy as np

# # Read the image in grayscale
# gray_img = cv2.imread('./Path1 Challenge Images for Validation/00001001-2.tif', cv2.IMREAD_GRAYSCALE)

# # Apply thresholding to segment the meat portion
# _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

# # Invert the binary image
# binary_img = cv2.bitwise_not(binary_img)

# # Find contours in the binary image
# contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Find the contour with the largest area
# largest_contour = max(contours, key=cv2.contourArea)

# # Create a mask for the largest contour
# mask = np.zeros_like(gray_img)
# cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

# # Apply the mask to the original grayscale image to extract the meat portion
# meat_portion = cv2.bitwise_and(gray_img, gray_img, mask=mask)

# # Display the result
# cv2.imshow("Meat Portion", meat_portion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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
    # Calculate the total number of pixels in the central region
    total_pixels = image.shape[0] * image.shape[1]
    
    # Calculate the percentage of white pixels (fat content) in the central region
    white_pixels = np.sum(image == 255)  # Assuming white pixels are coded as 255
    white_percentage = (white_pixels / total_pixels) * 100
    
    return white_percentage

# Load an example image
image_path = './Path1 Challenge Training Images/00000001-1.tif'
image = cv2.imread(image_path)

# Preprocess the image
processed_image = preprocess_image(image)

# Extract features from the processed image
features = extract_features(processed_image)

# Display the processed image and extracted features
cv2.imshow('Processed Image', processed_image)
print('Percentage of White Fat Content:', features)
cv2.waitKey(0)
cv2.destroyAllWindows()