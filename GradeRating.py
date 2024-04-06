import os
import cv2
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to preprocess an image
def preprocess_image(image):
    # Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to segment the meat portion
    _, binary_img = cv2.threshold(gray_img, 104, 255, cv2.THRESH_BINARY)
    
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

# Load images and preprocess them
def preprocess_images(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        central_region = preprocess_image(image)
        data.append([filename, central_region])
    return data

# Load training images and preprocess them
training_folder = './smalltrainset'
training_data = preprocess_images(training_folder)

# Extract features from training images
training_features = []
for filename, central_region in training_data:
    features = extract_features(central_region)
    training_features.append([filename] + list(features))
    print(filename, " trained")

# Load validation images and preprocess them
validation_folder = './smallvalset'
validation_data = preprocess_images(validation_folder)

# Extract features from validation images
validation_features = []
for filename, central_region in validation_data:
    features = extract_features(central_region)
    validation_features.append([filename] + list(features))
    print(filename, " validated")

# Load the CSV file containing the known data
known_data_file = './Path1 Challenge Training Data.csv'
known_data = []
with open(known_data_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        known_data.append(row)

# Extract known ratings for validation images
validation_results = []
for filename, fat_percentage, _ in validation_features:
    for row in known_data:
        if filename in row[0]:
            carcass_id = row[1]
            numerical_rating = row[2]
            break
    else:
        # If the filename is not found in known data, skip this image
        continue
    
    # Classify fat percentage into categories
    if fat_percentage < 400:
        rating = 'Low Choice'
    elif 400 <= fat_percentage < 500:
        rating = 'Upper 2/3 Choice'
    elif 500 <= fat_percentage < 700:
        rating = 'Select'
    elif 700 <= fat_percentage <= 1100:
        rating = 'Standard'
    else:
        rating = 'Unknown'

    validation_results.append([filename, carcass_id, fat_percentage, numerical_rating, rating])

# Print validation results
print("Number of validation results:", len(validation_results))
print("Validation results:", validation_results)

# Write validation results to CSV
with open('validation_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image', 'Carcass_ID', 'Fat_Percentage', 'Numerical_Rating', 'Grade'])
    writer.writerows(validation_results)
