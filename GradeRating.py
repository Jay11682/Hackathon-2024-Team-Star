import os
import cv2
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

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

# Function to write data to a CSV file
def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'Percentage_of_Gray_Fat_Content'])
        writer.writerows(data)

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

# Write training data to CSV
write_to_csv(training_features, 'training_data.csv')

# Write validation data to CSV
write_to_csv(validation_features, 'validation_data.csv')

# Split features and labels
X_train = np.array([row[1:] for row in training_features])
y_train = np.array([row[1] for row in training_features])

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_train)

# Correlate fat percentages with rating
known_data_file = 'Path1 Challenge Training Data.csv'
known_data = []
with open(known_data_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        known_data.append(row)

fat_percentages = [row[1] for row in training_features]
grade_ratings = []
for filename, fat_percentage in zip(training_features, fat_percentages):
    found = False
    for data in known_data:
        if data[0] == filename:
            grade_ratings.append([filename, fat_percentage, data[2]])
            found = True
            break
    if not found:
        print(f"No data found for {filename}")

# Write training results to CSV
with open('training_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image', 'Percentage_of_Gray_Fat_Content', 'Rating'])
    writer.writerows(grade_ratings)
    print("Training results have been written to 'training_results.csv'.")
