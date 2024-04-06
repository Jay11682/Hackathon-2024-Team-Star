import os
import cv2
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to preprocess an image
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

# Function to extract features from an image
def extract_features(image):
    # Divide the image into segments (for now, use a dummy implementation)
    segments = [image]
    
    # Calculate the percentage of white pixels (fat content) in each segment
    white_percentages = []
    for segment in segments:
        total_pixels = segment.shape[0] * segment.shape[1]
        white_pixels = np.sum(segment == 255)  # Assuming white pixels are coded as 255
        white_percentage = (white_pixels / total_pixels) * 100
        white_percentages.append(white_percentage)
    
    return white_percentages

# Function to write data to a CSV file
def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'Percentage_of_White_Fat_Content'])
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
training_folder = './Path1 Challenge Training Images'
training_data = preprocess_images(training_folder)

# Extract features from training images
training_features = []
for filename, central_region in training_data:
    features = extract_features(central_region)
    training_features.append([filename] + features)

# Load validation images and preprocess them
validation_folder = './Path1 Challenge Images for Validation'
validation_data = preprocess_images(validation_folder)

# Extract features from validation images
validation_features = []
for filename, central_region in validation_data:
    features = extract_features(central_region)
    validation_features.append([filename] + features)

# Write training data to CSV
write_to_csv(training_features, 'training_data.csv')

# Write validation data to CSV
write_to_csv(validation_features, 'validation_data.csv')

# Split features and labels
X_train = np.array([row[1:] for row in training_features])
y_train = np.array([row[1] for row in training_features])

X_val = np.array([row[1:] for row in validation_features])
y_val = np.array([row[1] for row in validation_features])

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
print("Mean Squared Error on Validation Set:", mse)
