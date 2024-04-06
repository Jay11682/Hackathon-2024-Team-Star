import cv2
import pandas as pd

# Variables to keep a running count of each steak type for averaging
total_fat_percentages = {
    'Select': [],
    'Prime': [],
    'Low Choice': [],
    'Upper 2/3 Choice': [],
    'Prime': []
}

# Load the data from the spreadsheet
dataframe = pd.read_excel('Path1 Challenge Training Data.xlsx')

def process_image_to_black_and_white(image_path):
    image = cv2.imread(image_path)
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_img = cv2.threshold(grayimage, 95, 255, cv2.THRESH_BINARY)
    return thresholded_img

# hi = process_image_to_black_and_white('./Path1 Challenge Images for Validation/00001001-2.tif')
# cv2.imshow("Display window", hi)
# cv2.waitKey(0)

# def calculate_fat_percentage(image):
#     # Identify the main part of the steak
#     # Divide into quadrants
#     # Calculate fat percentage
#     # This is a placeholder for the actual image processing code
#     pass

# for index, row in dataframe.iterrows():
#     # Convert current image to black and white for simplicity
#     image_path = f"Path1 Challenge Training Images/{row['Filename']}"
#     bw_image = process_image_to_black_and_white(image_path)
    
#     # Identify the main part of the steak from the background / outer fat
#     # Divide the main part into quadrants or more sections
#     quadrants = []  # This should be the result of actual image processing
#     total_fat = 0

#     for quadrant in quadrants:
#         # Calculate fat by percentage of white vs black pixels
#         fat_percentage = calculate_fat_percentage(quadrant)
#         total_fat += fat_percentage

#     # Find final fat percentage of the current steak by averaging all sections
#     final_fat = total_fat / len(quadrants)

#     # Find what steak type this was from the spreadsheet and add its avg to running average per steak type
#     steak_type = row['Grade Category']
#     total_fat_percentages[steak_type].append(final_fat)

# # Calculate final average fat percentage for each steak type
# avg_fat_percentages = {steak_type: sum(fats) / len(fats) if fats else 0 for steak_type, fats in total_fat_percentages.items()}

# # Output the average fat percentages
# for steak_type, avg_fat in avg_fat_percentages.items():
#     print(f"Average fat percentage for {steak_type}: {avg_fat:.2f}%")