import cv2
import pandas as pd
import numpy as np

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

def process_image_to_grayscale(image_path):
    image = cv2.imread(image_path)
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayimage

def calculate_fat_percentage(image):
    # Assuming the image is already cropped to only contain the steak.
    
    # Define the ROI as the central part of the image
    height, width = image.shape
    central_region = image[int(height*0.25):int(height*0.75), int(width*0.25):int(width*0.75)]
    
    # Divide the central region into quadrants
    # Top-left, Top-right, Bottom-left, Bottom-right quadrants
    h_centre, w_centre = central_region.shape[0] // 2, central_region.shape[1] // 2
    quadrants = [
        central_region[:h_centre, :w_centre],
        central_region[:h_centre, w_centre:],
        central_region[h_centre:, :w_centre],
        central_region[h_centre:, w_centre:]
    ]
    
    # Calculate the fat percentage in each quadrant
    fat_percentages = []
    for quadrant in quadrants:
        # Count white pixels (you may need to adjust the threshold)
        white_pixels = np.sum(quadrant > 200)  # Assuming fat is the whiter part
        total_pixels = quadrant.size
        
        # Calculate the fat percentage for the current quadrant
        fat_percentage = (white_pixels / total_pixels) * 100
        fat_percentages.append(fat_percentage)
    
    # Calculate the final fat percentage by averaging all quadrants
    final_fat_percentage = np.mean(fat_percentages)
    return final_fat_percentage

grayimage = process_image_to_grayscale('./Path1 Challenge Training Images/00007999-1.tif')
#cv2.imshow("Display window", grayimage)
#cv2.waitKey(0)
fat_percentage = calculate_fat_percentage(grayimage)
print(f"Fat Percentage: {fat_percentage}%")

# 00000114-1.tif	114	306.7	Select - 0.43524184149184153%
# 00007536-1.tif	7536	396.7	Select - 0.7748761655011654%

# 00000080-2.tif	80	400	Low Choice - 7.18331148018648%
# 00007999-1.tif	7999	496.7	Low Choice


# 00000221-1.tif	221	500	Upper 2/3 Choice
# 00008034-2.tif	8034	696.7	Upper 2/3 Choice

# 00007077-2.tif	7077	700	Prime
# 00000104-1.tif	104	866.7	Prime

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