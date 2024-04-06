import cv2
import numpy as np

# Read the image in grayscale
gray_img = cv2.imread('./Path1 Challenge Images for Validation/00001001-2.tif', cv2.IMREAD_GRAYSCALE)

# Apply thresholding to segment the meat portion
_, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

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

# Display the result
cv2.imshow("Meat Portion", meat_portion)
cv2.waitKey(0)
cv2.destroyAllWindows()
