import cv2

img = cv2.imread('./Path1 Challenge Images for Validation/00001001-2.tif')
grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresholded_img = cv2.threshold(grayimage, 90, 255, cv2.THRESH_BINARY)

cv2.imshow("Display window", thresholded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()