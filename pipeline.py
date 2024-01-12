import cv2
import numpy as np


fingerprint_image = cv2.imread(
    "path/to/your/fingerprint_image.jpg", 0)

blurred = cv2.GaussianBlur(fingerprint_image, (5, 5), 0)

thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)
gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)

erosion = cv2.erode(gradient, kernel, iterations=1)

cv2.imshow("Processed Fingerprint", erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
