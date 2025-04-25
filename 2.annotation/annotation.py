from google.colab import files
import cv2
import numpy as np
from IPython.display import Image, display

uploaded = files.upload()
filename = list(uploaded.keys())[0]
img = cv2.imread(filename)

if img is None:
    print("Error: Could not open or read the image.")
else:
    cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 3)  # Red line, thickness 3
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 2) # Green rectangle, thickness 2
    cv2.circle(img, (300, 150), 50, (255, 0, 0), -1)
    cv2.ellipse(img, (450, 250), (50, 30), 0, 0, 360, (255, 255, 0), 2) # Cyan ellipse, thickness 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'OpenCV Annotation', (10, 50), font, 1, (255, 0, 255), 2, cv2.LINE_AA)  # Purple text
    cv2.imwrite('annotated_image.jpg', img)
    display(Image('annotated_image.jpg'))
