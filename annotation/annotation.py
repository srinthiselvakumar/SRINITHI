import cv2
import numpy as np
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
  image_path = fn # C:\Users\hp\Download

try:
  # Load the image
  img = cv2.imdecode(np.frombuffer(uploaded[image_path], np.uint8), cv2.IMREAD_COLOR)

  if img is None:
    print("Error: Could not open or read the image.")
  else:
    # Line
    cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 3)  # Red line, thickness 3
    # Rectangle
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 2) # Green rectangle, thickness 2
    # Filled Circle
    cv2.circle(img, (300, 150), 50, (255, 0, 0), -1) # Blue filled circle
    # Ellipse
    cv2.ellipse(img, (450, 250), (50, 30), 0, 0, 360, (255, 255, 0), 2) # Cyan ellipse, thickness 2
    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'OpenCV Annotation', (10, 50), font, 1, (255, 0, 255), 2, cv2.LINE_AA)  # Purple text

    # Display the image with annotations (Colab specific)
    from google.colab.patches import cv2_imshow
    cv2_imshow(img)

    # Save the annotated image (optional)
    #cv2.imwrite('annotated_image.jpg', img)

except Exception as e:
  print(f"An error occurred: {e}")
