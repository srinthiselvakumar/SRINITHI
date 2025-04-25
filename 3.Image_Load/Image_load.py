from google.colab import files
import cv2
from IPython.display import Image, display

uploaded = files.upload()

filename = list(uploaded.keys())[0]

image = cv2.imread(filename)

if image is None:
  print("Error: Image not found or path is incorrect")
else:
  cv2.imwrite('temp_image.jpg', image)
  display(Image('temp_image.jpg'))
