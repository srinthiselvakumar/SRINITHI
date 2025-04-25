import cv2

# ---------- Part 1: Read and Resize ----------
# Read the image
image1_path = r"C:\Users\hp\Downloads\image.jpg"
image1 = cv2.imread(image1_path)

# Resize the image
new_width = 400
new_height = 300
resized_image = cv2.resize(image1, (new_width, new_height))
cv2.imshow('Resized Image', resized_image)

# ---------- Part 2: Flip in Different Directions ----------
# Read the second image
image2_path = r"C:\Users\hp\Downloads\image.jpg"
image2 = cv2.imread(image2_path)

# Display the original
cv2.imshow("Original Image", image2)

# Flip the image
image_flipped_horz = cv2.flip(image2, 1)
image_flipped_vert = cv2.flip(image2, 0)
image_flipped_both = cv2.flip(image2, -1)

# Show the flipped images
cv2.imshow("Flipped Horizontally", image_flipped_horz)
cv2.imshow("Flipped Vertically", image_flipped_vert)
cv2.imshow("Flipped Both", image_flipped_both)

# ---------- Part 3: Cropping ----------
# Crop region from the first image
x1, y1 = 100, 50  # Top-left corner
x2, y2 = 300, 200  # Bottom-right corner
cropped_image = image1[y1:y2, x1:x2]
cv2.imshow('Cropped Image', cropped_image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
