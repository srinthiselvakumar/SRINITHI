!pip install google-colab

import cv2
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt

# Prompt the user to upload an image
uploaded = files.upload()

# Get the filename of the uploaded image
filename = list(uploaded.keys())[0]

# Load the image
img = cv2.imread(filename, cv2.IMREAD_COLOR)

# Check if the image was loaded successfully
if img is None:
    print("Error: Could not load image. Please check the file and try again.")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Fourier Transform
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # Display using matplotlib 
    plt.figure(figsize=(8, 6))
    plt.imshow(np.uint8(magnitude_spectrum / np.max(magnitude_spectrum) * 255), cmap='gray')
    plt.title("Fourier Magnitude Spectrum")
    plt.show()

    # 2. Hough Transform for line detection
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
    hough_img = img.copy()
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(hough_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(hough_img, cv2.COLOR_BGR2RGB)) 
    plt.title("Hough Line Detection")
    plt.show()

    # 3. ORB Feature Detection
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray, None)
    img_orb = cv2.drawKeypoints(img, kp1, None, color=(0, 255, 0), flags=0)

    # Display using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
    plt.title("ORB Features")
    plt.show()

    # 4. Feature Matching and Alignment (using same image for demo)
    kp2, des2 = orb.detectAndCompute(gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    match_img = cv2.drawMatches(img, kp1, img, kp2, matches[:10], None, flags=2)

    # Display using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title("Feature Matching")
    plt.show()

    # 5. Simple Cloning based on ROI and matched location
    clone = img.copy()
    src_region = img[50:150, 50:150]
    clone[200:300, 200:300] = src_region

    # Display using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)) 
    plt.title("Image Cloning")
    plt.show()
