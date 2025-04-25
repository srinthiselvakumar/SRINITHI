import cv2
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from google.colab.patches import cv2_imshow
import io

# File upload widget for multiple images
uploader = widgets.FileUpload(
    accept='image/*',  # Only image files
    multiple=True      # Allow multiple file uploads
)

# Display the upload widget
display(uploader)

# Function to process all uploaded images
def process_images(change):
    if uploader.value:
        for filename, filedata in uploader.value.items():
            try:
                # Convert uploaded file content to OpenCV image
                image_stream = io.BytesIO(filedata['content'])
                file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"❌ Failed to decode image: {filename}")
                    continue

                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Fourier Transform
                dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)
                magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
                normalized_spectrum = np.uint8(magnitude_spectrum / np.max(magnitude_spectrum) * 255)

                # Display results
                print(f"\n📷 Fourier Magnitude Spectrum for: {filename}")
                cv2_imshow(normalized_spectrum)

            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")
    else:
        print("Please upload at least one image.")

# Observe the upload widget for new file uploads
uploader.observe(process_images, names='value')
