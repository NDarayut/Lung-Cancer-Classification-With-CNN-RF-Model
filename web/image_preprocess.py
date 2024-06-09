import cv2
import numpy as np
import os


def preprocess(image_path):

    print(f"Image path: {image_path}")
    # Define class labels
    classes = {'Bengin case': 0, 'Malignant case': 1, 'Normal case': 2}

    # Define the size to which the image will be resized
    image_size = 256

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image
    image = cv2.resize(image, (image_size, image_size))

    # If needed, expand the dimensions to match model input (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=0)

    X = np.array(image)
    X = X / 255.

    return X






