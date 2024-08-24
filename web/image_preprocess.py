import cv2
import numpy as np

def preprocess(image_path):

    # Define the size to which the image will be resized
    image_size = 224

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image
    image = cv2.resize(image, (image_size, image_size))

    # If needed, expand the dimensions to match model input (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=0)

    # convert the pixel value into a numpy array for faster processing
    X = np.array(image)
    # normalize the pixel scale to 0-1
    X = X / 255.

    return X






