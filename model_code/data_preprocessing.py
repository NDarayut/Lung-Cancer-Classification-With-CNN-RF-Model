import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# Define the base directory where images are stored
data_dir = "path\\to\\folder"

# Define class labels
classes = {'Bengin case', 'Malignant case', 'Normal case'}

# Initialize empty lists for images and labels
images = []
labels = []

image_size = 224

# Loop through each class directory
for class_name in classes:
    # Get the path to the current class directory
    class_dir = os.path.join(data_dir, class_name)

    # Loop through each image file in the class directory
    for filename in os.listdir(class_dir):
        # Get the full path to the image file
        image_path = os.path.join(class_dir, filename)

        # Load the image using OpenCV or PIL (adjust as needed)
        image = cv2.imread(image_path)  # Replace with your preferred image loading function
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size))
        # Append the image and its corresponding label to the lists
        images.append(image)
        labels.append(class_name)

label_dict = {"Bengin case": 0, "Malignant case": 1, "Normal case": 2}
encoded_labels = [label_dict[label] for label in labels]

# Generate a permutation of indices
permutation = np.random.permutation(len(encoded_labels))

X = np.array(images)
X = X / 255.
Y = np.array(encoded_labels)

X = X[permutation]
Y = Y[permutation]




