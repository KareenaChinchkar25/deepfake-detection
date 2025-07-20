import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

def load_data(data_dir, image_size=(224, 224)):
    """
    Loads images and corresponding labels from the given directory.
    """
    images = []
    labels = []
    label_map = {'real': 0, 'fake': 1}  # Assuming your subfolders are named 'real' and 'fake'
    
    for label in label_map:
        label_dir = os.path.join(data_dir, label)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)  # Resize to model input size
            image = img_to_array(image) / 255.0  # Normalize the image
            images.append(image)
            labels.append(label_map[label])
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def split_data(images, labels, test_size=0.2):
    """
    Splits data into training and testing sets.
    """
    return train_test_split(images, labels, test_size=test_size, random_state=42)

def save_processed_data(images, labels, save_dir='data/processed/'):
    """
    Saves the processed images and labels to the specified directory.
    """
    np.save(os.path.join(save_dir, 'images.npy'), images)
    np.save(os.path.join(save_dir, 'labels.npy'), labels)
