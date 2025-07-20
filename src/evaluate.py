from tensorflow.keras.models import load_model
from src.data_preprocessing import load_data
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model_path='deepfake_model.h5', data_dir='data/raw/', image_size=(224, 224)):
    """
    Evaluate the trained model on the test set.
    """
    model = load_model(model_path)
    
    # Load data
    images, labels = load_data(data_dir, image_size)
    
    # Make predictions
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Print evaluation metrics
    print("Classification Report:")
    print(classification_report(labels, predicted_labels))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, predicted_labels))

# Example call:
# evaluate_model('deepfake_model.h5', 'data/raw/')
