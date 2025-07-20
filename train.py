import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from src.data_preprocessing import load_data, split_data, save_processed_data
from src.cnn_model import create_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def setup_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured {len(gpus)} GPU(s) for training")
        except RuntimeError as e:
            print(e)

def create_callbacks():
    """Create training callbacks"""
    callbacks = [
        ModelCheckpoint(
            "models/best_model.h5",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            histogram_freq=1
        )
    ]
    return callbacks

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and generate reports"""
    print("\nEvaluating model performance...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=['Real', 'Fake']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('reports/confusion_matrix.png')
    plt.close()
    
    print("Evaluation metrics saved in reports/ directory")

def train_model(data_dir='data/raw/', image_size=(224, 224), batch_size=32, epochs=50):
    """
    Enhanced CNN model training for deepfake detection
    
    Args:
        data_dir: Path to raw image data directory
        image_size: Tuple of (height, width) for image resizing
        batch_size: Batch size for training
        epochs: Maximum number of training epochs
    """
    # Setup directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Configure GPU
    setup_gpu()
    
    try:
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        images, labels = load_data(data_dir, image_size)
        print(f"Loaded {len(images)} images with {len(labels)} labels")
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(images, labels)
        print(f"\nData split:")
        print(f"Training: {X_train.shape} images, {y_train.shape} labels")
        print(f"Testing: {X_test.shape} images, {y_test.shape} labels")
        
        # Save processed data
        save_processed_data(X_train, y_train)
        print("\nProcessed data saved successfully")
        
        # Create model
        model = create_model(input_shape=(image_size[0], image_size[1], 3))
        model.summary()
        
        # Training callbacks
        callbacks = create_callbacks()
        
        # Train model
        print("\nStarting model training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = "models/deepfake_model.h5"
        model.save(final_model_path)
        print(f"\nFinal model saved to {os.path.abspath(final_model_path)}")
        
        # Evaluate model
        evaluate_model(model, X_test, y_test)
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.savefig('reports/training_history.png')
        plt.close()
        print("\nTraining completed successfully")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    # Training configuration
    config = {
        'data_dir': 'data/raw/',
        'image_size': (224, 224),
        'batch_size': 32,
        'epochs': 30
    }
    
    print("Starting Deepfake Detection Model Training")
    print("="*50)
    print(f"Configuration:\n{config}")
    print("="*50)
    
    train_model(**config)