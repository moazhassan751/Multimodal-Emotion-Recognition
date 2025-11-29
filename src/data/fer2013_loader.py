"""
FER2013 Dataset Loader

This module handles loading and preprocessing of the FER2013 facial emotion dataset.
Dataset contains ~35,000 grayscale 48x48 images with 7 emotion labels.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FER2013DataLoader:
    """
    Data loader for FER2013 dataset.
    
    Emotions:
        0: Angry
        1: Disgust
        2: Fear
        3: Happy
        4: Sad
        5: Surprise
        6: Neutral
    """
    
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    NUM_CLASSES = 7
    IMAGE_SIZE = 48
    
    def __init__(self, data_path: str = "data/fer2013"):
        """
        Initialize the FER2013 data loader.
        
        Args:
            data_path: Path to the FER2013 dataset directory
        """
        self.data_path = data_path
        self.csv_path = os.path.join(data_path, "fer2013.csv")
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load FER2013 data from CSV file.
        
        Returns:
            Tuple of (images, labels) as numpy arrays
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"FER2013 dataset not found at {self.csv_path}. "
                "Please download from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013"
            )
        
        logger.info(f"Loading FER2013 data from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        
        # Parse pixel values
        pixels = df['pixels'].apply(lambda x: np.fromstring(x, sep=' '))
        images = np.stack(pixels.values)
        images = images.reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
        
        # Normalize pixel values to [0, 1]
        images = images.astype('float32') / 255.0
        
        # Get labels
        labels = df['emotion'].values
        
        logger.info(f"Loaded {len(images)} images with shape {images.shape}")
        return images, labels
    
    def load_data_from_directory(self, directory: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from directory structure (alternative format).
        Expected structure: directory/emotion_name/*.png
        
        Args:
            directory: Path to the directory containing emotion subdirectories
            
        Returns:
            Tuple of (images, labels) as numpy arrays
        """
        images = []
        labels = []
        
        for emotion_idx, emotion_name in enumerate(self.EMOTIONS):
            emotion_dir = os.path.join(directory, emotion_name)
            if not os.path.exists(emotion_dir):
                logger.warning(f"Directory not found: {emotion_dir}")
                continue
                
            for img_file in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
                        images.append(img)
                        labels.append(emotion_idx)
                except Exception as e:
                    logger.warning(f"Error loading {img_path}: {e}")
                    
        images = np.array(images).reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
        images = images.astype('float32') / 255.0
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(images)} images from directory")
        return images, labels
    
    def prepare_data(
        self, 
        images: np.ndarray, 
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> dict:
        """
        Split data into train, validation, and test sets.
        
        Args:
            images: Image array
            labels: Label array
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train, val, and test data
        """
        # Convert labels to one-hot encoding
        labels_onehot = to_categorical(labels, num_classes=self.NUM_CLASSES)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels_onehot, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            random_state=random_state
        )
        
        logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def get_class_weights(self, labels: np.ndarray) -> dict:
        """
        Calculate class weights for imbalanced dataset.
        
        Args:
            labels: Array of class labels
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        return dict(enumerate(class_weights))
    
    @staticmethod
    def get_emotion_name(label_idx: int) -> str:
        """Get emotion name from label index."""
        return FER2013DataLoader.EMOTIONS[label_idx]
    
    @staticmethod
    def get_emotion_index(emotion_name: str) -> int:
        """Get label index from emotion name."""
        return FER2013DataLoader.EMOTIONS.index(emotion_name)


if __name__ == "__main__":
    # Test the data loader
    loader = FER2013DataLoader()
    
    try:
        images, labels = loader.load_data()
        data = loader.prepare_data(images, labels)
        print(f"Data loaded successfully!")
        print(f"Training samples: {data['X_train'].shape}")
        print(f"Validation samples: {data['X_val'].shape}")
        print(f"Test samples: {data['X_test'].shape}")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
