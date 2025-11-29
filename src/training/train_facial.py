"""
Facial Emotion Model Training Script

This module handles training of the facial emotion recognition model.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from typing import Optional, Dict
import logging
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fer2013_loader import FER2013DataLoader
from models.facial_model import FacialEmotionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FacialTrainer:
    """Trainer class for facial emotion recognition model."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.data_loader = None
        self.history = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_data(self) -> Dict:
        """
        Setup data loading and preprocessing.
        
        Returns:
            Dictionary with train, val, test data
        """
        logger.info("Loading FER2013 dataset...")
        
        self.data_loader = FER2013DataLoader(
            data_path=self.config['data']['fer2013']['path']
        )
        
        try:
            images, labels = self.data_loader.load_data()
        except FileNotFoundError:
            # Try loading from directory structure
            logger.info("CSV not found, trying directory structure...")
            train_dir = os.path.join(self.config['data']['fer2013']['path'], 'train')
            images, labels = self.data_loader.load_data_from_directory(train_dir)
        
        data = self.data_loader.prepare_data(
            images, labels,
            test_size=self.config['data']['fer2013']['test_split'],
            val_size=self.config['data']['fer2013']['val_split']
        )
        
        logger.info(f"Data loaded: Train={data['X_train'].shape[0]}, "
                   f"Val={data['X_val'].shape[0]}, Test={data['X_test'].shape[0]}")
        
        return data
    
    def setup_model(self) -> FacialEmotionModel:
        """
        Setup the facial emotion model.
        
        Returns:
            Configured facial emotion model
        """
        logger.info("Setting up facial emotion model...")
        
        model_config = self.config['models']['facial']
        training_config = self.config['training']['facial']
        
        self.model = FacialEmotionModel(
            input_shape=tuple(model_config['input_shape']),
            num_classes=self.config['data']['fer2013']['num_classes'],
            architecture=model_config['architecture'],
            dropout_rate=model_config['dropout_rate'],
            l2_reg=model_config['l2_regularization']
        )
        
        self.model.build()
        self.model.compile(
            optimizer=training_config['optimizer'],
            learning_rate=training_config['learning_rate']
        )
        
        return self.model
    
    def setup_callbacks(self, model_name: str = "facial_emotion") -> list:
        """
        Setup training callbacks.
        
        Args:
            model_name: Name for saved models
            
        Returns:
            List of Keras callbacks
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        checkpoint_dir = self.config['paths']['checkpoints']
        log_dir = self.config['paths']['logs']
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        training_config = self.config['training']['facial']
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f"{model_name}_best.keras"),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Save latest model
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f"{model_name}_latest.keras"),
                verbose=0
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=training_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=training_config['reduce_lr_factor'],
                patience=training_config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=os.path.join(log_dir, f"{model_name}_{timestamp}"),
                histogram_freq=1
            ),
            
            # CSV logging
            CSVLogger(
                os.path.join(log_dir, f"{model_name}_{timestamp}.csv")
            )
        ]
        
        return callbacks
    
    def setup_data_augmentation(self) -> ImageDataGenerator:
        """
        Setup data augmentation for training.
        
        Returns:
            ImageDataGenerator for augmentation
        """
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )
        return datagen
    
    def train(
        self,
        data: Optional[Dict] = None,
        use_augmentation: bool = True,
        use_class_weights: bool = True
    ):
        """
        Train the facial emotion model.
        
        Args:
            data: Optional pre-loaded data dictionary
            use_augmentation: Whether to use data augmentation
            use_class_weights: Whether to use class weights for imbalanced data
        """
        # Setup data if not provided
        if data is None:
            data = self.setup_data()
        
        # Setup model if not already done
        if self.model is None:
            self.setup_model()
        
        # Get training config
        training_config = self.config['training']['facial']
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Calculate class weights if requested
        class_weights = None
        if use_class_weights:
            y_train_labels = np.argmax(data['y_train'], axis=1)
            class_weights = self.data_loader.get_class_weights(y_train_labels)
            logger.info(f"Using class weights: {class_weights}")
        
        # Train with or without augmentation
        if use_augmentation:
            logger.info("Training with data augmentation...")
            datagen = self.setup_data_augmentation()
            datagen.fit(data['X_train'])
            
            self.history = self.model.model.fit(
                datagen.flow(data['X_train'], data['y_train'], 
                           batch_size=training_config['batch_size']),
                steps_per_epoch=len(data['X_train']) // training_config['batch_size'],
                epochs=training_config['epochs'],
                validation_data=(data['X_val'], data['y_val']),
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
        else:
            logger.info("Training without data augmentation...")
            self.history = self.model.model.fit(
                data['X_train'], data['y_train'],
                batch_size=training_config['batch_size'],
                epochs=training_config['epochs'],
                validation_data=(data['X_val'], data['y_val']),
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
        
        return self.history
    
    def evaluate(self, data: Dict) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            data: Data dictionary with test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model on test data...")
        
        results = self.model.model.evaluate(
            data['X_test'], data['y_test'],
            verbose=1
        )
        
        metrics = {
            'test_loss': results[0],
            'test_accuracy': results[1]
        }
        
        # Get predictions for detailed metrics
        predictions = self.model.model.predict(data['X_test'])
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(data['y_test'], axis=1)
        
        # Calculate per-class accuracy
        for i, emotion in enumerate(FER2013DataLoader.EMOTIONS):
            mask = y_true == i
            if mask.sum() > 0:
                acc = (y_pred[mask] == y_true[mask]).mean()
                metrics[f'accuracy_{emotion}'] = acc
        
        logger.info(f"Test Results: Loss={metrics['test_loss']:.4f}, "
                   f"Accuracy={metrics['test_accuracy']:.4f}")
        
        return metrics
    
    def save_model(self, path: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        if path is None:
            path = os.path.join(
                self.config['paths']['models']['facial'],
                'facial_emotion_model.keras'
            )
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.model.save(path)
        logger.info(f"Model saved to {path}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Facial Emotion Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-class-weights', action='store_true',
                       help='Disable class weights')
    args = parser.parse_args()
    
    # Create trainer
    trainer = FacialTrainer(config_path=args.config)
    
    # Load data
    data = trainer.setup_data()
    
    # Setup model
    trainer.setup_model()
    trainer.model.summary()
    
    # Train
    trainer.train(
        data=data,
        use_augmentation=not args.no_augmentation,
        use_class_weights=not args.no_class_weights
    )
    
    # Evaluate
    metrics = trainer.evaluate(data)
    print("\nFinal Test Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save model
    trainer.save_model()


if __name__ == "__main__":
    main()
