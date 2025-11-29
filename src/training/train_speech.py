"""
Speech Emotion Model Training Script

This module handles training of the speech emotion recognition model.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from datetime import datetime
from typing import Optional, Dict
import logging
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ravdess_loader import RAVDESSDataLoader
from models.speech_model import SpeechEmotionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechTrainer:
    """Trainer class for speech emotion recognition model."""
    
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
    
    def setup_data(self, feature_type: str = 'mfcc') -> Dict:
        """
        Setup data loading and preprocessing.
        
        Args:
            feature_type: Type of audio features ('mfcc' or 'mel_spectrogram')
            
        Returns:
            Dictionary with train, val, test data
        """
        logger.info("Loading RAVDESS dataset...")
        
        ravdess_config = self.config['data']['ravdess']
        
        self.data_loader = RAVDESSDataLoader(
            data_path=ravdess_config['path'],
            sample_rate=ravdess_config['sample_rate'],
            duration=ravdess_config['duration'],
            n_mfcc=ravdess_config['n_mfcc'],
            n_mels=ravdess_config['n_mels']
        )
        
        features, labels = self.data_loader.load_data(feature_type=feature_type)
        
        data = self.data_loader.prepare_data(
            features, labels,
            test_size=ravdess_config['test_split'],
            val_size=ravdess_config['val_split']
        )
        
        logger.info(f"Data loaded: Train={data['X_train'].shape[0]}, "
                   f"Val={data['X_val'].shape[0]}, Test={data['X_test'].shape[0]}")
        logger.info(f"Feature shape: {data['X_train'].shape[1:]}")
        
        return data
    
    def setup_model(self, input_shape: Optional[tuple] = None) -> SpeechEmotionModel:
        """
        Setup the speech emotion model.
        
        Args:
            input_shape: Shape of input features (optional, auto-detected from data)
            
        Returns:
            Configured speech emotion model
        """
        logger.info("Setting up speech emotion model...")
        
        model_config = self.config['models']['speech']
        training_config = self.config['training']['speech']
        
        if input_shape is None:
            input_shape = tuple(model_config['input_shape'])
        
        self.model = SpeechEmotionModel(
            input_shape=input_shape,
            num_classes=self.config['data']['ravdess']['num_classes'],
            architecture=model_config['architecture'],
            lstm_units=model_config['lstm_units'],
            dropout_rate=model_config['dropout_rate']
        )
        
        self.model.build()
        self.model.compile(
            optimizer=training_config['optimizer'],
            learning_rate=training_config['learning_rate']
        )
        
        return self.model
    
    def setup_callbacks(self, model_name: str = "speech_emotion") -> list:
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
        
        training_config = self.config['training']['speech']
        
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
    
    def augment_audio_features(self, features: np.ndarray, labels: np.ndarray) -> tuple:
        """
        Apply data augmentation to audio features.
        
        Args:
            features: Input features array
            labels: Labels array
            
        Returns:
            Augmented features and labels
        """
        augmented_features = []
        augmented_labels = []
        
        for feature, label in zip(features, labels):
            # Original
            augmented_features.append(feature)
            augmented_labels.append(label)
            
            # Time shift
            shift = np.random.randint(-5, 5)
            shifted = np.roll(feature, shift, axis=1)
            augmented_features.append(shifted)
            augmented_labels.append(label)
            
            # Add noise
            noise = np.random.randn(*feature.shape) * 0.01
            noisy = feature + noise
            augmented_features.append(noisy)
            augmented_labels.append(label)
            
            # Scale
            scale = np.random.uniform(0.9, 1.1)
            scaled = feature * scale
            augmented_features.append(scaled)
            augmented_labels.append(label)
        
        return np.array(augmented_features), np.array(augmented_labels)
    
    def train(
        self,
        data: Optional[Dict] = None,
        use_augmentation: bool = True,
        use_class_weights: bool = True
    ):
        """
        Train the speech emotion model.
        
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
            input_shape = data['X_train'].shape[1:]
            self.setup_model(input_shape=input_shape)
        
        # Get training config
        training_config = self.config['training']['speech']
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Apply augmentation if requested
        X_train = data['X_train']
        y_train = data['y_train']
        
        if use_augmentation:
            logger.info("Applying data augmentation...")
            X_train, y_train = self.augment_audio_features(X_train, y_train)
            logger.info(f"Augmented training samples: {len(X_train)}")
        
        # Calculate class weights if requested
        class_weights = None
        if use_class_weights:
            y_train_labels = np.argmax(y_train, axis=1)
            class_weights = self.data_loader.get_class_weights(y_train_labels)
            logger.info(f"Using class weights: {class_weights}")
        
        # Train
        logger.info("Starting training...")
        self.history = self.model.model.fit(
            X_train, y_train,
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
        for i, emotion in enumerate(RAVDESSDataLoader.EMOTIONS):
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
                self.config['paths']['models']['speech'],
                'speech_emotion_model.keras'
            )
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.model.save(path)
        logger.info(f"Model saved to {path}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Speech Emotion Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--feature-type', type=str, default='mfcc',
                       choices=['mfcc', 'mel_spectrogram'],
                       help='Type of audio features')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-class-weights', action='store_true',
                       help='Disable class weights')
    args = parser.parse_args()
    
    # Create trainer
    trainer = SpeechTrainer(config_path=args.config)
    
    # Load data
    data = trainer.setup_data(feature_type=args.feature_type)
    
    # Setup model
    input_shape = data['X_train'].shape[1:]
    trainer.setup_model(input_shape=input_shape)
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
