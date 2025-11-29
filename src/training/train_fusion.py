"""
Multimodal Fusion Model Training Script

This module handles training of the multimodal fusion model that combines
facial and speech emotion recognition.
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
from typing import Optional, Dict, Tuple
import logging
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fer2013_loader import FER2013DataLoader
from data.ravdess_loader import RAVDESSDataLoader
from models.facial_model import FacialEmotionModel
from models.speech_model import SpeechEmotionModel
from models.fusion_model import MultimodalFusionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionTrainer:
    """Trainer class for multimodal fusion model."""
    
    # Mapping between FER2013 and RAVDESS emotions
    EMOTION_MAPPING = {
        'FER2013': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
        'RAVDESS': ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised'],
        'Common': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    }
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.facial_model = None
        self.speech_model = None
        self.fusion_model = None
        self.history = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_pretrained_models(
        self,
        facial_model_path: Optional[str] = None,
        speech_model_path: Optional[str] = None
    ):
        """
        Load pre-trained facial and speech models.
        
        Args:
            facial_model_path: Path to pre-trained facial model
            speech_model_path: Path to pre-trained speech model
        """
        logger.info("Loading pre-trained models...")
        
        # Load facial model
        if facial_model_path is None:
            facial_model_path = os.path.join(
                self.config['paths']['models']['facial'],
                'facial_emotion_model.keras'
            )
        
        if os.path.exists(facial_model_path):
            self.facial_model = tf.keras.models.load_model(facial_model_path)
            logger.info(f"Loaded facial model from {facial_model_path}")
        else:
            logger.warning(f"Facial model not found at {facial_model_path}")
            # Build a new one
            model = FacialEmotionModel(
                input_shape=tuple(self.config['models']['facial']['input_shape']),
                num_classes=self.config['data']['fer2013']['num_classes']
            )
            model.build()
            self.facial_model = model.model
        
        # Load speech model
        if speech_model_path is None:
            speech_model_path = os.path.join(
                self.config['paths']['models']['speech'],
                'speech_emotion_model.keras'
            )
        
        if os.path.exists(speech_model_path):
            self.speech_model = tf.keras.models.load_model(speech_model_path)
            logger.info(f"Loaded speech model from {speech_model_path}")
        else:
            logger.warning(f"Speech model not found at {speech_model_path}")
            # Build a new one
            model = SpeechEmotionModel(
                input_shape=tuple(self.config['models']['speech']['input_shape']),
                num_classes=self.config['data']['ravdess']['num_classes']
            )
            model.build()
            self.speech_model = model.model
    
    def get_feature_extractors(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """
        Get feature extractors from pre-trained models.
        
        Returns:
            Tuple of (facial_extractor, speech_extractor)
        """
        # Get layer before final classification
        # Facial model feature extractor
        facial_extractor = tf.keras.Model(
            inputs=self.facial_model.input,
            outputs=self.facial_model.layers[-3].output  # Before final dropout and dense
        )
        
        # Speech model feature extractor
        speech_extractor = tf.keras.Model(
            inputs=self.speech_model.input,
            outputs=self.speech_model.layers[-3].output
        )
        
        # Freeze extractors
        for layer in facial_extractor.layers:
            layer.trainable = False
        for layer in speech_extractor.layers:
            layer.trainable = False
        
        return facial_extractor, speech_extractor
    
    def prepare_paired_data(
        self,
        facial_data: Dict,
        speech_data: Dict,
        num_samples: int = 1000
    ) -> Dict:
        """
        Prepare paired data for fusion training.
        
        Since FER2013 and RAVDESS have different samples, we create
        synthetic pairs by matching emotion labels.
        
        Args:
            facial_data: Facial data dictionary
            speech_data: Speech data dictionary
            num_samples: Number of paired samples to generate
            
        Returns:
            Dictionary with paired training data
        """
        logger.info("Preparing paired multimodal data...")
        
        # Map emotions to common set
        common_emotions = self.EMOTION_MAPPING['Common']
        
        # Get indices for each emotion in both datasets
        facial_labels = np.argmax(facial_data['X_train'], axis=1) if len(facial_data['y_train'].shape) > 1 else facial_data['y_train']
        facial_labels = np.argmax(facial_data['y_train'], axis=1)
        
        speech_labels = np.argmax(speech_data['y_train'], axis=1)
        
        # Map RAVDESS labels to common emotions
        ravdess_to_common = {
            0: 6,  # Neutral -> Neutral
            1: 6,  # Calm -> Neutral (closest)
            2: 3,  # Happy -> Happy
            3: 4,  # Sad -> Sad
            4: 0,  # Angry -> Angry
            5: 2,  # Fearful -> Fear
            6: 1,  # Disgust -> Disgust
            7: 5,  # Surprised -> Surprise
        }
        
        speech_labels_mapped = np.array([ravdess_to_common[l] for l in speech_labels])
        
        # Create paired samples
        paired_facial = []
        paired_speech = []
        paired_labels = []
        
        for emotion_idx in range(len(common_emotions)):
            # Find samples with this emotion in both datasets
            facial_mask = facial_labels == emotion_idx
            speech_mask = speech_labels_mapped == emotion_idx
            
            facial_indices = np.where(facial_mask)[0]
            speech_indices = np.where(speech_mask)[0]
            
            if len(facial_indices) == 0 or len(speech_indices) == 0:
                continue
            
            # Sample pairs
            n_pairs = min(num_samples // len(common_emotions), len(facial_indices), len(speech_indices))
            
            for _ in range(n_pairs):
                fi = np.random.choice(facial_indices)
                si = np.random.choice(speech_indices)
                
                paired_facial.append(facial_data['X_train'][fi])
                paired_speech.append(speech_data['X_train'][si])
                
                # One-hot encode
                label = np.zeros(len(common_emotions))
                label[emotion_idx] = 1
                paired_labels.append(label)
        
        # Convert to arrays
        paired_facial = np.array(paired_facial)
        paired_speech = np.array(paired_speech)
        paired_labels = np.array(paired_labels)
        
        # Shuffle
        indices = np.random.permutation(len(paired_facial))
        paired_facial = paired_facial[indices]
        paired_speech = paired_speech[indices]
        paired_labels = paired_labels[indices]
        
        # Split into train/val/test
        n_train = int(0.8 * len(paired_facial))
        n_val = int(0.1 * len(paired_facial))
        
        data = {
            'facial_train': paired_facial[:n_train],
            'speech_train': paired_speech[:n_train],
            'y_train': paired_labels[:n_train],
            'facial_val': paired_facial[n_train:n_train+n_val],
            'speech_val': paired_speech[n_train:n_train+n_val],
            'y_val': paired_labels[n_train:n_train+n_val],
            'facial_test': paired_facial[n_train+n_val:],
            'speech_test': paired_speech[n_train+n_val:],
            'y_test': paired_labels[n_train+n_val:]
        }
        
        logger.info(f"Paired data: Train={len(data['y_train'])}, "
                   f"Val={len(data['y_val'])}, Test={len(data['y_test'])}")
        
        return data
    
    def extract_features(
        self,
        facial_extractor: tf.keras.Model,
        speech_extractor: tf.keras.Model,
        data: Dict
    ) -> Dict:
        """
        Extract features from paired data using pre-trained models.
        
        Args:
            facial_extractor: Facial feature extractor
            speech_extractor: Speech feature extractor
            data: Paired data dictionary
            
        Returns:
            Dictionary with extracted features
        """
        logger.info("Extracting features...")
        
        feature_data = {}
        
        for split in ['train', 'val', 'test']:
            facial_key = f'facial_{split}'
            speech_key = f'speech_{split}'
            
            # Extract facial features
            facial_features = facial_extractor.predict(data[facial_key], verbose=0)
            
            # Extract speech features
            speech_features = speech_extractor.predict(data[speech_key], verbose=0)
            
            feature_data[f'facial_features_{split}'] = facial_features
            feature_data[f'speech_features_{split}'] = speech_features
            feature_data[f'y_{split}'] = data[f'y_{split}']
        
        logger.info(f"Facial features shape: {feature_data['facial_features_train'].shape}")
        logger.info(f"Speech features shape: {feature_data['speech_features_train'].shape}")
        
        return feature_data
    
    def setup_model(self, facial_dim: int, speech_dim: int) -> MultimodalFusionModel:
        """
        Setup the fusion model.
        
        Args:
            facial_dim: Dimension of facial features
            speech_dim: Dimension of speech features
            
        Returns:
            Configured fusion model
        """
        logger.info("Setting up fusion model...")
        
        fusion_config = self.config['models']['fusion']
        training_config = self.config['training']['fusion']
        
        self.fusion_model = MultimodalFusionModel(
            facial_feature_dim=facial_dim,
            speech_feature_dim=speech_dim,
            num_classes=fusion_config['num_classes'],
            fusion_method=fusion_config['method'],
            dense_units=fusion_config['dense_units'],
            dropout_rate=fusion_config['dropout_rate']
        )
        
        self.fusion_model.build()
        self.fusion_model.compile(
            optimizer=training_config['optimizer'],
            learning_rate=training_config['learning_rate']
        )
        
        return self.fusion_model
    
    def setup_callbacks(self, model_name: str = "fusion_emotion") -> list:
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
        
        training_config = self.config['training']['fusion']
        
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f"{model_name}_best.keras"),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f"{model_name}_latest.keras"),
                verbose=0
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=training_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=training_config['reduce_lr_factor'],
                patience=training_config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(log_dir, f"{model_name}_{timestamp}"),
                histogram_freq=1
            ),
            CSVLogger(
                os.path.join(log_dir, f"{model_name}_{timestamp}.csv")
            )
        ]
        
        return callbacks
    
    def train(self, feature_data: Dict):
        """
        Train the fusion model.
        
        Args:
            feature_data: Dictionary with extracted features
        """
        # Setup model
        facial_dim = feature_data['facial_features_train'].shape[-1]
        speech_dim = feature_data['speech_features_train'].shape[-1]
        
        if self.fusion_model is None:
            self.setup_model(facial_dim, speech_dim)
        
        # Get training config
        training_config = self.config['training']['fusion']
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train
        logger.info("Training fusion model...")
        self.history = self.fusion_model.model.fit(
            [feature_data['facial_features_train'], 
             feature_data['speech_features_train']],
            feature_data['y_train'],
            batch_size=training_config['batch_size'],
            epochs=training_config['epochs'],
            validation_data=(
                [feature_data['facial_features_val'],
                 feature_data['speech_features_val']],
                feature_data['y_val']
            ),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, feature_data: Dict) -> Dict:
        """
        Evaluate the fusion model.
        
        Args:
            feature_data: Dictionary with extracted features
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating fusion model...")
        
        results = self.fusion_model.model.evaluate(
            [feature_data['facial_features_test'],
             feature_data['speech_features_test']],
            feature_data['y_test'],
            verbose=1
        )
        
        metrics = {
            'test_loss': results[0],
            'test_accuracy': results[1]
        }
        
        # Get predictions
        predictions = self.fusion_model.model.predict(
            [feature_data['facial_features_test'],
             feature_data['speech_features_test']]
        )
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(feature_data['y_test'], axis=1)
        
        # Per-class accuracy
        common_emotions = self.EMOTION_MAPPING['Common']
        for i, emotion in enumerate(common_emotions):
            mask = y_true == i
            if mask.sum() > 0:
                acc = (y_pred[mask] == y_true[mask]).mean()
                metrics[f'accuracy_{emotion}'] = acc
        
        logger.info(f"Test Results: Loss={metrics['test_loss']:.4f}, "
                   f"Accuracy={metrics['test_accuracy']:.4f}")
        
        return metrics
    
    def save_model(self, path: Optional[str] = None):
        """
        Save the trained fusion model.
        
        Args:
            path: Path to save the model
        """
        if path is None:
            path = os.path.join(
                self.config['paths']['models']['fusion'],
                'fusion_emotion_model.keras'
            )
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fusion_model.model.save(path)
        logger.info(f"Model saved to {path}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Fusion Emotion Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--facial-model', type=str, default=None,
                       help='Path to pre-trained facial model')
    parser.add_argument('--speech-model', type=str, default=None,
                       help='Path to pre-trained speech model')
    parser.add_argument('--num-pairs', type=int, default=2000,
                       help='Number of paired samples to generate')
    args = parser.parse_args()
    
    # Create trainer
    trainer = FusionTrainer(config_path=args.config)
    
    # Load pre-trained models
    trainer.load_pretrained_models(
        facial_model_path=args.facial_model,
        speech_model_path=args.speech_model
    )
    
    # Get feature extractors
    facial_extractor, speech_extractor = trainer.get_feature_extractors()
    
    # Load individual datasets
    logger.info("Loading datasets...")
    
    facial_loader = FER2013DataLoader(trainer.config['data']['fer2013']['path'])
    speech_loader = RAVDESSDataLoader(trainer.config['data']['ravdess']['path'])
    
    try:
        facial_images, facial_labels = facial_loader.load_data()
        facial_data = facial_loader.prepare_data(facial_images, facial_labels)
    except FileNotFoundError:
        logger.error("FER2013 dataset not found. Please download it first.")
        return
    
    try:
        speech_features, speech_labels = speech_loader.load_data()
        speech_data = speech_loader.prepare_data(speech_features, speech_labels)
    except FileNotFoundError:
        logger.error("RAVDESS dataset not found. Please download it first.")
        return
    
    # Prepare paired data
    paired_data = trainer.prepare_paired_data(
        facial_data, speech_data,
        num_samples=args.num_pairs
    )
    
    # Extract features
    feature_data = trainer.extract_features(
        facial_extractor, speech_extractor,
        paired_data
    )
    
    # Train
    trainer.train(feature_data)
    
    # Evaluate
    metrics = trainer.evaluate(feature_data)
    print("\nFinal Test Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save model
    trainer.save_model()


if __name__ == "__main__":
    main()
