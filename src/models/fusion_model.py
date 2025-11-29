"""
Multimodal Fusion Model

Combines facial and speech emotion recognition models using various fusion strategies.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate, Add, Multiply,
    BatchNormalization, GlobalAveragePooling1D, Attention,
    LayerNormalization
)
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalFusionModel:
    """
    Multimodal Fusion Model for emotion recognition.
    
    Combines features from facial and speech models using various
    fusion strategies: concatenation, attention, weighted, etc.
    """
    
    # Common emotions between FER2013 and RAVDESS
    COMMON_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def __init__(
        self,
        facial_feature_dim: int = 256,
        speech_feature_dim: int = 256,
        num_classes: int = 7,
        fusion_method: str = 'concatenate',
        dense_units: List[int] = [256, 128],
        dropout_rate: float = 0.4
    ):
        """
        Initialize the multimodal fusion model.
        
        Args:
            facial_feature_dim: Dimension of facial features
            speech_feature_dim: Dimension of speech features
            num_classes: Number of output classes
            fusion_method: Fusion strategy ('concatenate', 'attention', 'weighted', 'gated')
            dense_units: List of dense layer units
            dropout_rate: Dropout rate
        """
        self.facial_feature_dim = facial_feature_dim
        self.speech_feature_dim = speech_feature_dim
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_concatenate_fusion(self) -> Model:
        """
        Build a simple concatenation-based fusion model.
        
        Returns:
            Keras model
        """
        # Input layers
        facial_input = Input(shape=(self.facial_feature_dim,), name='facial_features')
        speech_input = Input(shape=(self.speech_feature_dim,), name='speech_features')
        
        # Normalize inputs
        facial_norm = LayerNormalization()(facial_input)
        speech_norm = LayerNormalization()(speech_input)
        
        # Concatenate features
        fused = Concatenate()([facial_norm, speech_norm])
        
        # Dense layers
        x = fused
        for units in self.dense_units:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='emotion_output')(x)
        
        model = Model(
            inputs=[facial_input, speech_input],
            outputs=outputs
        )
        return model
    
    def build_attention_fusion(self) -> Model:
        """
        Build an attention-based fusion model.
        
        Returns:
            Keras model
        """
        # Input layers
        facial_input = Input(shape=(self.facial_feature_dim,), name='facial_features')
        speech_input = Input(shape=(self.speech_feature_dim,), name='speech_features')
        
        # Project to same dimension
        proj_dim = max(self.facial_feature_dim, self.speech_feature_dim)
        
        facial_proj = Dense(proj_dim, activation='relu')(facial_input)
        speech_proj = Dense(proj_dim, activation='relu')(speech_input)
        
        # Reshape for attention: (batch, seq_len=2, features)
        facial_exp = tf.expand_dims(facial_proj, axis=1)
        speech_exp = tf.expand_dims(speech_proj, axis=1)
        multimodal = Concatenate(axis=1)([facial_exp, speech_exp])
        
        # Self-attention
        attention_scores = Dense(1, activation='tanh')(multimodal)
        attention_scores = tf.squeeze(attention_scores, axis=-1)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        
        # Apply attention
        attended = multimodal * attention_weights
        fused = tf.reduce_sum(attended, axis=1)
        
        # Dense layers
        x = fused
        for units in self.dense_units:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='emotion_output')(x)
        
        model = Model(
            inputs=[facial_input, speech_input],
            outputs=outputs
        )
        return model
    
    def build_weighted_fusion(self) -> Model:
        """
        Build a learnable weighted fusion model.
        
        Returns:
            Keras model
        """
        # Input layers
        facial_input = Input(shape=(self.facial_feature_dim,), name='facial_features')
        speech_input = Input(shape=(self.speech_feature_dim,), name='speech_features')
        
        # Project to same dimension
        proj_dim = 256
        
        facial_proj = Dense(proj_dim, activation='relu')(facial_input)
        facial_proj = LayerNormalization()(facial_proj)
        
        speech_proj = Dense(proj_dim, activation='relu')(speech_input)
        speech_proj = LayerNormalization()(speech_proj)
        
        # Learn modality weights
        combined = Concatenate()([facial_proj, speech_proj])
        weight_input = Dense(64, activation='relu')(combined)
        weights = Dense(2, activation='softmax', name='modality_weights')(weight_input)
        
        # Apply weights
        facial_weight = tf.expand_dims(weights[:, 0], axis=-1)
        speech_weight = tf.expand_dims(weights[:, 1], axis=-1)
        
        weighted_facial = facial_proj * facial_weight
        weighted_speech = speech_proj * speech_weight
        
        fused = Add()([weighted_facial, weighted_speech])
        
        # Dense layers
        x = fused
        for units in self.dense_units:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='emotion_output')(x)
        
        model = Model(
            inputs=[facial_input, speech_input],
            outputs=outputs
        )
        return model
    
    def build_gated_fusion(self) -> Model:
        """
        Build a gated fusion model with learned gates.
        
        Returns:
            Keras model
        """
        # Input layers
        facial_input = Input(shape=(self.facial_feature_dim,), name='facial_features')
        speech_input = Input(shape=(self.speech_feature_dim,), name='speech_features')
        
        # Project to same dimension
        proj_dim = 256
        
        facial_proj = Dense(proj_dim, activation='relu')(facial_input)
        speech_proj = Dense(proj_dim, activation='relu')(speech_input)
        
        # Gating mechanism
        combined = Concatenate()([facial_proj, speech_proj])
        
        # Facial gate
        facial_gate = Dense(proj_dim, activation='sigmoid', name='facial_gate')(combined)
        gated_facial = Multiply()([facial_proj, facial_gate])
        
        # Speech gate
        speech_gate = Dense(proj_dim, activation='sigmoid', name='speech_gate')(combined)
        gated_speech = Multiply()([speech_proj, speech_gate])
        
        # Combine gated features
        fused = Add()([gated_facial, gated_speech])
        fused = LayerNormalization()(fused)
        
        # Dense layers
        x = fused
        for units in self.dense_units:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='emotion_output')(x)
        
        model = Model(
            inputs=[facial_input, speech_input],
            outputs=outputs
        )
        return model
    
    def build_bilinear_fusion(self) -> Model:
        """
        Build a bilinear fusion model.
        
        Returns:
            Keras model
        """
        # Input layers
        facial_input = Input(shape=(self.facial_feature_dim,), name='facial_features')
        speech_input = Input(shape=(self.speech_feature_dim,), name='speech_features')
        
        # Project to same dimension
        proj_dim = 128
        
        facial_proj = Dense(proj_dim, activation='relu')(facial_input)
        speech_proj = Dense(proj_dim, activation='relu')(speech_input)
        
        # Bilinear interaction (outer product approximation)
        # Use factorized bilinear pooling for efficiency
        facial_expand = tf.expand_dims(facial_proj, axis=2)  # (batch, proj_dim, 1)
        speech_expand = tf.expand_dims(speech_proj, axis=1)  # (batch, 1, proj_dim)
        
        bilinear = tf.matmul(facial_expand, speech_expand)  # (batch, proj_dim, proj_dim)
        bilinear_flat = tf.reshape(bilinear, [-1, proj_dim * proj_dim])
        
        # Reduce dimensionality
        fused = Dense(256, activation='relu')(bilinear_flat)
        fused = BatchNormalization()(fused)
        fused = Dropout(self.dropout_rate)(fused)
        
        # Dense layers
        x = fused
        for units in self.dense_units:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='emotion_output')(x)
        
        model = Model(
            inputs=[facial_input, speech_input],
            outputs=outputs
        )
        return model
    
    def build(self) -> Model:
        """
        Build the fusion model based on specified method.
        
        Returns:
            Keras model
        """
        logger.info(f"Building {self.fusion_method} fusion model...")
        
        if self.fusion_method == 'concatenate':
            self.model = self.build_concatenate_fusion()
        elif self.fusion_method == 'attention':
            self.model = self.build_attention_fusion()
        elif self.fusion_method == 'weighted':
            self.model = self.build_weighted_fusion()
        elif self.fusion_method == 'gated':
            self.model = self.build_gated_fusion()
        elif self.fusion_method == 'bilinear':
            self.model = self.build_bilinear_fusion()
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        logger.info(f"Fusion model built with {self.model.count_params():,} parameters")
        return self.model
    
    def compile(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.0001,
        loss: str = 'categorical_crossentropy',
        metrics: list = ['accuracy']
    ):
        """
        Compile the model.
        
        Args:
            optimizer: Optimizer name or instance
            learning_rate: Learning rate
            loss: Loss function
            metrics: List of metrics
        """
        if self.model is None:
            self.build()
        
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
        
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
        logger.info("Fusion model compiled successfully")
    
    def summary(self):
        """Print model summary."""
        if self.model is not None:
            self.model.summary()


class EndToEndMultimodalModel:
    """
    End-to-end multimodal model that combines raw inputs
    through feature extractors and fusion network.
    """
    
    def __init__(
        self,
        facial_model,
        speech_model,
        num_classes: int = 7,
        fusion_method: str = 'concatenate',
        freeze_feature_extractors: bool = True
    ):
        """
        Initialize end-to-end multimodal model.
        
        Args:
            facial_model: Pre-trained facial emotion model
            speech_model: Pre-trained speech emotion model
            num_classes: Number of output classes
            fusion_method: Fusion strategy
            freeze_feature_extractors: Whether to freeze feature extractors
        """
        self.facial_model = facial_model
        self.speech_model = speech_model
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.freeze_feature_extractors = freeze_feature_extractors
        self.model = None
    
    def build(self) -> Model:
        """
        Build the end-to-end model.
        
        Returns:
            Keras model
        """
        # Get feature extractors
        facial_extractor = self.facial_model.get_feature_extractor()
        speech_extractor = self.speech_model.get_feature_extractor()
        
        # Freeze if requested
        if self.freeze_feature_extractors:
            for layer in facial_extractor.layers:
                layer.trainable = False
            for layer in speech_extractor.layers:
                layer.trainable = False
        
        # Input layers
        facial_input = Input(shape=self.facial_model.input_shape, name='facial_input')
        speech_input = Input(shape=self.speech_model.input_shape, name='speech_input')
        
        # Extract features
        facial_features = facial_extractor(facial_input)
        speech_features = speech_extractor(speech_input)
        
        # Get feature dimensions
        facial_dim = facial_features.shape[-1]
        speech_dim = speech_features.shape[-1]
        
        # Build fusion model
        fusion_model = MultimodalFusionModel(
            facial_feature_dim=facial_dim,
            speech_feature_dim=speech_dim,
            num_classes=self.num_classes,
            fusion_method=self.fusion_method
        )
        fusion_model.build()
        
        # Get outputs
        outputs = fusion_model.model([facial_features, speech_features])
        
        self.model = Model(
            inputs=[facial_input, speech_input],
            outputs=outputs
        )
        
        return self.model
    
    def compile(self, **kwargs):
        """Compile the model."""
        if self.model is None:
            self.build()
        self.model.compile(**kwargs)
    
    def summary(self):
        """Print model summary."""
        if self.model is not None:
            self.model.summary()


if __name__ == "__main__":
    # Test fusion models
    import numpy as np
    
    for method in ['concatenate', 'attention', 'weighted', 'gated', 'bilinear']:
        print(f"\n{'='*50}")
        print(f"Testing {method} fusion")
        print('='*50)
        
        model = MultimodalFusionModel(
            facial_feature_dim=256,
            speech_feature_dim=256,
            fusion_method=method
        )
        model.build()
        model.compile()
        
        # Test with dummy inputs
        facial_features = np.random.randn(2, 256)
        speech_features = np.random.randn(2, 256)
        
        output = model.model.predict([facial_features, speech_features], verbose=0)
        print(f"Output shape: {output.shape}")
        print(f"Output sum per sample: {output.sum(axis=1)}")
