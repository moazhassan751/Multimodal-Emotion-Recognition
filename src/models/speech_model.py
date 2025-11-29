"""
Speech Emotion Recognition Model

LSTM/CNN-based model for recognizing emotions from speech audio.
Uses MFCC or Mel spectrogram features as input.
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    BatchNormalization, GlobalAveragePooling2D, LSTM, Bidirectional,
    TimeDistributed, Reshape, Permute, GRU, Attention, Concatenate
)
from tensorflow.keras.regularizers import l2
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechEmotionModel:
    """
    Speech Emotion Recognition Model.
    
    Supports multiple architectures for speech emotion recognition
    using MFCC or Mel spectrogram features.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (40, 130, 1),
        num_classes: int = 8,
        architecture: str = 'cnn_lstm',
        lstm_units: int = 128,
        dropout_rate: float = 0.3
    ):
        """
        Initialize the speech emotion model.
        
        Args:
            input_shape: Shape of input features (n_mfcc/n_mels, time_steps, channels)
            num_classes: Number of emotion classes
            architecture: Model architecture ('cnn_lstm', 'lstm', 'cnn', 'attention')
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_cnn(self) -> Model:
        """
        Build a CNN-only model for speech emotion recognition.
        
        Returns:
            Keras model
        """
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), activation='relu', padding='same',
                   input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 3
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 4
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Classification head
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(128, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_lstm(self) -> Model:
        """
        Build an LSTM-only model for speech emotion recognition.
        
        Returns:
            Keras model
        """
        inputs = Input(shape=self.input_shape)
        
        # Reshape for LSTM: (batch, time_steps, features)
        # From (n_mfcc, time_steps, 1) to (time_steps, n_mfcc)
        x = Permute((2, 1, 3))(inputs)
        x = Reshape((self.input_shape[1], self.input_shape[0]))(x)
        
        # Bidirectional LSTM layers
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Bidirectional(LSTM(self.lstm_units))(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Classification head
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_cnn_lstm(self) -> Model:
        """
        Build a hybrid CNN-LSTM model for speech emotion recognition.
        
        Returns:
            Keras model
        """
        inputs = Input(shape=self.input_shape)
        
        # CNN feature extraction
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Reshape for LSTM
        # Get the shape dynamically
        shape = x.shape
        x = Reshape((shape[1], shape[2] * shape[3]))(x)
        
        # LSTM layers
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Bidirectional(LSTM(self.lstm_units))(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Classification head
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_attention(self) -> Model:
        """
        Build an attention-based model for speech emotion recognition.
        
        Returns:
            Keras model
        """
        inputs = Input(shape=self.input_shape)
        
        # CNN feature extraction
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Reshape for LSTM
        shape = x.shape
        x = Reshape((shape[1], shape[2] * shape[3]))(x)
        
        # Bidirectional LSTM with attention
        lstm_out = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(x)
        lstm_out = Dropout(self.dropout_rate)(lstm_out)
        
        # Self-attention mechanism
        attention = Dense(1, activation='tanh')(lstm_out)
        attention = Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(self.lstm_units * 2)(attention)
        attention = Permute([2, 1])(attention)
        
        # Apply attention
        x = tf.keras.layers.Multiply()([lstm_out, attention])
        x = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
        
        # Classification head
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_gru(self) -> Model:
        """
        Build a GRU-based model (lighter than LSTM).
        
        Returns:
            Keras model
        """
        inputs = Input(shape=self.input_shape)
        
        # CNN feature extraction
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Reshape for GRU
        shape = x.shape
        x = Reshape((shape[1], shape[2] * shape[3]))(x)
        
        # GRU layers
        x = Bidirectional(GRU(self.lstm_units, return_sequences=True))(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Bidirectional(GRU(self.lstm_units))(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Classification head
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build(self) -> Model:
        """
        Build the model based on specified architecture.
        
        Returns:
            Keras model
        """
        logger.info(f"Building {self.architecture} model...")
        
        if self.architecture == 'cnn':
            self.model = self.build_cnn()
        elif self.architecture == 'lstm':
            self.model = self.build_lstm()
        elif self.architecture == 'cnn_lstm':
            self.model = self.build_cnn_lstm()
        elif self.architecture == 'attention':
            self.model = self.build_attention()
        elif self.architecture == 'gru':
            self.model = self.build_gru()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        logger.info(f"Model built successfully with {self.model.count_params():,} parameters")
        return self.model
    
    def compile(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
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
        logger.info("Model compiled successfully")
    
    def summary(self):
        """Print model summary."""
        if self.model is not None:
            self.model.summary()
    
    def get_feature_extractor(self) -> Model:
        """
        Get the feature extractor part of the model (without classification head).
        
        Returns:
            Feature extractor model
        """
        if self.model is None:
            self.build()
        
        # Get the layer before the final dense layer
        feature_layer = self.model.layers[-3].output
        feature_extractor = Model(inputs=self.model.input, outputs=feature_layer)
        
        return feature_extractor


if __name__ == "__main__":
    # Test model building
    for arch in ['cnn', 'lstm', 'cnn_lstm', 'attention', 'gru']:
        print(f"\n{'='*50}")
        print(f"Testing {arch} architecture")
        print('='*50)
        
        model = SpeechEmotionModel(architecture=arch)
        model.build()
        model.compile()
        
        # Test with dummy input
        import numpy as np
        dummy_input = np.random.randn(1, 40, 130, 1)
        output = model.model.predict(dummy_input, verbose=0)
        print(f"Output shape: {output.shape}")
        print(f"Output sum: {output.sum():.4f} (should be ~1.0)")
