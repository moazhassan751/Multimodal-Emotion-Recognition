"""
Facial Emotion Recognition Model

CNN-based model for recognizing emotions from facial expressions.
Supports multiple architectures: Custom CNN, VGG16, ResNet50, MobileNet.
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    BatchNormalization, GlobalAveragePooling2D, Add, Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FacialEmotionModel:
    """
    Facial Emotion Recognition Model.
    
    Supports multiple architectures for facial emotion recognition
    trained on grayscale 48x48 images.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (48, 48, 1),
        num_classes: int = 7,
        architecture: str = 'custom_cnn',
        dropout_rate: float = 0.5,
        l2_reg: float = 0.01
    ):
        """
        Initialize the facial emotion model.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of emotion classes
            architecture: Model architecture ('custom_cnn', 'vgg16', 'resnet50', 'mobilenet')
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        
    def build_custom_cnn(self) -> Model:
        """
        Build a custom CNN architecture for facial emotion recognition.
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # Block 1
            Conv2D(64, (3, 3), padding='same', activation='relu',
                   kernel_regularizer=l2(self.l2_reg),
                   input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding='same', activation='relu',
                   kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(128, (3, 3), padding='same', activation='relu',
                   kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding='same', activation='relu',
                   kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 3
            Conv2D(256, (3, 3), padding='same', activation='relu',
                   kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Conv2D(256, (3, 3), padding='same', activation='relu',
                   kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 4
            Conv2D(512, (3, 3), padding='same', activation='relu',
                   kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Conv2D(512, (3, 3), padding='same', activation='relu',
                   kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Fully Connected Layers
            Flatten(),
            Dense(512, activation='relu', kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            Dense(256, activation='relu', kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_vgg_based(self) -> Model:
        """
        Build a VGG-based model (modified for grayscale input).
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Convert grayscale to RGB by repeating channels
        if self.input_shape[-1] == 1:
            x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
            # Resize to VGG input size
            x = tf.keras.layers.Resizing(48, 48)(x)
        else:
            x = inputs
        
        # Build custom VGG-like architecture for small images
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Classification head
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_resnet_block(self, x, filters: int, kernel_size: int = 3, stride: int = 1):
        """Build a ResNet block."""
        shortcut = x
        
        x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        
        # Adjust shortcut if dimensions don't match
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        
        return x
    
    def build_resnet_based(self) -> Model:
        """
        Build a ResNet-based model for facial emotion recognition.
        
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=self.input_shape)
        
        # Initial convolution
        x = Conv2D(64, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # ResNet blocks
        x = self.build_resnet_block(x, 64)
        x = self.build_resnet_block(x, 64)
        
        x = self.build_resnet_block(x, 128, stride=2)
        x = self.build_resnet_block(x, 128)
        
        x = self.build_resnet_block(x, 256, stride=2)
        x = self.build_resnet_block(x, 256)
        
        x = self.build_resnet_block(x, 512, stride=2)
        x = self.build_resnet_block(x, 512)
        
        # Global pooling and classification
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_mobilenet_based(self) -> Model:
        """
        Build a MobileNet-based lightweight model.
        
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=self.input_shape)
        
        # Convert grayscale to RGB
        if self.input_shape[-1] == 1:
            x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
        else:
            x = inputs
        
        # Depthwise separable convolution blocks
        def depthwise_sep_conv(x, filters, stride=1):
            x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filters, 1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
        
        # Initial conv
        x = Conv2D(32, 3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Depthwise separable blocks
        x = depthwise_sep_conv(x, 64)
        x = depthwise_sep_conv(x, 128, stride=2)
        x = depthwise_sep_conv(x, 128)
        x = depthwise_sep_conv(x, 256, stride=2)
        x = depthwise_sep_conv(x, 256)
        x = depthwise_sep_conv(x, 512, stride=2)
        
        for _ in range(5):
            x = depthwise_sep_conv(x, 512)
        
        x = depthwise_sep_conv(x, 1024, stride=2)
        x = depthwise_sep_conv(x, 1024)
        
        # Classification head
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build(self) -> Model:
        """
        Build the model based on specified architecture.
        
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building {self.architecture} model...")
        
        if self.architecture == 'custom_cnn':
            self.model = self.build_custom_cnn()
        elif self.architecture == 'vgg16':
            self.model = self.build_vgg_based()
        elif self.architecture == 'resnet50':
            self.model = self.build_resnet_based()
        elif self.architecture == 'mobilenet':
            self.model = self.build_mobilenet_based()
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
        feature_layer = self.model.layers[-3].output  # Before last dropout and dense
        feature_extractor = Model(inputs=self.model.input, outputs=feature_layer)
        
        return feature_extractor


if __name__ == "__main__":
    # Test model building
    for arch in ['custom_cnn', 'vgg16', 'resnet50', 'mobilenet']:
        print(f"\n{'='*50}")
        print(f"Testing {arch} architecture")
        print('='*50)
        
        model = FacialEmotionModel(architecture=arch)
        model.build()
        model.compile()
        
        # Test with dummy input
        import numpy as np
        dummy_input = np.random.randn(1, 48, 48, 1)
        output = model.model.predict(dummy_input, verbose=0)
        print(f"Output shape: {output.shape}")
        print(f"Output sum: {output.sum():.4f} (should be ~1.0)")
