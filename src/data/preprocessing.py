"""
Data Preprocessing Utilities

This module provides preprocessing functions for images and audio data.
"""

import numpy as np
import cv2
import librosa
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocessor for facial images."""
    
    def __init__(self, target_size: Tuple[int, int] = (48, 48)):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
        
        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract face from image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Cropped face image or None if no face detected
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Get the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y:y+h, x:x+w]
        
        return face
    
    def preprocess(self, image: np.ndarray, detect_face: bool = True) -> Optional[np.ndarray]:
        """
        Preprocess image for emotion recognition.
        
        Args:
            image: Input image
            detect_face: Whether to detect and crop face
            
        Returns:
            Preprocessed image ready for model input
        """
        # Detect face if requested
        if detect_face:
            face = self.detect_face(image)
            if face is None:
                return None
            image = face
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        image = cv2.resize(image, self.target_size)
        
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        
        # Add channel dimension
        image = np.expand_dims(image, axis=-1)
        
        return image
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to image.
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random rotation (-15 to 15 degrees)
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Random brightness adjustment
        brightness = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 1)
        
        return image


class AudioPreprocessor:
    """Preprocessor for audio data."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        duration: float = 3.0,
        n_mfcc: int = 40,
        n_mels: int = 128
    ):
        """
        Initialize audio preprocessor.
        
        Args:
            sample_rate: Target sample rate
            duration: Target duration in seconds
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel bands
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.samples = int(sample_rate * duration)
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load and preprocess audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio time series
        """
        audio, _ = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
        
        # Pad or truncate
        if len(audio) < self.samples:
            audio = np.pad(audio, (0, self.samples - len(audio)), mode='constant')
        else:
            audio = audio[:self.samples]
        
        return audio
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio time series
            
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc
        )
        return mfcc
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram from audio.
        
        Args:
            audio: Audio time series
            
        Returns:
            Mel spectrogram (in dB)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def preprocess(self, audio: np.ndarray, feature_type: str = 'mfcc') -> np.ndarray:
        """
        Preprocess audio for emotion recognition.
        
        Args:
            audio: Audio time series
            feature_type: Type of feature to extract ('mfcc' or 'mel_spectrogram')
            
        Returns:
            Extracted features ready for model input
        """
        # Extract features
        if feature_type == 'mfcc':
            features = self.extract_mfcc(audio)
        elif feature_type == 'mel_spectrogram':
            features = self.extract_mel_spectrogram(audio)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Normalize
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        # Add channel dimension
        features = np.expand_dims(features, axis=-1)
        
        return features
    
    def augment(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to audio.
        
        Args:
            audio: Audio time series
            
        Returns:
            Augmented audio
        """
        # Random time shift
        shift = np.random.randint(-self.sample_rate // 4, self.sample_rate // 4)
        audio = np.roll(audio, shift)
        
        # Random noise addition
        noise_factor = np.random.uniform(0, 0.005)
        noise = np.random.randn(len(audio)) * noise_factor
        audio = audio + noise
        
        # Random time stretch
        rate = np.random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Ensure correct length
        if len(audio) < self.samples:
            audio = np.pad(audio, (0, self.samples - len(audio)), mode='constant')
        else:
            audio = audio[:self.samples]
        
        return audio


if __name__ == "__main__":
    # Test preprocessors
    
    # Test ImagePreprocessor
    img_preprocessor = ImagePreprocessor()
    test_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    processed = img_preprocessor.preprocess(test_img, detect_face=False)
    print(f"Image preprocessor output shape: {processed.shape}")
    
    # Test AudioPreprocessor
    audio_preprocessor = AudioPreprocessor()
    test_audio = np.random.randn(22050 * 3)  # 3 seconds of audio
    processed = audio_preprocessor.preprocess(test_audio)
    print(f"Audio preprocessor output shape: {processed.shape}")
