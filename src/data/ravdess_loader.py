"""
RAVDESS Dataset Loader

This module handles loading and preprocessing of the RAVDESS speech emotion dataset.
Dataset contains audio recordings from 24 actors expressing 8 emotions.
"""

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from typing import Tuple, Optional, List
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAVDESSDataLoader:
    """
    Data loader for RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).
    
    Filename format: XX-XX-XX-XX-XX-XX-XX.wav
    - Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
    - Vocal channel (01 = speech, 02 = song)
    - Emotion (01-08)
    - Emotional intensity (01 = normal, 02 = strong)
    - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
    - Repetition (01 = 1st repetition, 02 = 2nd repetition)
    - Actor (01 to 24)
    
    Emotions:
        01: Neutral
        02: Calm
        03: Happy
        04: Sad
        05: Angry
        06: Fearful
        07: Disgust
        08: Surprised
    """
    
    EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    NUM_CLASSES = 8
    
    def __init__(
        self, 
        data_path: str = "data/ravdess",
        sample_rate: int = 22050,
        duration: float = 3.0,
        n_mfcc: int = 40,
        n_mels: int = 128
    ):
        """
        Initialize the RAVDESS data loader.
        
        Args:
            data_path: Path to the RAVDESS dataset directory
            sample_rate: Audio sample rate
            duration: Duration in seconds to load/pad audio
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel bands
        """
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.samples = int(sample_rate * duration)
        
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio time series
            
        Returns:
            MFCC features array
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
            Mel spectrogram array
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_features(self, audio: np.ndarray) -> dict:
        """
        Extract multiple audio features.
        
        Args:
            audio: Audio time series
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # MFCC
        features['mfcc'] = self.extract_mfcc(audio)
        
        # Mel spectrogram
        features['mel_spectrogram'] = self.extract_mel_spectrogram(audio)
        
        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        
        # Spectral contrast
        features['spectral_contrast'] = librosa.feature.spectral_contrast(
            y=audio, sr=self.sample_rate
        )
        
        # Zero crossing rate
        features['zcr'] = librosa.feature.zero_crossing_rate(audio)
        
        return features
    
    def load_audio_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess a single audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Preprocessed audio array or None if loading fails
        """
        try:
            audio, _ = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Pad or truncate to fixed length
            if len(audio) < self.samples:
                audio = np.pad(audio, (0, self.samples - len(audio)), mode='constant')
            else:
                audio = audio[:self.samples]
                
            return audio
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            return None
    
    def parse_filename(self, filename: str) -> dict:
        """
        Parse RAVDESS filename to extract metadata.
        
        Args:
            filename: RAVDESS filename
            
        Returns:
            Dictionary with parsed metadata
        """
        parts = filename.replace('.wav', '').split('-')
        
        if len(parts) != 7:
            return None
            
        return {
            'modality': int(parts[0]),
            'vocal_channel': int(parts[1]),
            'emotion': int(parts[2]),
            'intensity': int(parts[3]),
            'statement': int(parts[4]),
            'repetition': int(parts[5]),
            'actor': int(parts[6])
        }
    
    def load_data(self, feature_type: str = 'mfcc') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load RAVDESS dataset and extract features.
        
        Args:
            feature_type: Type of feature to extract ('mfcc', 'mel_spectrogram')
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        features_list = []
        labels = []
        
        # Find all audio files
        audio_files = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            raise FileNotFoundError(
                f"No audio files found in {self.data_path}. "
                "Please download RAVDESS from: https://zenodo.org/record/1188976"
            )
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        for file_path in tqdm(audio_files, desc="Loading audio files"):
            filename = os.path.basename(file_path)
            metadata = self.parse_filename(filename)
            
            if metadata is None:
                continue
                
            # Load audio
            audio = self.load_audio_file(file_path)
            if audio is None:
                continue
            
            # Extract features
            if feature_type == 'mfcc':
                feature = self.extract_mfcc(audio)
            elif feature_type == 'mel_spectrogram':
                feature = self.extract_mel_spectrogram(audio)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
            
            features_list.append(feature)
            labels.append(metadata['emotion'] - 1)  # Convert to 0-indexed
        
        # Stack features and pad to same length
        max_len = max(f.shape[1] for f in features_list)
        padded_features = []
        for f in features_list:
            if f.shape[1] < max_len:
                pad_width = max_len - f.shape[1]
                f = np.pad(f, ((0, 0), (0, pad_width)), mode='constant')
            padded_features.append(f)
        
        features_array = np.array(padded_features)
        features_array = features_array[..., np.newaxis]  # Add channel dimension
        labels_array = np.array(labels)
        
        logger.info(f"Loaded {len(features_array)} samples with shape {features_array.shape}")
        return features_array, labels_array
    
    def prepare_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> dict:
        """
        Split data into train, validation, and test sets.
        
        Args:
            features: Feature array
            labels: Label array
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train, val, and test data
        """
        # Convert labels to one-hot encoding
        labels_onehot = to_categorical(labels, num_classes=self.NUM_CLASSES)
        
        # Normalize features
        mean = np.mean(features)
        std = np.std(features)
        features_normalized = (features - mean) / (std + 1e-8)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            features_normalized, labels_onehot,
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
            'y_test': y_test,
            'mean': mean,
            'std': std
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
        return RAVDESSDataLoader.EMOTIONS[label_idx]
    
    @staticmethod
    def get_emotion_index(emotion_name: str) -> int:
        """Get label index from emotion name."""
        return RAVDESSDataLoader.EMOTIONS.index(emotion_name)


if __name__ == "__main__":
    # Test the data loader
    loader = RAVDESSDataLoader()
    
    try:
        features, labels = loader.load_data(feature_type='mfcc')
        data = loader.prepare_data(features, labels)
        print(f"Data loaded successfully!")
        print(f"Training samples: {data['X_train'].shape}")
        print(f"Validation samples: {data['X_val'].shape}")
        print(f"Test samples: {data['X_test'].shape}")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
