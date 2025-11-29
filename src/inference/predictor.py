"""
Emotion prediction module for inference on images and audio files.
"""

import numpy as np
import cv2
import librosa
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union
import os


class EmotionPredictor:
    """
    Unified emotion predictor for facial, speech, and multimodal inference.
    """
    
    # Emotion labels
    FACIAL_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    SPEECH_EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    
    def __init__(
        self,
        facial_model_path: Optional[str] = None,
        speech_model_path: Optional[str] = None,
        fusion_model_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the emotion predictor.
        
        Args:
            facial_model_path: Path to trained facial emotion model
            speech_model_path: Path to trained speech emotion model
            fusion_model_path: Path to trained fusion model
            config: Configuration dictionary
        """
        self.config = config or {}
        self.facial_model = None
        self.speech_model = None
        self.fusion_model = None
        
        # Load face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load models if paths provided
        if facial_model_path and os.path.exists(facial_model_path):
            self.load_facial_model(facial_model_path)
        
        if speech_model_path and os.path.exists(speech_model_path):
            self.load_speech_model(speech_model_path)
            
        if fusion_model_path and os.path.exists(fusion_model_path):
            self.load_fusion_model(fusion_model_path)
    
    def load_facial_model(self, model_path: str) -> None:
        """Load facial emotion recognition model."""
        self.facial_model = tf.keras.models.load_model(model_path)
        print(f"Loaded facial model from {model_path}")
    
    def load_speech_model(self, model_path: str) -> None:
        """Load speech emotion recognition model."""
        self.speech_model = tf.keras.models.load_model(model_path)
        print(f"Loaded speech model from {model_path}")
    
    def load_fusion_model(self, model_path: str) -> None:
        """Load multimodal fusion model."""
        self.fusion_model = tf.keras.models.load_model(model_path)
        print(f"Loaded fusion model from {model_path}")
    
    def preprocess_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (48, 48),
        detect_face: bool = True
    ) -> Optional[np.ndarray]:
        """
        Preprocess image for facial emotion recognition.
        
        Args:
            image: Input image (BGR or grayscale)
            target_size: Target size for the model
            detect_face: Whether to detect and crop face
            
        Returns:
            Preprocessed image array or None if no face detected
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if detect_face:
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
            face_roi = gray[y:y+h, x:x+w]
        else:
            face_roi = gray
        
        # Resize to target size
        face_resized = cv2.resize(face_roi, target_size)
        
        # Normalize
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Add channel and batch dimensions
        face_input = np.expand_dims(np.expand_dims(face_normalized, axis=-1), axis=0)
        
        return face_input
    
    def preprocess_audio(
        self,
        audio_path: str,
        sr: int = 22050,
        n_mfcc: int = 40,
        max_length: int = 100
    ) -> np.ndarray:
        """
        Preprocess audio file for speech emotion recognition.
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients
            max_length: Maximum length for padding/truncating
            
        Returns:
            Preprocessed MFCC features
        """
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate
        if mfcc.shape[1] < max_length:
            pad_width = max_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_length]
        
        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        # Transpose and add batch dimension
        mfcc = np.expand_dims(mfcc.T, axis=0)
        
        return mfcc
    
    def preprocess_audio_stream(
        self,
        audio_data: np.ndarray,
        sr: int = 22050,
        n_mfcc: int = 40,
        max_length: int = 100
    ) -> np.ndarray:
        """
        Preprocess audio stream data for real-time inference.
        
        Args:
            audio_data: Raw audio samples
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients
            max_length: Maximum length for padding/truncating
            
        Returns:
            Preprocessed MFCC features
        """
        # Convert to float if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate
        if mfcc.shape[1] < max_length:
            pad_width = max_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_length]
        
        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        # Transpose and add batch dimension
        mfcc = np.expand_dims(mfcc.T, axis=0)
        
        return mfcc
    
    def predict_facial_emotion(
        self,
        image: np.ndarray,
        return_probabilities: bool = False
    ) -> Union[str, Tuple[str, np.ndarray]]:
        """
        Predict emotion from facial image.
        
        Args:
            image: Input image (BGR format)
            return_probabilities: Whether to return probability distribution
            
        Returns:
            Predicted emotion label (and probabilities if requested)
        """
        if self.facial_model is None:
            raise ValueError("Facial model not loaded")
        
        # Preprocess image
        processed = self.preprocess_image(image)
        
        if processed is None:
            return ("No face detected", np.zeros(len(self.FACIAL_EMOTIONS))) if return_probabilities else "No face detected"
        
        # Predict
        predictions = self.facial_model.predict(processed, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        emotion = self.FACIAL_EMOTIONS[emotion_idx]
        
        if return_probabilities:
            return emotion, predictions
        return emotion
    
    def predict_speech_emotion(
        self,
        audio_path: str,
        return_probabilities: bool = False
    ) -> Union[str, Tuple[str, np.ndarray]]:
        """
        Predict emotion from audio file.
        
        Args:
            audio_path: Path to audio file
            return_probabilities: Whether to return probability distribution
            
        Returns:
            Predicted emotion label (and probabilities if requested)
        """
        if self.speech_model is None:
            raise ValueError("Speech model not loaded")
        
        # Preprocess audio
        processed = self.preprocess_audio(audio_path)
        
        # Predict
        predictions = self.speech_model.predict(processed, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        emotion = self.SPEECH_EMOTIONS[emotion_idx]
        
        if return_probabilities:
            return emotion, predictions
        return emotion
    
    def predict_multimodal(
        self,
        image: np.ndarray,
        audio_path: str,
        return_probabilities: bool = False
    ) -> Union[str, Tuple[str, np.ndarray]]:
        """
        Predict emotion using both facial and speech inputs.
        
        Args:
            image: Input image (BGR format)
            audio_path: Path to audio file
            return_probabilities: Whether to return probability distribution
            
        Returns:
            Predicted emotion label (and probabilities if requested)
        """
        if self.fusion_model is None:
            raise ValueError("Fusion model not loaded")
        
        # Preprocess inputs
        image_processed = self.preprocess_image(image)
        audio_processed = self.preprocess_audio(audio_path)
        
        if image_processed is None:
            # Fall back to speech only
            return self.predict_speech_emotion(audio_path, return_probabilities)
        
        # Predict with fusion model
        predictions = self.fusion_model.predict(
            [image_processed, audio_processed],
            verbose=0
        )[0]
        
        # Use speech emotions (typically more comprehensive)
        emotion_idx = np.argmax(predictions)
        emotion = self.SPEECH_EMOTIONS[emotion_idx]
        
        if return_probabilities:
            return emotion, predictions
        return emotion
    
    def predict_from_file(
        self,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        mode: str = 'auto'
    ) -> Dict:
        """
        Predict emotion from image and/or audio file.
        
        Args:
            image_path: Path to image file
            audio_path: Path to audio file
            mode: 'facial', 'speech', 'fusion', or 'auto'
            
        Returns:
            Dictionary with prediction results
        """
        results = {}
        
        # Determine mode if auto
        if mode == 'auto':
            if image_path and audio_path:
                mode = 'fusion'
            elif image_path:
                mode = 'facial'
            elif audio_path:
                mode = 'speech'
            else:
                raise ValueError("No input provided")
        
        # Facial prediction
        if mode in ['facial', 'fusion'] and image_path:
            image = cv2.imread(image_path)
            if image is not None:
                emotion, probs = self.predict_facial_emotion(image, return_probabilities=True)
                results['facial'] = {
                    'emotion': emotion,
                    'probabilities': {e: float(p) for e, p in zip(self.FACIAL_EMOTIONS, probs)},
                    'confidence': float(np.max(probs))
                }
        
        # Speech prediction
        if mode in ['speech', 'fusion'] and audio_path:
            emotion, probs = self.predict_speech_emotion(audio_path, return_probabilities=True)
            results['speech'] = {
                'emotion': emotion,
                'probabilities': {e: float(p) for e, p in zip(self.SPEECH_EMOTIONS, probs)},
                'confidence': float(np.max(probs))
            }
        
        # Fusion prediction
        if mode == 'fusion' and image_path and audio_path and self.fusion_model:
            image = cv2.imread(image_path)
            if image is not None:
                emotion, probs = self.predict_multimodal(image, audio_path, return_probabilities=True)
                results['fusion'] = {
                    'emotion': emotion,
                    'probabilities': {e: float(p) for e, p in zip(self.SPEECH_EMOTIONS, probs)},
                    'confidence': float(np.max(probs))
                }
        
        return results
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect all faces in an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return [tuple(f) for f in faces]
    
    def predict_all_faces(
        self,
        image: np.ndarray
    ) -> List[Dict]:
        """
        Predict emotions for all faces in an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of dictionaries with face location and emotion prediction
        """
        if self.facial_model is None:
            raise ValueError("Facial model not loaded")
        
        results = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            # Extract and preprocess face
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (48, 48))
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_input = np.expand_dims(np.expand_dims(face_normalized, axis=-1), axis=0)
            
            # Predict
            predictions = self.facial_model.predict(face_input, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            
            results.append({
                'bbox': (int(x), int(y), int(w), int(h)),
                'emotion': self.FACIAL_EMOTIONS[emotion_idx],
                'confidence': float(predictions[emotion_idx]),
                'probabilities': {e: float(p) for e, p in zip(self.FACIAL_EMOTIONS, predictions)}
            })
        
        return results


def main():
    """Example usage of EmotionPredictor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Emotion Prediction')
    parser.add_argument('--facial-model', type=str, help='Path to facial model')
    parser.add_argument('--speech-model', type=str, help='Path to speech model')
    parser.add_argument('--fusion-model', type=str, help='Path to fusion model')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['facial', 'speech', 'fusion', 'auto'])
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = EmotionPredictor(
        facial_model_path=args.facial_model,
        speech_model_path=args.speech_model,
        fusion_model_path=args.fusion_model
    )
    
    # Make prediction
    results = predictor.predict_from_file(
        image_path=args.image,
        audio_path=args.audio,
        mode=args.mode
    )
    
    # Print results
    print("\n=== Emotion Prediction Results ===")
    for modality, result in results.items():
        print(f"\n{modality.upper()}:")
        print(f"  Emotion: {result['emotion']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print("  Probabilities:")
        for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
            print(f"    {emotion}: {prob:.2%}")


if __name__ == '__main__':
    main()
