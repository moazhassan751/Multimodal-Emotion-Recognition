"""
Real-time multimodal emotion recognition using webcam and microphone.
"""

import numpy as np
import cv2
import threading
import queue
import time
from typing import Dict, Optional, Tuple, Callable
from collections import deque
import tensorflow as tf

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: PyAudio not available. Audio capture will be disabled.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available. Audio processing will be disabled.")


class RealtimeEmotionRecognizer:
    """
    Real-time emotion recognition from webcam and microphone.
    """
    
    FACIAL_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    SPEECH_EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    
    # Emotion colors for visualization (BGR format)
    EMOTION_COLORS = {
        'Angry': (0, 0, 255),      # Red
        'Disgust': (0, 128, 0),     # Green
        'Fear': (128, 0, 128),      # Purple
        'Happy': (0, 255, 255),     # Yellow
        'Sad': (255, 0, 0),         # Blue
        'Surprise': (0, 165, 255),  # Orange
        'Neutral': (128, 128, 128), # Gray
        'Calm': (144, 238, 144),    # Light green
        'Fearful': (128, 0, 128),   # Purple
        'Surprised': (0, 165, 255)  # Orange
    }
    
    def __init__(
        self,
        facial_model_path: Optional[str] = None,
        speech_model_path: Optional[str] = None,
        fusion_model_path: Optional[str] = None,
        camera_id: int = 0,
        sample_rate: int = 22050,
        audio_chunk_duration: float = 2.0,
        smoothing_window: int = 5
    ):
        """
        Initialize real-time emotion recognizer.
        
        Args:
            facial_model_path: Path to trained facial emotion model
            speech_model_path: Path to trained speech emotion model
            fusion_model_path: Path to trained fusion model
            camera_id: Camera device ID
            sample_rate: Audio sample rate
            audio_chunk_duration: Duration of audio chunks for analysis
            smoothing_window: Number of predictions to average for smoothing
        """
        self.camera_id = camera_id
        self.sample_rate = sample_rate
        self.audio_chunk_duration = audio_chunk_duration
        self.smoothing_window = smoothing_window
        
        # Load models
        self.facial_model = None
        self.speech_model = None
        self.fusion_model = None
        
        if facial_model_path:
            self.facial_model = tf.keras.models.load_model(facial_model_path)
            print(f"Loaded facial model from {facial_model_path}")
        
        if speech_model_path:
            self.speech_model = tf.keras.models.load_model(speech_model_path)
            print(f"Loaded speech model from {speech_model_path}")
        
        if fusion_model_path:
            self.fusion_model = tf.keras.models.load_model(fusion_model_path)
            print(f"Loaded fusion model from {fusion_model_path}")
        
        # Face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Prediction smoothing buffers
        self.facial_predictions = deque(maxlen=smoothing_window)
        self.speech_predictions = deque(maxlen=smoothing_window)
        
        # Threading and state
        self.is_running = False
        self.video_capture = None
        self.audio_stream = None
        self.pyaudio_instance = None
        
        # Audio buffer
        self.audio_buffer = queue.Queue()
        self.audio_chunk_samples = int(sample_rate * audio_chunk_duration)
        
        # Latest results
        self.latest_facial_result = None
        self.latest_speech_result = None
        self.latest_fusion_result = None
    
    def _preprocess_face(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess face region for model input."""
        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, None
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray_image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_input = np.expand_dims(np.expand_dims(face_normalized, axis=-1), axis=0)
        
        return face_input, (x, y, w, h)
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio data for model input."""
        if not LIBROSA_AVAILABLE:
            return None
        
        # Convert to float
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Extract MFCC
        n_mfcc = 40
        max_length = 100
        
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=n_mfcc)
        
        # Pad or truncate
        if mfcc.shape[1] < max_length:
            pad_width = max_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_length]
        
        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        # Transpose and add batch dimension
        return np.expand_dims(mfcc.T, axis=0)
    
    def _smooth_predictions(
        self,
        predictions: deque,
        emotions: list
    ) -> Tuple[str, float, Dict[str, float]]:
        """Apply temporal smoothing to predictions."""
        if len(predictions) == 0:
            return None, 0.0, {}
        
        # Average predictions
        avg_probs = np.mean(list(predictions), axis=0)
        emotion_idx = np.argmax(avg_probs)
        emotion = emotions[emotion_idx]
        confidence = float(avg_probs[emotion_idx])
        
        probs_dict = {e: float(p) for e, p in zip(emotions, avg_probs)}
        
        return emotion, confidence, probs_dict
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream."""
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.audio_buffer.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def _process_audio_thread(self):
        """Thread for processing audio data."""
        audio_samples = []
        
        while self.is_running:
            try:
                # Get audio chunk from buffer
                chunk = self.audio_buffer.get(timeout=0.1)
                audio_samples.extend(chunk)
                
                # Process when we have enough samples
                if len(audio_samples) >= self.audio_chunk_samples:
                    audio_data = np.array(audio_samples[:self.audio_chunk_samples])
                    audio_samples = audio_samples[self.audio_chunk_samples // 2:]  # Overlap
                    
                    if self.speech_model is not None:
                        # Preprocess and predict
                        audio_input = self._preprocess_audio(audio_data)
                        
                        if audio_input is not None:
                            predictions = self.speech_model.predict(audio_input, verbose=0)[0]
                            self.speech_predictions.append(predictions)
                            
                            # Update latest result
                            emotion, confidence, probs = self._smooth_predictions(
                                self.speech_predictions,
                                self.SPEECH_EMOTIONS
                            )
                            self.latest_speech_result = {
                                'emotion': emotion,
                                'confidence': confidence,
                                'probabilities': probs
                            }
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")
    
    def _draw_results(
        self,
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Draw emotion recognition results on frame."""
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw face rectangle and facial emotion
        if face_bbox is not None and self.latest_facial_result is not None:
            x, y, w, h = face_bbox
            emotion = self.latest_facial_result['emotion']
            confidence = self.latest_facial_result['confidence']
            color = self.EMOTION_COLORS.get(emotion, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw emotion label
            label = f"{emotion}: {confidence:.0%}"
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Background rectangle for text
            cv2.rectangle(
                display_frame,
                (x, y - text_height - 10),
                (x + text_width + 10, y),
                color,
                -1
            )
            cv2.putText(
                display_frame,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        # Draw speech emotion panel
        panel_x = width - 250
        panel_y = 10
        panel_width = 240
        panel_height = 100
        
        # Semi-transparent background
        overlay = display_frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # Speech emotion text
        cv2.putText(
            display_frame,
            "Speech Emotion:",
            (panel_x + 10, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        if self.latest_speech_result is not None:
            speech_emotion = self.latest_speech_result['emotion']
            speech_confidence = self.latest_speech_result['confidence']
            color = self.EMOTION_COLORS.get(speech_emotion, (255, 255, 255))
            
            cv2.putText(
                display_frame,
                f"{speech_emotion}",
                (panel_x + 10, panel_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )
            cv2.putText(
                display_frame,
                f"Confidence: {speech_confidence:.0%}",
                (panel_x + 10, panel_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
        else:
            cv2.putText(
                display_frame,
                "No audio",
                (panel_x + 10, panel_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (128, 128, 128),
                1
            )
        
        # Instructions
        cv2.putText(
            display_frame,
            "Press 'q' to quit, 's' to save screenshot",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        return display_frame
    
    def start(
        self,
        window_name: str = "Multimodal Emotion Recognition",
        show_fps: bool = True,
        callback: Optional[Callable[[Dict], None]] = None
    ):
        """
        Start real-time emotion recognition.
        
        Args:
            window_name: Name of display window
            show_fps: Whether to show FPS counter
            callback: Optional callback function called with results each frame
        """
        if self.facial_model is None and self.speech_model is None:
            raise ValueError("At least one model (facial or speech) must be loaded")
        
        self.is_running = True
        
        # Start video capture
        self.video_capture = cv2.VideoCapture(self.camera_id)
        if not self.video_capture.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        # Start audio capture
        audio_thread = None
        if PYAUDIO_AVAILABLE and self.speech_model is not None:
            try:
                self.pyaudio_instance = pyaudio.PyAudio()
                self.audio_stream = self.pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=1024,
                    stream_callback=self._audio_callback
                )
                self.audio_stream.start_stream()
                
                # Start audio processing thread
                audio_thread = threading.Thread(target=self._process_audio_thread)
                audio_thread.start()
            except Exception as e:
                print(f"Could not start audio capture: {e}")
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        print(f"\nStarting real-time emotion recognition...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        try:
            while self.is_running:
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Process facial emotion
                face_bbox = None
                if self.facial_model is not None:
                    face_input, face_bbox = self._preprocess_face(gray)
                    
                    if face_input is not None:
                        predictions = self.facial_model.predict(face_input, verbose=0)[0]
                        self.facial_predictions.append(predictions)
                        
                        # Update latest result
                        emotion, confidence, probs = self._smooth_predictions(
                            self.facial_predictions,
                            self.FACIAL_EMOTIONS
                        )
                        self.latest_facial_result = {
                            'emotion': emotion,
                            'confidence': confidence,
                            'probabilities': probs
                        }
                
                # Draw results
                display_frame = self._draw_results(frame, face_bbox)
                
                # Calculate and display FPS
                fps_counter += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    current_fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_start_time = time.time()
                
                if show_fps:
                    cv2.putText(
                        display_frame,
                        f"FPS: {current_fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                
                # Call callback if provided
                if callback is not None:
                    callback({
                        'facial': self.latest_facial_result,
                        'speech': self.latest_speech_result,
                        'frame': frame,
                        'fps': current_fps
                    })
                
                # Display
                cv2.imshow(window_name, display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"emotion_capture_{timestamp}.png"
                    cv2.imwrite(filename, display_frame)
                    print(f"Screenshot saved: {filename}")
        
        finally:
            self.stop()
            if audio_thread is not None:
                audio_thread.join(timeout=1.0)
    
    def stop(self):
        """Stop real-time recognition."""
        self.is_running = False
        
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        
        if self.audio_stream is not None:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        
        if self.pyaudio_instance is not None:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None
        
        cv2.destroyAllWindows()
        print("Real-time recognition stopped")
    
    def get_latest_results(self) -> Dict:
        """Get the latest emotion recognition results."""
        return {
            'facial': self.latest_facial_result,
            'speech': self.latest_speech_result,
            'fusion': self.latest_fusion_result
        }


def main():
    """Example usage of RealtimeEmotionRecognizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Emotion Recognition')
    parser.add_argument('--facial-model', type=str, help='Path to facial model')
    parser.add_argument('--speech-model', type=str, help='Path to speech model')
    parser.add_argument('--fusion-model', type=str, help='Path to fusion model')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--sample-rate', type=int, default=22050, help='Audio sample rate')
    
    args = parser.parse_args()
    
    # Check if at least one model is provided
    if not any([args.facial_model, args.speech_model, args.fusion_model]):
        print("Error: At least one model path must be provided")
        print("Usage: python realtime.py --facial-model path/to/model.h5")
        return
    
    # Initialize recognizer
    recognizer = RealtimeEmotionRecognizer(
        facial_model_path=args.facial_model,
        speech_model_path=args.speech_model,
        fusion_model_path=args.fusion_model,
        camera_id=args.camera,
        sample_rate=args.sample_rate
    )
    
    # Start recognition
    recognizer.start()


if __name__ == '__main__':
    main()
