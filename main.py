"""
Main entry point for Multimodal Emotion Recognition System.

This script provides a unified interface for:
- Training facial emotion recognition model
- Training speech emotion recognition model
- Training multimodal fusion model
- Running inference on images/audio
- Real-time emotion recognition
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def train_facial(args):
    """Train facial emotion recognition model."""
    from src.training.train_facial import FacialTrainer
    from src.utils.helpers import load_config, setup_logging
    
    config = load_config(args.config)
    logger = setup_logging(log_dir=config.get('paths', {}).get('logs', 'outputs/logs'))
    
    logger.info("Starting facial emotion recognition training...")
    
    trainer = FacialTrainer(config)
    history = trainer.train()
    
    if args.evaluate:
        trainer.evaluate()
    
    trainer.save_model(args.output or config['paths']['models'] + '/facial')
    logger.info("Facial model training completed!")
    
    return history


def train_speech(args):
    """Train speech emotion recognition model."""
    from src.training.train_speech import SpeechTrainer
    from src.utils.helpers import load_config, setup_logging
    
    config = load_config(args.config)
    logger = setup_logging(log_dir=config.get('paths', {}).get('logs', 'outputs/logs'))
    
    logger.info("Starting speech emotion recognition training...")
    
    trainer = SpeechTrainer(config)
    history = trainer.train()
    
    if args.evaluate:
        trainer.evaluate()
    
    trainer.save_model(args.output or config['paths']['models'] + '/speech')
    logger.info("Speech model training completed!")
    
    return history


def train_fusion(args):
    """Train multimodal fusion model."""
    from src.training.train_fusion import FusionTrainer
    from src.utils.helpers import load_config, setup_logging
    
    config = load_config(args.config)
    logger = setup_logging(log_dir=config.get('paths', {}).get('logs', 'outputs/logs'))
    
    logger.info("Starting multimodal fusion training...")
    
    trainer = FusionTrainer(
        config,
        facial_model_path=args.facial_model,
        speech_model_path=args.speech_model
    )
    history = trainer.train()
    
    if args.evaluate:
        trainer.evaluate()
    
    trainer.save_model(args.output or config['paths']['models'] + '/fusion')
    logger.info("Fusion model training completed!")
    
    return history


def predict(args):
    """Run prediction on image/audio files."""
    from src.inference.predictor import EmotionPredictor
    
    predictor = EmotionPredictor(
        facial_model_path=args.facial_model,
        speech_model_path=args.speech_model,
        fusion_model_path=args.fusion_model
    )
    
    results = predictor.predict_from_file(
        image_path=args.image,
        audio_path=args.audio,
        mode=args.mode
    )
    
    print("\n" + "="*50)
    print("EMOTION PREDICTION RESULTS")
    print("="*50)
    
    for modality, result in results.items():
        print(f"\n{modality.upper()}:")
        print(f"  Predicted Emotion: {result['emotion']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print("  All Probabilities:")
        for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            print(f"    {emotion:12s}: {prob:.2%} {bar}")
    
    return results


def realtime(args):
    """Run real-time emotion recognition."""
    from src.inference.realtime import RealtimeEmotionRecognizer
    
    recognizer = RealtimeEmotionRecognizer(
        facial_model_path=args.facial_model,
        speech_model_path=args.speech_model,
        fusion_model_path=args.fusion_model,
        camera_id=args.camera,
        sample_rate=args.sample_rate
    )
    
    print("\n" + "="*50)
    print("REAL-TIME EMOTION RECOGNITION")
    print("="*50)
    print("Starting camera and microphone...")
    print("Press 'q' to quit, 's' to save screenshot")
    print("="*50 + "\n")
    
    recognizer.start()


def evaluate(args):
    """Evaluate trained model."""
    from src.inference.predictor import EmotionPredictor
    from src.utils.helpers import load_config
    from src.utils.visualization import plot_confusion_matrix, create_emotion_report
    import numpy as np
    
    config = load_config(args.config)
    
    # Initialize predictor
    predictor = EmotionPredictor(
        facial_model_path=args.facial_model,
        speech_model_path=args.speech_model
    )
    
    # Load test data based on mode
    if args.mode == 'facial':
        from src.data.fer2013_loader import FER2013DataLoader
        loader = FER2013DataLoader(config)
        X_train, X_val, X_test, y_train, y_val, y_test = loader.load_data()
        
        # Predict
        y_pred = []
        for img in X_test:
            processed = predictor.preprocess_image(img[..., 0] * 255, detect_face=False)
            if processed is not None:
                pred = predictor.facial_model.predict(processed, verbose=0)
                y_pred.append(np.argmax(pred))
            else:
                y_pred.append(0)
        
        y_pred = np.array(y_pred)
        class_names = predictor.FACIAL_EMOTIONS
        
    elif args.mode == 'speech':
        from src.data.ravdess_loader import RAVDESSDataLoader
        loader = RAVDESSDataLoader(config)
        X_train, X_val, X_test, y_train, y_val, y_test = loader.load_data()
        
        y_pred = np.argmax(predictor.speech_model.predict(X_test, verbose=0), axis=1)
        class_names = predictor.SPEECH_EMOTIONS
    
    # Generate report
    report = create_emotion_report(y_test, y_pred, class_names, save_path=args.output)
    print(report)
    
    # Plot confusion matrix
    if args.plot:
        plot_confusion_matrix(
            y_test, y_pred, class_names,
            save_path=args.output.replace('.txt', '_confusion.png') if args.output else None
        )


def main():
    parser = argparse.ArgumentParser(
        description='Multimodal Emotion Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train facial model
  python main.py train-facial --config configs/config.yaml
  
  # Train speech model
  python main.py train-speech --config configs/config.yaml
  
  # Train fusion model (requires pre-trained facial and speech models)
  python main.py train-fusion --config configs/config.yaml \\
      --facial-model models/facial/model.keras \\
      --speech-model models/speech/model.keras
  
  # Predict from image
  python main.py predict --facial-model models/facial/model.keras --image test.jpg
  
  # Predict from audio
  python main.py predict --speech-model models/speech/model.keras --audio test.wav
  
  # Real-time recognition
  python main.py realtime --facial-model models/facial/model.keras
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train facial model
    train_facial_parser = subparsers.add_parser('train-facial', help='Train facial emotion model')
    train_facial_parser.add_argument('--config', type=str, default='configs/config.yaml',
                                     help='Path to configuration file')
    train_facial_parser.add_argument('--output', type=str, help='Output path for model')
    train_facial_parser.add_argument('--evaluate', action='store_true',
                                     help='Evaluate after training')
    
    # Train speech model
    train_speech_parser = subparsers.add_parser('train-speech', help='Train speech emotion model')
    train_speech_parser.add_argument('--config', type=str, default='configs/config.yaml',
                                     help='Path to configuration file')
    train_speech_parser.add_argument('--output', type=str, help='Output path for model')
    train_speech_parser.add_argument('--evaluate', action='store_true',
                                     help='Evaluate after training')
    
    # Train fusion model
    train_fusion_parser = subparsers.add_parser('train-fusion', help='Train multimodal fusion model')
    train_fusion_parser.add_argument('--config', type=str, default='configs/config.yaml',
                                     help='Path to configuration file')
    train_fusion_parser.add_argument('--facial-model', type=str, required=True,
                                     help='Path to trained facial model')
    train_fusion_parser.add_argument('--speech-model', type=str, required=True,
                                     help='Path to trained speech model')
    train_fusion_parser.add_argument('--output', type=str, help='Output path for model')
    train_fusion_parser.add_argument('--evaluate', action='store_true',
                                     help='Evaluate after training')
    
    # Predict
    predict_parser = subparsers.add_parser('predict', help='Predict emotion from file')
    predict_parser.add_argument('--facial-model', type=str, help='Path to facial model')
    predict_parser.add_argument('--speech-model', type=str, help='Path to speech model')
    predict_parser.add_argument('--fusion-model', type=str, help='Path to fusion model')
    predict_parser.add_argument('--image', type=str, help='Path to image file')
    predict_parser.add_argument('--audio', type=str, help='Path to audio file')
    predict_parser.add_argument('--mode', type=str, default='auto',
                               choices=['facial', 'speech', 'fusion', 'auto'],
                               help='Prediction mode')
    
    # Real-time
    realtime_parser = subparsers.add_parser('realtime', help='Real-time emotion recognition')
    realtime_parser.add_argument('--facial-model', type=str, help='Path to facial model')
    realtime_parser.add_argument('--speech-model', type=str, help='Path to speech model')
    realtime_parser.add_argument('--fusion-model', type=str, help='Path to fusion model')
    realtime_parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    realtime_parser.add_argument('--sample-rate', type=int, default=22050, help='Audio sample rate')
    
    # Evaluate
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    evaluate_parser.add_argument('--config', type=str, default='configs/config.yaml',
                                 help='Path to configuration file')
    evaluate_parser.add_argument('--facial-model', type=str, help='Path to facial model')
    evaluate_parser.add_argument('--speech-model', type=str, help='Path to speech model')
    evaluate_parser.add_argument('--mode', type=str, required=True,
                                choices=['facial', 'speech'],
                                help='Evaluation mode')
    evaluate_parser.add_argument('--output', type=str, help='Output path for report')
    evaluate_parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    commands = {
        'train-facial': train_facial,
        'train-speech': train_speech,
        'train-fusion': train_fusion,
        'predict': predict,
        'realtime': realtime,
        'evaluate': evaluate
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
