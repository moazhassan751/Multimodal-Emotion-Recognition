# Multimodal Emotion Recognition Using Deep Learning

A comprehensive deep learning system for emotion recognition that combines **Facial Expression Analysis** and **Speech Emotion Recognition** using multimodal fusion techniques.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This project implements a multimodal emotion recognition system that leverages both visual (facial expressions) and audio (speech) modalities to achieve more accurate and robust emotion detection. The system can:

- Recognize **7 emotions** from facial expressions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- Recognize **8 emotions** from speech (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised)
- Combine both modalities using advanced fusion techniques for improved accuracy
- Perform **real-time emotion recognition** using webcam and microphone

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input Layer                                   │
│         ┌─────────────────┬─────────────────┐                   │
│         │  Facial Image   │   Audio Signal  │                   │
│         │   (48x48x1)     │    (MFCC)       │                   │
│         └────────┬────────┴────────┬────────┘                   │
│                  │                 │                             │
│         ┌────────▼────────┐ ┌──────▼──────┐                     │
│         │   CNN Model     │ │ LSTM/CNN    │                     │
│         │ (VGG16/ResNet)  │ │   Model     │                     │
│         └────────┬────────┘ └──────┬──────┘                     │
│                  │                 │                             │
│         ┌────────▼─────────────────▼────────┐                   │
│         │         Fusion Layer              │                   │
│         │  (Feature/Decision/Attention)     │                   │
│         └────────────────┬──────────────────┘                   │
│                          │                                       │
│                  ┌───────▼───────┐                              │
│                  │  Dense Layers │                              │
│                  └───────┬───────┘                              │
│                          │                                       │
│                  ┌───────▼───────┐                              │
│                  │   Softmax     │                              │
│                  │  (Emotions)   │                              │
│                  └───────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Multimodal-Emotion-Recognition/
├── configs/
│   └── config.yaml              # Main configuration file
├── data/
│   ├── fer2013/                 # FER2013 facial expression dataset
│   └── ravdess/                 # RAVDESS speech emotion dataset
├── models/
│   ├── facial/                  # Saved facial models
│   ├── speech/                  # Saved speech models
│   └── fusion/                  # Saved fusion models
├── notebooks/                   # Jupyter notebooks for exploration
├── outputs/
│   ├── checkpoints/             # Training checkpoints
│   ├── logs/                    # Training logs
│   └── results/                 # Evaluation results
├── src/
│   ├── data/
│   │   ├── fer2013_loader.py    # FER2013 dataset loader
│   │   ├── ravdess_loader.py    # RAVDESS dataset loader
│   │   └── preprocessing.py     # Data preprocessing utilities
│   ├── models/
│   │   ├── facial_model.py      # Facial emotion CNN models
│   │   ├── speech_model.py      # Speech emotion LSTM/CNN models
│   │   └── fusion_model.py      # Multimodal fusion models
│   ├── training/
│   │   ├── train_facial.py      # Facial model training
│   │   ├── train_speech.py      # Speech model training
│   │   └── train_fusion.py      # Fusion model training
│   ├── inference/
│   │   ├── predictor.py         # Emotion prediction module
│   │   └── realtime.py          # Real-time recognition
│   └── utils/
│       ├── helpers.py           # Helper functions
│       └── visualization.py     # Visualization utilities
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Webcam (for real-time recognition)
- Microphone (for speech emotion recognition)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Multimodal-Emotion-Recognition.git
cd Multimodal-Emotion-Recognition
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download datasets**

- **FER2013**: Download from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and extract to `data/fer2013/`
- **RAVDESS**: Download from [Zenodo](https://zenodo.org/record/1188976) and extract to `data/ravdess/`

### Training Models

**Train Facial Emotion Model:**
```bash
python main.py train-facial --config configs/config.yaml --evaluate
```

**Train Speech Emotion Model:**
```bash
python main.py train-speech --config configs/config.yaml --evaluate
```

**Train Fusion Model:**
```bash
python main.py train-fusion --config configs/config.yaml \
    --facial-model models/facial/model.keras \
    --speech-model models/speech/model.keras \
    --evaluate
```

### Inference

**Predict from Image:**
```bash
python main.py predict --facial-model models/facial/model.keras --image path/to/image.jpg
```

**Predict from Audio:**
```bash
python main.py predict --speech-model models/speech/model.keras --audio path/to/audio.wav
```

**Multimodal Prediction:**
```bash
python main.py predict \
    --facial-model models/facial/model.keras \
    --speech-model models/speech/model.keras \
    --fusion-model models/fusion/model.keras \
    --image path/to/image.jpg \
    --audio path/to/audio.wav
```

### Real-Time Recognition

```bash
python main.py realtime \
    --facial-model models/facial/model.keras \
    --speech-model models/speech/model.keras
```

## 📊 Datasets

### FER2013 (Facial Expression Recognition)
- **Size**: ~35,000 grayscale images (48x48 pixels)
- **Classes**: 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Split**: Training (28,709), Validation (3,589), Test (3,589)

### RAVDESS (Ryerson Audio-Visual Database)
- **Size**: 7,356 audio files
- **Actors**: 24 professional actors (12 male, 12 female)
- **Classes**: 8 emotions (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised)
- **Format**: WAV audio files

## 🧠 Model Architectures

### Facial Emotion Model
- **Custom CNN**: 4 convolutional blocks with batch normalization and dropout
- **VGG16**: Pre-trained on ImageNet, fine-tuned for emotion recognition
- **ResNet50**: Deep residual network with skip connections
- **MobileNet**: Lightweight model for mobile/edge deployment

### Speech Emotion Model
- **LSTM**: Bidirectional LSTM layers for temporal modeling
- **CNN**: 1D convolutional layers for MFCC features
- **CNN-LSTM**: Hybrid architecture combining both approaches

### Fusion Strategies
- **Feature-level**: Concatenate features from both modalities
- **Decision-level**: Combine predictions using weighted averaging
- **Attention-based**: Learn dynamic weights for each modality

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Model settings
facial_model:
  architecture: 'custom'  # Options: custom, vgg16, resnet50, mobilenet
  input_shape: [48, 48, 1]
  num_classes: 7

speech_model:
  architecture: 'lstm'  # Options: lstm, cnn, cnn_lstm
  n_mfcc: 40
  num_classes: 8

fusion_model:
  strategy: 'attention'  # Options: feature, decision, attention

# Training settings
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 10
```

## 📈 Results

| Model | Dataset | Accuracy | F1-Score |
|-------|---------|----------|----------|
| Facial CNN | FER2013 | ~68% | ~0.65 |
| Speech LSTM | RAVDESS | ~75% | ~0.73 |
| Multimodal Fusion | Combined | ~80% | ~0.78 |

*Note: Results may vary based on hyperparameters and training configuration.*

## 🔧 API Usage

```python
from src.inference.predictor import EmotionPredictor

# Initialize predictor
predictor = EmotionPredictor(
    facial_model_path='models/facial/model.keras',
    speech_model_path='models/speech/model.keras'
)

# Predict from image
result = predictor.predict_from_file(image_path='face.jpg')
print(f"Emotion: {result['facial']['emotion']}")
print(f"Confidence: {result['facial']['confidence']:.2%}")

# Predict from audio
result = predictor.predict_from_file(audio_path='speech.wav')
print(f"Emotion: {result['speech']['emotion']}")
```

## 📚 References

1. Goodfellow, I.J., et al. (2013). "Challenges in Representation Learning: A report on three machine learning contests." *Neural Information Processing*.

2. Livingstone, S.R., & Russo, F.A. (2018). "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)." *PLoS ONE*.

3. Poria, S., et al. (2017). "A review of affective computing: From unimodal analysis to multimodal fusion." *Information Fusion*.

4. Tzirakis, P., et al. (2017). "End-to-End Multimodal Emotion Recognition using Deep Neural Networks." *IEEE Journal of Selected Topics in Signal Processing*.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ **Star this repository if you find it helpful!**
Multimodal Emotion Recognition: A deep learning project fusing CNN-based facial analysis (FER2013) and LSTM/CNN speech processing (RAVDESS) for real-time, holistic emotion detection.
