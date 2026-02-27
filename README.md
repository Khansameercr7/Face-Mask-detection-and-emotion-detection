# 🎭 Face Mask & Emotion Detection

Real-time face mask detection and emotion recognition using custom CNNs built with TensorFlow/Keras + OpenCV.

## 📁 Project Structure
```
face-detection/
├── models.py          ← CNN architectures (EmotionNet + MaskNet)
├── prepare_data.py    ← Dataset setup + FER-2013 CSV parser
├── train.py           ← Full training pipeline
├── detect.py          ← Real-time OpenCV webcam inference
├── demo.html          ← Live browser demo (no Python needed!)
├── requirements.txt
└── README.md
```

## ⚡ Quick Start
```bash
pip install -r requirements.txt
python train.py --task both --synthetic    # quick test
python train.py --task both --epochs 50   # real training
python detect.py                          # live webcam
python detect.py --source image.jpg       # image inference
```

## 🧠 Models
- EmotionNet: 7-class emotion CNN, ~450K params, 48x48 grayscale input
- MaskNet: Binary mask CNN, ~600K params, 128x128 RGB input, MobileNet-style

## 📊 Datasets
- Emotions: FER-2013 (Kaggle, 35K images)
- Mask: RMFD or Kaggle Face Mask dataset

## 🌐 Browser Demo
Open demo.html — runs face-api.js in browser, no Python needed.
