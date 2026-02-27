"""
train_emotion.py  –  Emotion Detection CNN
==========================================
Dataset expected layout (FER-2013 format):
    data/emotions/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/

Download the real FER-2013 dataset from:
    https://www.kaggle.com/datasets/msambare/fer2013

If not present, a synthetic grayscale dataset is auto-generated for demo.

Run:
    python train_emotion.py
"""

import os, warnings, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

from models.architectures import build_emotion_model

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 48
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-3
DATA_DIR    = 'data/emotions'
MODEL_PATH  = 'models/emotion_detector.h5'
REPORT_DIR  = 'reports'

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOJI_MAP = {
    'angry'   : '😠',
    'disgust' : '🤢',
    'fear'    : '😨',
    'happy'   : '😊',
    'neutral' : '😐',
    'sad'     : '😢',
    'surprise': '😲',
}

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 62)
print("  EMOTION DETECTION  –  CNN Training Pipeline")
print("=" * 62)
print(f"  TensorFlow : {tf.__version__}")

# ─────────────────────────────────────────────────────────────
#  SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────────────────────
def generate_synthetic_emotion_data(n_per_class=400):
    """
    Generates 48×48 grayscale images with stylised facial feature
    variations per emotion class. Replace with real FER-2013 data.
    """
    print("\n⚙️   Generating synthetic emotion data …")

    emotion_configs = {
        'angry'   : dict(eyebrow_angle=20,  mouth_curve=-15, eye_size=10),
        'disgust' : dict(eyebrow_angle=15,  mouth_curve=-10, eye_size=9),
        'fear'    : dict(eyebrow_angle=-10, mouth_curve=-5,  eye_size=14),
        'happy'   : dict(eyebrow_angle=-5,  mouth_curve=12,  eye_size=10),
        'neutral' : dict(eyebrow_angle=0,   mouth_curve=0,   eye_size=10),
        'sad'     : dict(eyebrow_angle=-20, mouth_curve=-12, eye_size=9),
        'surprise': dict(eyebrow_angle=-25, mouth_curve=18,  eye_size=15),
    }

    for emo, cfg in emotion_configs.items():
        os.makedirs(f'{DATA_DIR}/{emo}', exist_ok=True)
        existing = len(os.listdir(f'{DATA_DIR}/{emo}'))
        to_gen   = max(0, n_per_class - existing)
        for i in range(to_gen):
            img = np.full((IMG_SIZE, IMG_SIZE), 180, dtype=np.uint8)
            cx, cy = IMG_SIZE//2, IMG_SIZE//2
            # Face oval
            cv2.ellipse(img, (cx, cy+2), (18, 22), 0, 0, 360, 210, -1)
            # Eyes
            eye_r = max(2, cfg['eye_size'] // 3)
            cv2.circle(img, (cx-7, cy-5), eye_r, 60, -1)
            cv2.circle(img, (cx+7, cy-5), eye_r, 60, -1)
            # Mouth curve
            curve = cfg['mouth_curve']
            pts = np.array([[cx-6, cy+8], [cx, cy+8+curve//2], [cx+6, cy+8]])
            cv2.polylines(img, [pts], False, 60, 1)
            # Eyebrows with angle variation
            ang = cfg['eyebrow_angle']
            cv2.line(img, (cx-10, cy-9+ang//4), (cx-3, cy-9-ang//4), 60, 1)
            cv2.line(img, (cx+3,  cy-9+ang//4), (cx+10, cy-9-ang//4), 60, 1)
            # Add noise
            noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16)+noise, 0, 255).astype(np.uint8)
            cv2.imwrite(f'{DATA_DIR}/{emo}/{i:05d}.jpg', img)

    total = sum(len(os.listdir(f'{DATA_DIR}/{e}')) for e in EMOTIONS)
    print(f"   ✅  {total} synthetic emotion images ready")


if not os.path.exists(DATA_DIR) or \
   not all(os.path.exists(f'{DATA_DIR}/{e}') and len(os.listdir(f'{DATA_DIR}/{e}')) > 10
           for e in EMOTIONS):
    generate_synthetic_emotion_data()

# ─────────────────────────────────────────────────────────────
#  DATA GENERATORS
# ─────────────────────────────────────────────────────────────
print("\n📂  Loading emotion data …")

train_gen = ImageDataGenerator(
    rescale            = 1./255,
    validation_split   = 0.2,
    rotation_range     = 15,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
    horizontal_flip    = True,
    zoom_range         = 0.15,
    shear_range        = 0.1,
    fill_mode          = 'nearest',
)

def gray_preprocess(img):
    """Convert to grayscale for the 1-channel model."""
    gray = np.mean(img, axis=-1, keepdims=True)
    return gray

train_data = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42,
)
val_data = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42,
)

print(f"   Classes      : {train_data.class_indices}")
print(f"   Train samples: {train_data.samples}")
print(f"   Val   samples: {val_data.samples}")

# Class weight balancing (FER-2013 is imbalanced)
from sklearn.utils.class_weight import compute_class_weight
class_weights_arr = compute_class_weight(
    'balanced',
    classes=np.arange(len(EMOTIONS)),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights_arr))
print(f"   Class weights: {dict((EMOTIONS[k],round(v,2)) for k,v in class_weights.items())}")

# ─────────────────────────────────────────────────────────────
#  BUILD & COMPILE
# ─────────────────────────────────────────────────────────────
print("\n🏗️   Building Emotion CNN …")
model = build_emotion_model(num_classes=len(EMOTIONS), input_shape=(IMG_SIZE, IMG_SIZE, 1))

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')],
)
print(f"   Total params: {model.count_params():,}")

# ─────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy',
                    mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=8, mode='max',
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                      min_lr=1e-7, verbose=1),
    CSVLogger('reports/emotion_training_log.csv'),
]

# ─────────────────────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────────────────────
print("\n🚀  Training …")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
)

# ─────────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n📊  Evaluating …")
val_data.reset()
y_true, y_pred = [], []
for _ in range(len(val_data)):
    X_b, y_b = next(val_data)
    preds = model.predict(X_b, verbose=0)
    y_true.extend(np.argmax(y_b, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(f"\n   Classification Report:")
print(classification_report(y_true, y_pred, target_names=EMOTIONS))

metrics = {
    'accuracy' : float((y_true == y_pred).mean()),
    'model'    : MODEL_PATH,
}
with open('reports/emotion_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ─────────────────────────────────────────────────────────────
#  VISUALISATIONS
# ─────────────────────────────────────────────────────────────
PALETTE = ['#00D4FF', '#FF3366', '#00FF88', '#FFB800']

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#0d1117')
for ax in axes: ax.set_facecolor('#161b22')

h = history.history
axes[0].plot(h['accuracy'],     color=PALETTE[0], lw=2, label='Train Acc')
axes[0].plot(h['val_accuracy'], color=PALETTE[1], lw=2, ls='--', label='Val Acc')
axes[0].plot(h['top2_acc'],     color=PALETTE[2], lw=1.5, ls=':', label='Train Top-2')
axes[0].set_title('Accuracy', color='white', fontweight='bold')
axes[0].legend(facecolor='#161b22', labelcolor='white')
axes[0].tick_params(colors='#888')

axes[1].plot(h['loss'],     color=PALETTE[0], lw=2, label='Train Loss')
axes[1].plot(h['val_loss'], color=PALETTE[1], lw=2, ls='--', label='Val Loss')
axes[1].set_title('Loss', color='white', fontweight='bold')
axes[1].legend(facecolor='#161b22', labelcolor='white')
axes[1].tick_params(colors='#888')

plt.suptitle('Emotion Detection – Training Curves', color='white',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/emotion_training_curves.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()

# Confusion matrix
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2f',
            xticklabels=[f"{EMOJI_MAP[e]} {e}" for e in EMOTIONS],
            yticklabels=[f"{EMOJI_MAP[e]} {e}" for e in EMOTIONS],
            cmap='Blues', ax=ax, linewidths=0.5)
ax.set_title('Emotion Confusion Matrix (normalised)', color='white', fontweight='bold')
ax.tick_params(colors='white')
plt.tight_layout()
plt.savefig('reports/emotion_confusion_matrix.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()

print("\n✅  Reports saved → reports/")
print(f"✅  Model saved  → {MODEL_PATH}")
print(f"\n  Final Val Accuracy : {max(h['val_accuracy']):.4f}")
