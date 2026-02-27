"""
train_mask.py  –  Face Mask Detection CNN
==========================================
Dataset expected layout:
    data/mask/
        with_mask/      ← images of masked faces
        without_mask/   ← images of unmasked faces

If not present, a small synthetic dataset is auto-generated for demo.

Run:
    python train_mask.py
"""

import os, warnings, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from models.architectures import build_mask_model

# ── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 30          # EarlyStopping kicks in earlier
LR          = 1e-4
DATA_DIR    = 'data/mask'
MODEL_PATH  = 'models/mask_detector.h5'
REPORT_DIR  = 'reports'
CLASSES     = ['without_mask', 'with_mask']   # alphabetical = keras default

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 62)
print("  FACE MASK DETECTION  –  CNN Training Pipeline")
print("=" * 62)
print(f"  TensorFlow : {tf.__version__}")
print(f"  GPU        : {tf.config.list_physical_devices('GPU')}")

# ─────────────────────────────────────────────────────────────
#  AUTO-GENERATE SYNTHETIC DATA (if real dataset missing)
# ─────────────────────────────────────────────────────────────
def generate_synthetic_mask_data(n_per_class=300):
    """
    Creates synthetic face-like images for quick demo training.
    Replace this with real data (e.g. Kaggle face mask dataset).
    """
    print("\n⚙️   Generating synthetic demo data …")
    for cls_name in CLASSES:
        os.makedirs(f'{DATA_DIR}/{cls_name}', exist_ok=True)
        existing = len(os.listdir(f'{DATA_DIR}/{cls_name}'))
        if existing >= n_per_class:
            continue
        for i in range(n_per_class - existing):
            img = np.random.randint(100, 220, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            # Simulate face-like oval
            cv2.ellipse(img, (IMG_SIZE//2, IMG_SIZE//2), (80, 100), 0, 0, 360, (220,190,160), -1)
            # Eyes
            cv2.circle(img, (IMG_SIZE//2-30, IMG_SIZE//2-20), 10, (40,40,40), -1)
            cv2.circle(img, (IMG_SIZE//2+30, IMG_SIZE//2-20), 10, (40,40,40), -1)
            if cls_name == 'with_mask':
                # Blue surgical mask rectangle
                cv2.rectangle(img, (IMG_SIZE//2-60, IMG_SIZE//2+10),
                              (IMG_SIZE//2+60, IMG_SIZE//2+60), (60,130,200), -1)
                cv2.ellipse(img, (IMG_SIZE//2, IMG_SIZE//2+35), (60, 30), 0, 0, 180, (40,100,170), -1)
            else:
                # Mouth + nose
                cv2.ellipse(img, (IMG_SIZE//2, IMG_SIZE//2+30), (25, 12), 0, 0, 180, (160,80,80), 2)
                cv2.circle(img, (IMG_SIZE//2, IMG_SIZE//2+5), 6, (180,140,130), -1)
            noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(f'{DATA_DIR}/{cls_name}/{i:04d}.jpg', img)
    total = sum(len(os.listdir(f'{DATA_DIR}/{c}')) for c in CLASSES)
    print(f"   ✅  {total} synthetic images ready")


if not os.path.exists(DATA_DIR) or \
   not all(os.path.exists(f'{DATA_DIR}/{c}') and len(os.listdir(f'{DATA_DIR}/{c}')) > 10
           for c in CLASSES):
    generate_synthetic_mask_data()

# ─────────────────────────────────────────────────────────────
#  DATA GENERATORS  (with augmentation)
# ─────────────────────────────────────────────────────────────
print("\n📂  Loading data …")

train_gen = ImageDataGenerator(
    rescale          = 1./255,
    validation_split = 0.2,
    rotation_range   = 20,
    width_shift_range= 0.15,
    height_shift_range=0.15,
    shear_range      = 0.1,
    zoom_range       = 0.2,
    horizontal_flip  = True,
    brightness_range = [0.7, 1.3],
    fill_mode        = 'nearest',
)

train_data = train_gen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='binary',
    subset='training', seed=42,
)
val_data = train_gen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='binary',
    subset='validation', seed=42,
)

print(f"   Classes      : {train_data.class_indices}")
print(f"   Train samples: {train_data.samples}")
print(f"   Val   samples: {val_data.samples}")

# ─────────────────────────────────────────────────────────────
#  BUILD & COMPILE MODEL
# ─────────────────────────────────────────────────────────────
print("\n🏗️   Building MobileNetV2 mask detector …")
model = build_mask_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')],
)
print(f"   Total params: {model.count_params():,}")

# ─────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_auc',
                    mode='max', verbose=1),
    EarlyStopping(monitor='val_auc', patience=6, mode='max',
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                      min_lr=1e-7, verbose=1),
    CSVLogger('reports/mask_training_log.csv'),
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
)

# ─────────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n📊  Evaluating …")
val_data.reset()
y_true, y_pred_prob = [], []
for _ in range(len(val_data)):
    X_b, y_b = next(val_data)
    y_true.extend(y_b)
    y_pred_prob.extend(model.predict(X_b, verbose=0).flatten())

y_true      = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)
y_pred      = (y_pred_prob > 0.5).astype(int)

auc = roc_auc_score(y_true, y_pred_prob)
print(f"\n   AUC-ROC : {auc:.4f}")
print("\n   Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Save metrics
metrics = {
    'auc'     : float(auc),
    'accuracy': float((y_true == y_pred).mean()),
    'model'   : MODEL_PATH,
}
with open('reports/mask_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ─────────────────────────────────────────────────────────────
#  VISUALISATIONS
# ─────────────────────────────────────────────────────────────
PALETTE = ['#00D4FF', '#FF3366', '#00FF88', '#FFB800']

# Training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.patch.set_facecolor('#0d1117')
for ax in axes: ax.set_facecolor('#161b22')

h = history.history
for ax, metric, title in zip(
    axes,
    [['accuracy','val_accuracy'], ['loss','val_loss'], ['auc','val_auc']],
    ['Accuracy', 'Loss', 'AUC-ROC']
):
    ax.plot(h[metric[0]], color=PALETTE[0], linewidth=2, label='Train')
    ax.plot(h[metric[1]], color=PALETTE[1], linewidth=2, linestyle='--', label='Val')
    ax.set_title(title, color='white', fontweight='bold')
    ax.tick_params(colors='#888')
    ax.spines[:].set_color('#333')
    ax.legend(facecolor='#161b22', labelcolor='white')

plt.suptitle('Mask Detection – Training Curves', color='white', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/mask_training_curves.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES,
            cmap='Blues', ax=ax, linewidths=1)
ax.set_title('Confusion Matrix', color='white', fontweight='bold')
ax.tick_params(colors='white')
plt.tight_layout()
plt.savefig('reports/mask_confusion_matrix.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()

print("\n✅  Reports saved → reports/")
print(f"✅  Model saved  → {MODEL_PATH}")
print(f"\n  Final Val Accuracy : {max(h['val_accuracy']):.4f}")
print(f"  Final Val AUC      : {auc:.4f}")
