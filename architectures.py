"""
models/architectures.py
=======================
CNN architectures for:
  1. Face Mask Detection  (Binary: with_mask / without_mask)
  2. Emotion Detection    (7 classes: angry, disgust, fear, happy, neutral, sad, surprise)

Both use MobileNetV2 as a pretrained base (transfer learning)
with custom classification heads fine-tuned for each task.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2


# ─────────────────────────────────────────────────────────────
#  1.  MASK DETECTION MODEL
# ─────────────────────────────────────────────────────────────

def build_mask_model(input_shape=(224, 224, 3), fine_tune_from=100):
    """
    Transfer learning on MobileNetV2 for binary mask classification.

    Architecture:
      MobileNetV2 (pretrained, partial fine-tuning)
      → GlobalAveragePooling2D
      → Dense(256) + BatchNorm + Dropout
      → Dense(128) + BatchNorm + Dropout
      → Dense(1, sigmoid)   ← binary output

    Parameters
    ----------
    input_shape     : (H, W, C) — default 224×224 RGB
    fine_tune_from  : unfreeze layers from this index onwards
    """
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze early layers, fine-tune later ones
    base.trainable = True
    for layer in base.layers[:fine_tune_from]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation='sigmoid', name='mask_output')(x)

    model = models.Model(inputs, outputs, name='MaskDetector')
    return model


# ─────────────────────────────────────────────────────────────
#  2.  EMOTION DETECTION MODEL
# ─────────────────────────────────────────────────────────────

def build_emotion_model(num_classes=7, input_shape=(48, 48, 1)):
    """
    Custom lightweight CNN for emotion recognition from grayscale face crops.

    Designed for 48×48 grayscale (FER-2013 format) — fast & efficient.

    Architecture:
      Block 1: Conv(64) → BN → Conv(64) → BN → MaxPool → Dropout
      Block 2: Conv(128) → BN → Conv(128) → BN → MaxPool → Dropout
      Block 3: Conv(256) → BN → Conv(256) → BN → MaxPool → Dropout
      Block 4: Conv(512) → BN → MaxPool → Dropout
      Flatten → Dense(512) → BN → Dropout
      Dense(256) → BN → Dropout
      Dense(num_classes, softmax)

    Parameters
    ----------
    num_classes  : 7 for standard FER dataset
    input_shape  : (48, 48, 1) grayscale
    """
    inputs = tf.keras.Input(shape=input_shape)

    # ── Block 1 ──
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 2 ──
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 3 ──
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # ── Block 4 ──
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # ── Classifier ──
    x = layers.Flatten()(x)
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax', name='emotion_output')(x)

    model = models.Model(inputs, outputs, name='EmotionDetector')
    return model


# ─────────────────────────────────────────────────────────────
#  3.  COMBINED  (optional: single model, dual head)
# ─────────────────────────────────────────────────────────────

def build_combined_model(input_shape=(128, 128, 3), num_emotions=7, fine_tune_from=80):
    """
    Single model with two output heads:
      - mask_output   : sigmoid (binary)
      - emotion_output: softmax (7-class)

    Useful for simultaneous inference on a single face crop.
    """
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base.trainable = True
    for layer in base.layers[:fine_tune_from]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    feat = base(inputs, training=False)
    feat = layers.GlobalAveragePooling2D()(feat)
    feat = layers.Dense(512)(feat)
    feat = layers.BatchNormalization()(feat)
    feat = layers.Activation('relu')(feat)
    feat = layers.Dropout(0.4)(feat)

    # Mask head
    mask_branch = layers.Dense(128, activation='relu')(feat)
    mask_out    = layers.Dense(1, activation='sigmoid', name='mask_output')(mask_branch)

    # Emotion head
    emo_branch = layers.Dense(256, activation='relu')(feat)
    emo_branch = layers.Dropout(0.3)(emo_branch)
    emo_out    = layers.Dense(num_emotions, activation='softmax', name='emotion_output')(emo_branch)

    model = models.Model(inputs, [mask_out, emo_out], name='CombinedFaceAnalyser')
    return model


# ─────────────────────────────────────────────────────────────
#  Quick sanity-check
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n── Mask Detector ──")
    m = build_mask_model()
    m.summary()
    print(f"\nTrainable params : {sum(tf.size(v).numpy() for v in m.trainable_variables):,}")

    print("\n── Emotion Detector ──")
    e = build_emotion_model()
    e.summary()
    print(f"\nTrainable params : {sum(tf.size(v).numpy() for v in e.trainable_variables):,}")

    print("\n── Combined Model ──")
    c = build_combined_model()
    c.summary()
