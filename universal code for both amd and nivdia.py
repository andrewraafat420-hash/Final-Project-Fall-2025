import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os
import sys

# --- 1. UNIVERSAL HARDWARE DETECTION ---
# This block automatically adjusts settings based on which card is found.

print(f"Python Version: {sys.version}")
print(f"TensorFlow Version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✅ GPU DETECTED: {len(gpus)} found")
    for gpu in gpus:
        print(f"   Device: {gpu.name}")
        
    # ATTEMPT MEMORY GROWTH (Essential for NVIDIA, harmless for AMD)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("   Memory Growth: ENABLED (Prevents VRAM crashes)")
    except RuntimeError as e:
        # If DirectML (AMD) doesn't support this specific flag, we just ignore it.
        print(f"   Note: Memory growth setting skipped ({e})")
else:
    print("\n⚠️ WARNING: No GPU detected. Training will be SLOW (CPU only).")

# --- 2. CONFIGURATION ---
# UPDATE THIS PATH! 

DATASET_PATH = r"F:\Games\coooooding\Dataset\training_set" 

IMG_HEIGHT = 180
IMG_WIDTH = 180

# BATCH SIZE STRATEGY:
# - Use 16 for AMD with 4GB VRAM.
# - NVIDIA can probably increase this to 32 or 64.
BATCH_SIZE = 16 
EPOCHS = 20

# --- 3. DATA LOADING ---
print("\nLOADING DATA...")

# We use CPU for loading to prevent GPU bottlenecks on some systems
with tf.device('/CPU:0'):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

# OPTIMIZATION
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 4. DATA AUGMENTATION ---
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
])

# --- 5. BUILD MODEL ---
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    
    # Block 1
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Block 2
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Block 3
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Flatten & Dense
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), # Critical for preventing overfitting
    layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# --- 6. CALLBACKS ---
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=4, 
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    filepath='cell_analyzer_model.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# --- 7. TRAIN ---
print("\nSTARTING TRAINING...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

# --- 8. VISUALIZATION ---
# This block handles potential errors if training stops too early to plot
try:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
except Exception as e:
    print(f"Skipping plot generation: {e}")

print("\nDone! Model saved as 'cell_analyzer_model.h5'")