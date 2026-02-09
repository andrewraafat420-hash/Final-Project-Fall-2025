import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
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

DATASET_PATH = r"C:\Users\andre\Downloads\Compressed\training_set"  # Path to your dataset folder

IMG_HEIGHT = 150
IMG_WIDTH = 150

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
# تعريف طبقة زيادة البيانات
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
layers.RandomRotation(0.05),   # تدوير بسيط
    layers.RandomZoom(0.05),     # تقريب
    layers.RandomTranslation(0.05, 0.05), # تحريك بسيط
    layers.RandomContrast(0.2), # مهم جداً لصور العيون
    layers.RandomBrightness(0.2),
    # بنحرك الصورة بنسبة بسيطه عشان الموديل ميتعودش على المركز
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
])

# --- 5. BUILD MODEL ---

model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # Data Augmentation Layer
    data_augmentation,
    #layers.GaussianNoise(0.1),
    layers.Rescaling(1./255),
    
    # Block 1
    layers.Conv2D(32, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),

    # Block 2
    layers.Conv2D(64, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),

    # Block 3 
    layers.Conv2D(128, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, 3, padding='same', use_bias=False), # طبقة زيادة للدقة
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),

    # Block 4 (الأخير - هنقف عند 256 عشان نوفر رامات)
    layers.Conv2D(256, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    
    # تقنية الـ Global Average Pooling (ممتازة للـ 4GB VRAM لأنها بتلغي الـ Flatten التقيلة)
    layers.GlobalAveragePooling2D(),

    # طبقات التصنيف
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dropout(0.3),
    
    
    
    layers.Dense(2, activation='softmax') 
])
# --- 6. CALLBACKS ---

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


model.summary()

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy', 
    factor=0.5,   
    patience=5,    # لو متحسنش لمدة 5 مرات
    min_lr=0.00001,
    verbose=1
)
early_stopping = callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    filepath='cell_analyzer_model.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# --- 7. TRAIN ---
print("\nSTARTING TRAINING...")

# Example class weights to handle class imbalance

class_weights = {0: 1, 1: 1}  # Adjust weights as necessary

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint, lr_scheduler],
    class_weight=class_weights
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