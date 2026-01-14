import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os

# --- 1. GPU CONFIGURATION (CRITICAL FOR NVIDIA) ---
# This block checks for the GPU and prevents it from crashing by managing memory better.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ SUCCESS: NVIDIA GPU Detected: {len(gpus)} found")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ WARNING: No NVIDIA GPU detected. Running on CPU (will be slow).")

# --- 2. CONFIGURATION ---
# UPDATE THIS PATH to your actual folder
DATASET_PATH = r"C:\Users\YourName\Desktop\dataset" 
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
EPOCHS = 20  # Increased because EarlyStopping will stop it if needed

# --- 3. DATA LOADING PIPELINE ---
print("LOADING DATA...")

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

# PERFORMANCE BOOST: Cache and Prefetch
# This keeps the GPU busy while the CPU prepares the next batch of images.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 4. DATA AUGMENTATION ---
# This creates "fake" new images by rotating/flipping existing ones.
# It helps the AI generalize better and not just memorize the specific images.
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
])

# --- 5. BUILD THE MODEL ---
model = models.Sequential([
    # Add Augmentation layer first
    data_augmentation,
    
    # Rescale pixel values
    layers.Rescaling(1./255),
    
    # Convolutional Block 1
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Convolutional Block 2
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Convolutional Block 3
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Flatten
    layers.Flatten(),
    
    # Dense Layers
    layers.Dense(256, activation='relu'),
    
    # Dropout: Randomly turns off neurons during training to prevent overfitting
    layers.Dropout(0.5),
    
    layers.Dense(2) # Output layer (Healthy vs Damaged)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# --- 6. CALLBACKS (SMART TRAINING) ---
# Stop training if validation loss doesn't improve for 3 epochs
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=4, 
    restore_best_weights=True
)

# Save the best model automatically
checkpoint = callbacks.ModelCheckpoint(
    filepath='best_cell_model.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# --- 7. TRAIN ---
print("\nSTARTING TRAINING ON CUDA CORES...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

# --- 8. VISUALIZATION ---
# Plot accuracy and loss to see how well the model learned
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

print("\nTraining Complete! Best model saved as 'best_cell_model.h5'")