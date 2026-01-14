import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("cell_model v4.keras")

# --- STEP 1: ENABLE CPU OPTIMIZATIONS (Must be before importing TensorFlow) ---
# This enables "OneDNN", which automatically uses the best instructions 
# My processor has (SSE4.2 and AVX) to speed up math.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# Suppress warnings about missing AVX2 (since my CPU physically doesn't have it)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# --- STEP 2: AMD GPU SETUP ---
print("Checking hardware...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"SUCCESS: Training will run on AMD GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("WARNING: GPU not found. Ensure 'tensorflow-directml-plugin' is installed.")

# --- STEP 3: CONFIGURE DATASET ---
DATASET_PATH = r"F:\Games\coooooding\Dataset\training_set" 
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32

print("\nPREPARING DATA PIPELINE...")

# Load images
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

# --- STEP 4: MAXIMIZE MY CPU PERFORMANCE ---

AUTOTUNE = tf.data.AUTOTUNE

# cache(): Keeps images in RAM after the first time they are loaded.
# prefetch(): Uses my CPU's AVX instructions to load the NEXT batch 
#             while the GPU is still training on the CURRENT batch.
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- STEP 5: BUILD & TRAIN MODEL ---
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("\nSTARTING TRAINING...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)