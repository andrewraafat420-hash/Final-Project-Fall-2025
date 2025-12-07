import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --- CONFIGURATION ---
# Replace this with the actual path to your folders
DATASET_PATH = r"C:\Users\YourName\Desktop\dataset" 
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32

print("LOADING DATA...")

# 1. Load images and split them into Training (80%) and Validation (20%)
# This allows the AI to learn on one set and test itself on the other.
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

# 2. Build the AI Model (The "Brain")
# We use Conv2D layers to detect patterns (edges, shapes) in the cells.
model = models.Sequential([
    # Rescale pixel values from 0-255 to 0-1 (easier for AI to do math)
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # First layer: Look for simple features (edges of the cell)
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Second layer: Look for complex features (nucleus shape, texture)
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Flatten: Turn the 2D image data into a 1D list of numbers
    layers.Flatten(),
    
    # Dense: Make the final decision
    layers.Dense(128, activation='relu'),
    layers.Dense(2) # Output: 2 neurons (one for Healthy, one for Damaged)
])

# 3. Compile the Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 4. Train the AI
print("\nSTARTING TRAINING...")
# epochs=10 means it will study the entire dataset 10 times.
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# 5. Save the trained brain
model.save("cell_analyzer_model.h5")
print("\nTraining Complete! Model saved as 'cell_analyzer_model.h5'")
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("cell_analyzer_model.h5")

# The names of your folders (must match exactly)
class_names = ['damaged_cells', 'healthy_cells']

def predict_cell(image_path):
    img = tf.keras.utils.load_img(
        image_path, target_size=(180, 180)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This cell is most likely {} ({:.2f}% confidence)."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

# Test it on a new image
predict_cell("new_patient_sample.jpg")

