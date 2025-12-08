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
