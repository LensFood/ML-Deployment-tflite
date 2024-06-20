import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Define the path to the TFLite model
model_path = 'MobileNetV3-L.tflite'

# Check if the model file exists
if not os.path.exists(model_path):
    raise ValueError(f"Model file not found at {model_path}")

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the image
def preprocess_image(img_path, img_size):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to run inference on an image
def run_inference(img_path):
    img_size = 224  # Define the image size based on your model
    input_data = preprocess_image(img_path, img_size)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Define the class names
class_names = [
    "ayam bakar", "ayam goreng", "bakso", "bakwan", "batagor", "bihun", "capcay", "gado-gado",
    "ikan goreng", "kerupuk", "martabak telur", "mie", "nasi goreng", "nasi putih", "nugget",
    "opor ayam", "pempek", "rendang", "roti", "sate", "sosis", "soto", "steak", "tahu", "telur",
    "tempe", "terong balado", "tumis kangkung", "udang"
]

# Define the path to the test image
img_path = ('test2.jpeg')

# Check if the image file exists
if not os.path.exists(img_path):
    raise ValueError(f"Image file not found at {img_path}")

# Run inference and get the predictions
predictions = run_inference(img_path)

# Get the index of the highest prediction
predicted_index = np.argmax(predictions)

# Get the name of the predicted class
predicted_class = class_names[predicted_index]

# Print the predicted class name
print("Predicted class:", predicted_class)
