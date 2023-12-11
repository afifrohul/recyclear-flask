from flask import Flask, request, jsonify, send_file
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from keras.models import model_from_json

app = Flask(__name__)

# Load the model architecture from JSON file
with open('model_recyclear.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Create model from JSON
loaded_model = model_from_json(loaded_model_json)

# Load weights into the model
loaded_model.load_weights('model_05-0.97.h5')

def predict_image(image):
    img = Image.open(image)
    img = img.resize((300, 300))
    img_array = np.asarray(img)
    img_array = img_array / 255.0

    # Make prediction
    predictions = loaded_model.predict(np.expand_dims(img_array, axis=0))

    predicted_class_index = np.argmax(predictions)

    # Map index to class name
    class_mapping = ['HDPE', 'LDPE', 'PET', 'PP', 'PS', 'PVS']  # Replace with your actual class names
    predicted_class = class_mapping[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    return [predicted_class, confidence]

@app.route("/")
def welcome():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})
    
    image = request.files['image']
    result = predict_image(image)
    
    # Modify this part to structure the response according to your requirements
    return jsonify({'class': result[0], 'confidence': str(result[1]) + '%'})