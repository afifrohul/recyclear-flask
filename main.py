from flask import Flask, request, jsonify, send_file
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from keras.models import model_from_json
import os
import datetime
import mysql.connector

app = Flask(__name__)

config = {
    'user': 'root',
    'password': '',
    'host': '127.0.0.1',
    'database': 'recyclear'
}

conn = mysql.connector.connect(**config)

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

    cursor = conn.cursor()

    # Query INSERT
    insert_query = "INSERT INTO predictions (image, class_image, confidence, user_id, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s)"
    
    # Data yang ingin di-insert
    data_to_insert = (str(datetime.datetime.now()) + image.filename, predicted_class, confidence , 1, datetime.datetime.now(), datetime.datetime.now())
    
    # Menjalankan query dengan data yang ingin di-insert
    cursor.execute(insert_query, data_to_insert)
    
    # Commit perubahan
    conn.commit()

    cursor.close()
    conn.close()

    return [predicted_class, confidence]

@app.route("/")
def welcome():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})
    
    image = request.files['image']
    file_path = os.path.join('images', str(datetime.datetime.now().strftime("%Y-%m-%d")) + image.filename)
    image.save(file_path)
    result = predict_image(image)
    
    # Modify this part to structure the response according to your requirements
    return jsonify({'class': result[0], 'confidence': str(result[1]) + '%'})