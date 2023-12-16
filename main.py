from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.models import model_from_json
import os
import datetime
import mysql.connector
from google.cloud import storage
import random
import string

app = Flask(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key/recyclear-dev-0003-8859ed001a13.json"

config = {
    'user': 'recyclear-admin',
    'password': 'recyclear-admin',
    'host': '34.101.230.24',
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

def generate_random_string():
    # Pilih karakter alfabet dan angka secara acak sepanjang 4 karakter
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    return str(random_string)

def predict_image(image, user_id, image_name):
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
    data_to_insert = (image_name, predicted_class, confidence , user_id, datetime.datetime.now(), datetime.datetime.now())
    
    # Menjalankan query dengan data yang ingin di-insert
    cursor.execute(insert_query, data_to_insert)
    
    # Commit perubahan
    conn.commit()

    return [predicted_class, confidence]

def upload_to_gcs(file, bucket_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(file, content_type='image/jpeg')

@app.route("/")
def welcome():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})
    
    image = request.files['image']
    image_name = image.filename
    user_id = request.form['user_id']
    str_random = generate_random_string()

    bucket_name = 'recyclear-images-classification'
    destination_blob_name = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + str_random + image_name.replace(" ", "")

    upload_to_gcs(image, bucket_name, destination_blob_name)

    result = predict_image(image, user_id, destination_blob_name)

    return jsonify({'class': result[0], 'confidence': str(result[1]) + '%'})

@app.route("/api/predict", methods=["POST"])
def predict_api():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image found'})
    
        image = request.files['image']
        image_name = image.filename
        user_id = request.form['user_id']
        str_random = generate_random_string()
        bucket_name = 'recyclear-images-classification'
        destination_blob_name = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + str_random + image_name.replace(" ", "")

        upload_to_gcs(image, bucket_name, destination_blob_name)
        result = predict_image(image, user_id, destination_blob_name)

        return jsonify({'class': result[0], 'confidence': str(result[1]) + '%'})