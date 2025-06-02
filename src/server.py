from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import tensorflow as tf
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.h5')
TOP_N = 3

class_names = [
    "abimanyu", "duryodana", "arjuna", "anoman", "bagong", "baladewa",
    "buta", "cakil", "durna", "dursasana", "gatotkaca", "karna",
    "kresna", "nakula_sadewa", "petruk", "puntadewa", "semar",
    "sengkuni", "tagog", "patih_sebrang", "gareng", "bima"
]

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model(MODEL_PATH)
input_shape = (224, 224)

@app.route('/', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'status': 'failed', 'error': 'No image provided'}), 400
    
    file = request.files['image']

    if file.content_type not in ['image/jpeg', 'image/png']:
        return jsonify({'status': 'failed', 'error': 'File is not an image'}), 400
    
    file_bytes = file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        return jsonify({'status': 'failed', 'error': 'File is too large'}), 400

    try:
        img = tf.keras.preprocessing.image.load_img(
            io.BytesIO(file_bytes), target_size=input_shape
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        top_indices = predictions.argsort()[-TOP_N:][::-1]
        top_predictions = [
            {'label': class_names[i], 'prediction': f"{predictions[i] * 100:.2f}"} for i in top_indices
        ]

        return jsonify({
            'status': 'success',
            'data': top_predictions
        }), 200

    except Exception as e:
        return jsonify({'status': 'failed', 'error': str(e)}), 500

@app.errorhandler(404)
@app.errorhandler(405)
def handle_invalid_usage(error):
    return jsonify({
        'status': 'error',
        'error': 'Path or method not exist'
    }), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


