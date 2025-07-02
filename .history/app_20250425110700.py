from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Load trained model
model = load_model('model/my_model.h5')

# Define class labels
class_names = ['Non-coated Tongue', 'Coated Tongue']

# Health quotes
quotes = [
    "Wellness is the natural state of your body. 🌱",
    "Take care of your body. It's the only place you have to live. 💪",
    "A healthy outside starts from the inside. ✨",
    "Health is wealth — treat it like treasure. 💸",
    "Healing is a matter of time, but also a matter of choice. 🕳️"
]

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_keys = ['front_image', 'left_image', 'right_image']
    predictions = []
    image_paths = []

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    for key in image_keys:
        file = request.files.get(key)
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_paths.append(filepath)

            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array)[0]
            predictions.append(prediction)

    if not predictions:
        return redirect('/')

    avg_prediction = np.mean(predictions, axis=0)
    predicted_index = np.argmax(avg_prediction)
    confidence = round(float(avg_prediction[predicted_index]) * 100, 2)
    label = class_names[predicted_index]

    if label == 'Non-coated Tongue':
        health_msg = "You're Healthy ✅"
    else:
        health_msg = "You need to see a Doctor 🤯"

    quote = random.choice(quotes)
    display_image = image_paths[0] if image_paths else None

    return render_template('result.html', label=label, confidence=confidence,
                           health_msg=health_msg, quote=quote, image_path=display_image)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)