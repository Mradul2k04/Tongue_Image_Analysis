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

model = load_model('model/my_model.h5')
class_names = ['Non-coated Tongue', 'Coated Tongue']

quotes = [
    "Your tongue tells a tale your heart can't hide.",
    "A healthy body begins with awareness.",
    "Wellness starts with observation.",
    "Prevention is the best medicine.",
    "Caring for your health is an act of self-respect."
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
    images = []
    image_paths = []

    for key in ['front_image', 'left_image', 'right_image']:
        file = request.files.get(key)
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            image_paths.append(filepath)
            images.append(preprocess_image(filepath))

    if not images:
        return redirect('/')

    predictions = [model.predict(img)[0] for img in images]
    avg_prediction = np.mean(predictions, axis=0)
    predicted_index = np.argmax(avg_prediction)
    confidence = round(float(avg_prediction[predicted_index]) * 100, 2)
    label = class_names[predicted_index]

    health_message = "You are healthy! 😊" if label == "Non-coated Tongue" else "You may not be healthy. 🧐"
    quote = random.choice(quotes)

    return render_template(
        'result.html',
        label=label,
        confidence=confidence,
        health_message=health_message,
        quote=quote,
        image_paths=image_paths
    )

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
