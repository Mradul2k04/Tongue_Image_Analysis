from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Load the trained CNN model
model = load_model('model/my_model.h5')

# Define class labels
class_names = ['Non-coated Tongue', 'Coated Tongue']

# Some positive quotes
quotes = {
    'Non-coated Tongue': [
        "A healthy outside starts from a healthy inside. 🌟",
        "Your health shines through you. Keep glowing! ✨",
        "Wellness is the natural state of your body. 🌱"
    ],
    'Coated Tongue': [
        "Healing is a matter of time, but it is sometimes also a matter of opportunity. 🌸",
        "Every day is a new chance to nurture your health. 🌿",
        "Listen to your body. It's the only place you have to live. ❤️"
    ]
}

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
    if 'front_image' not in request.files:
        return redirect('/')
    uploaded_images = []
    predictions = []

    for key in ['front_image', 'left_image', 'right_image']:
        file = request.files.get(key)
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            uploaded_images.append('uploads/' + filename)  # Relative path for HTML

            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array)[0]
            predicted_index = np.argmax(prediction)
            predictions.append(class_names[predicted_index])

    if not predictions:
        return redirect('/')

    # Majority vote
    final_prediction = max(set(predictions), key=predictions.count)
    selected_quote = random.choice(quotes[final_prediction])

    return render_template('result.html', image_paths=uploaded_images, quote=selected_quote)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
