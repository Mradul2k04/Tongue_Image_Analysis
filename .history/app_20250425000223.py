from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Load the trained CNN model
model = load_model('model/my_model.h5')

# Define class labels based on your dataset
class_names = ['Non-coated Tongue', 'Coated Tongue']

# Preprocess image before prediction
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
    if 'image' not in request.files:
        return redirect('/')
    file = request.files['image']
    if file.filename == '':
        return redirect('/')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    img_array = preprocess_image(filepath)
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    confidence = round(float(prediction[predicted_index]) * 100, 2)
    label = class_names[predicted_index]

    return render_template('result.html', label=label, confidence=confidence, image_path=filepath)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
