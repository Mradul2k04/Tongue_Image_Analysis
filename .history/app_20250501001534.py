import base64
import os
from io import BytesIO

import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = load_model('model/my_model.h5')
class_names = ['Non-coated Tongue', 'Coated Tongue']

def preprocess_image(image):
    image = image.convert("RGB").resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    results = []
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Handle front image - camera or file upload
    front_image_b64 = request.form.get('front_image_base64')
    if front_image_b64:
        # Handle base64 image
        header, encoded = front_image_b64.split(',', 1)
        img_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(img_data))
        img_array = preprocess_image(image)
    else:
        # Handle file upload
        front_image_file = request.files.get('front_image')
        if front_image_file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], front_image_file.filename)
            front_image_file.save(filename)
            image = Image.open(filename)
            img_array = preprocess_image(image)

    # Prediction
    prediction = model.predict(img_array)[0]
    label = class_names[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)) * 100, 2)
    health_status = "You are Healthy." if label == 'Non-coated Tongue' else "You are Not Healthy."
    quote = "A healthy tongue is the mirror to your inner well-being." if label == 'Non-coated Tongue' else "Take care of your health—small signs speak volumes."

    # Save the image file path
    filename = f"front_image_{len(results)}.png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    results.append({
        'image_path': filepath,
        'confidence': confidence,
        'label': label,
        'health_status': health_status,
        'quote': quote
    })

    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
