from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

model = load_model('model/my_model.h5')
class_names = ['Non-coated Tongue', 'Coated Tongue']

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
    # Check if an image was uploaded
    if 'tongue_image' not in request.files:
        return redirect(request.url)
    file = request.files['tongue_image']

    if file and file.filename:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        confidence = round(float(prediction[predicted_index]) * 100, 2)
        label = class_names[predicted_index]
        health_status = "You are Healthy." if label == 'Non-coated Tongue' else "You are Not Healthy."
        quote = "A healthy tongue is the mirror to your inner well-being." if label == 'Non-coated Tongue' else "Take care of your health—small signs speak volumes."

        # Send the results to the result page
        return render_template('result.html', image_path=filepath, confidence=confidence, label=label, health_status=health_status, quote=quote)
    return redirect(request.url)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
