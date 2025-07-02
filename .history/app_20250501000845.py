from flask import Flask, render_template, request
import os
from io import BytesIO
import base64
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = load_model('model/my_model.h5')
class_names = ['Non-coated Tongue', 'Coated Tongue']

def preprocess_image(image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def decode_base64_image(data_url):
    header, encoded = data_url.split(',', 1)
    return Image.open(BytesIO(base64.b64decode(encoded)))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    results = []
    for position in ['front', 'left', 'right']:
        data_url = request.form.get(f'{position}_image')
        if data_url:
            img = decode_base64_image(data_url)
            img_array = preprocess_image(img)
            prediction = model.predict(img_array)[0]
            index = np.argmax(prediction)
            confidence = round(float(prediction[index]) * 100, 2)
            label = class_names[index]
            health = "You are Healthy." if label == 'Non-coated Tongue' else "You are Not Healthy."
            quote = "A healthy tongue is a mirror to well-being." if label == 'Non-coated Tongue' else "Take care of your health—small signs speak volumes."

            save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{position}.png")
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            img.save(save_path)

            results.append({
                'image_path': save_path,
                'label': label,
                'confidence': confidence,
                'health_status': health,
                'quote': quote
            })
    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
