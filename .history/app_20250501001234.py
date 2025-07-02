from flask import Flask, render_template, request
import os
import base64
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO

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

    for view in ['front', 'left', 'right']:
        image_b64 = request.form.get(f'{view}_image_base64')
        if image_b64:
            header, encoded = image_b64.split(',', 1)
            img_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(img_data))
            img_array = preprocess_image(image)

            prediction = model.predict(img_array)[0]
            label = class_names[np.argmax(prediction)]
            confidence = round(float(np.max(prediction)) * 100, 2)
            health_status = "You are Healthy." if label == 'Non-coated Tongue' else "You are Not Healthy."
            quote = "A healthy tongue is the mirror to your inner well-being." if label == 'Non-coated Tongue' else "Take care of your health—small signs speak volumes."

            filename = f"{view}_{len(results)}.png"
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
