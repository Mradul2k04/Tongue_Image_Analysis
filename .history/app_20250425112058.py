from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dummy predict function (you will replace this with your model)
def predict_image(image_path):
    import random
    confidence = random.uniform(40, 100)
    return confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        confidence = predict_image(filepath)
        confidence = round(confidence, 2)
        
        return render_template('result.html', confidence=confidence, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
