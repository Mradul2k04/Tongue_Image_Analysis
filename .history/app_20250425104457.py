from flask import Flask, render_template, request import os from werkzeug.utils import secure_filename import numpy as np from tensorflow.keras.models import load_model from tensorflow.keras.preprocessing import image

app = Flask(name) model = load_model("model/my_model.h5")

def preprocess(img_path): img = image.load_img(img_path, target_size=(224, 224)) # Adjust based on your model img_array = image.img_to_array(img) img_array = np.expand_dims(img_array, axis=0) return img_array / 255.0

@app.route('/') def home(): return render_template("index.html")

@app.route('/predict', methods=['POST']) def predict(): predictions = {} for view in ['front_image', 'left_image', 'right_image']: uploaded_file = request.files.get(view) if uploaded_file and uploaded_file.filename != '': filename = secure_filename(uploaded_file.filename) filepath = os.path.join("static", filename) uploaded_file.save(filepath) img_input = preprocess(filepath) pred = model.predict(img_input) predictions[view] = str(np.argmax(pred)) # Adjust if using labels