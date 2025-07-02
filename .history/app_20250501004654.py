@app.route('/predict', methods=['POST'])
def predict():
    image_results = []
    uploaded_image = None

    # Get the uploaded image
    file = request.files.get('tongue_image')
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

        # Store the result
        image_results.append({
            'image_path': f"uploads/{filename}",  # Make sure this path is correct
            'confidence': confidence,
            'label': label,
            'health_status': health_status,
            'quote': quote
        })

    return render_template('result.html', results=image_results)
