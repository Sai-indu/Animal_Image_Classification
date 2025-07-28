from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('animal_classifier_model.h5')

# Set your class labels based on folders used in training
class_names = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
    'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
    'Panda', 'Tiger', 'Zebra'
]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text="Please select an image.")

    filepath = os.path.join('static', 'uploaded_image.jpg')
    file.save(filepath)

    img_array = preprocess_image(filepath)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]

    return render_template('index.html',
                           prediction_text=f'🐾 Predicted Animal: {predicted_label.title()}',
                           uploaded_image='uploaded_image.jpg')

if __name__ == '__main__':
    app.run(debug=True)
