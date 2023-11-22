from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import base64
from PIL import Image
import io
from tensorflow.keras.models import load_model
from keras.preprocessing import image as keras_image

app = Flask(__name__)
model = load_model('models/Fresh_Rotten_fruis.h5')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/api', methods=['POST'])
def process():
    if 'img' not in request.files:
        return 'No file'
    file = request.files['img'].read()
    npimg = np.frombuffer(file, np.uint8)
    original_image = Image.open(io.BytesIO(npimg))
    
    # image_data = request.form['image']
    # image_bytes = base64.b64decode(image_data)
    # original_image = Image.open(io.BytesIO(image_bytes))

    resized_image = original_image.resize((64, 64))
    reshaped_image = np.reshape(resized_image, (1, 64, 64, 3))
    reshaped_array = np.array(reshaped_image)

    buffered = io.BytesIO()
    original_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    processed_result = predict_with_model(reshaped_array)

    return render_template("predict.html", prediction=processed_result, uploaded_image=img_str)

def predict_with_model(input_data):
    classes = ['Fresh Apple','Fresh Banana','Fresh Orange','Rotten Apple','Rotten Banana','Rotten Orange']
    predictions = model.predict(input_data)

    predicted_class_index = np.argmax(predictions)
    prediction = classes[predicted_class_index]
    print(prediction)
    print(predictions.tolist())

    return prediction

if __name__ == '__main__':
   app.run(host='0.0.0.0', debug=True)
