from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from PIL import Image
import io
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing import image as keras_image

from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in the app

# Load the pre-trained TensorFlow model
model = load_model('./ex5/Fresh_Rotten_fruis.h5')

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/api', methods=['POST'])
def process():
    image_data = request.form['image']
    image_bytes = base64.b64decode(image_data)
    original_image = Image.open(io.BytesIO(image_bytes))

    resized_image = original_image.resize((64, 64))
    reshaped_image = np.reshape(resized_image, (1, 64, 64, 3))
    reshaped_array = np.array(reshaped_image)

    processed_result = predict_with_model(reshaped_array)

    return jsonify({'result': processed_result})

def predict_with_model(input_data):
    classes = ['Fresh Apple','Fresh Banana','Fresh Orange','Rotten Apple','Rotten Banana','Rotten Orange']
    predictions = model.predict(input_data)

    for i in range(6):
      if predictions.tolist()[0][i] == 1:
        break
    prediction = classes[i]
    print(prediction)
    print(predictions.tolist())

    return prediction

if __name__ == '__main__':
    app.run(debug=True)
