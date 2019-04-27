""" Creating a NN web service"""

# Libraries

import base64
import pickle
import numpy as np
import io
from PIL import Image
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask


app = Flask(__name__)
labels_dict = pickle.load(open('labels', 'rb'))


def get_model():
    global model
    model = load_model('cnn_classifier.h5')
    model._make_predict_function()
    print(" * Model Loaded!")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image.reshape(1, 32, 32, 3)
    # image = np.expand_dims(image, axis=0)

    return image


print(" * Loading Keras Model...")
get_model()
global graph
graph = tf.get_default_graph()


@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    dataBytesIO = io.BytesIO(decoded)
    image = Image.open(dataBytesIO)
    processed_image = preprocess_image(image, target_size=(32, 32))

    with graph.as_default():
        prediction = model.predict(processed_image).tolist()

    response = {
        'prediction': {'pred': labels_dict[np.argmax(prediction)]}
    }

    return jsonify(response)


if __name__ == "__main__":
    image = Image.open("C:/Users/mjcastaneda4/Pictures/car3.jpg")
    image = image.resize((32, 32))
    image = img_to_array(image).reshape(1, 32, 32, 3)
    with graph.as_default():
        prediction = model.predict(image)
    print(prediction)
    response = {'prediction': str(labels_dict[np.argmax(prediction)])}
    print(response)
