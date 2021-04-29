import json
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential, model_from_json, load_model
from train import load_datasets, normalize_data, reshape_data

img_size = 250

def generate_prediction(x, y, model, i):
    plt.style.use("dark_background")
    fig = plt.figure()
    data = x.reshape(img_size, img_size)
    plt.imshow(data)
    pred = model.predict(x.reshape(1, img_size, img_size, 1))[0][0]
    
    # Determine model's prediction
    if pred > 0.5:
        prediction = "PNEUMONIA"
    else:
        prediction = "NORMAL"

    # Determine actual
    if y == 1:
        actual = "PNEUMONIA"
    else:
        actual = "NORMAL"

    confidence = (abs(0.5 - pred) / 0.5) * 100
    print("X-ray #" + str(i) + " - Prediction: " + str(pred))
    title = 'Prediction: ' + prediction + ' Actual: ' + actual + ' Confidence: ' + str(round(confidence, 5)) + "%" 
    plt.title(title)
    plt.show()

def test_validation(validation_x, validation_y, model):
    for i in range(len(validation_x)):
        generate_prediction(validation_x[i], validation_y[i], model, i)

def fix_serialize(json_model_str):
    json_model = json.loads(json_model_str)

    for layer in json_model["config"]["layers"]:
        if "activation" in layer["config"].keys():
            if layer["config"]["activation"] == "softmax_v2":
                layer["config"]["activation"] = "softmax"

    return json.dumps(json_model)


def load_model():
    with open("model/model.json", 'r') as json_file:
        json_model = json_file.read()

        json_model = fix_serialize(json_model)

        model = model_from_json(json_model)

        model.load_weights("model/model.h5")
        print("Loaded Model")
        return model


def main():
    # Load model from save file
    model = load_model()

    # Load validaiton data
    path = 'data/dataset/'
    x_validation = np.load(path + 'x_validation.npy')
    y_validation = np.load(path + 'y_validation.npy')

    # Normalize validation data
    x_validation = x_validation / 255

    # Reshape validation data
    x_validation = x_validation.reshape(-1, img_size, img_size, 1)
    y_validation = np.array(y_validation)

    # Test using known inputs
    test_validation(x_validation, y_validation, model)


if __name__ == '__main__':
    main()
