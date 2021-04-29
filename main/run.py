import json
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential, model_from_json, load_model
from train import load_datasets, normalize_data, reshape_data
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

img_size = 250

def display_predictions(predictions, actuals, image_data):
    for i in range(len(predictions)):
        plt.style.use("dark_background")
        fig = plt.figure()
        data = image_data[i].reshape(img_size, img_size)
        plt.imshow(data, cmap='gray')
        
        # Determine model's prediction
        if predictions[i] == 1:
            prediction = "PNEUMONIA"
        else:
            prediction = "NORMAL"

        # Determine actual
        if actuals[i] == 1:
            actual = "PNEUMONIA"
        else:
            actual = "NORMAL"

        title = 'Prediction: ' + prediction + ' Actual: ' + actual
        plt.title(title)
        plt.show()

def generate_confusion_matrix(predictions, y_validation):
    cm  = confusion_matrix(y_validation, predictions)
    plt.figure()
    plt.style.use("dark_background")
    plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True, cmap=plt.cm.Greens)
    plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    plt.show()


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
    predictions = model.predict(x_validation)
    predictions = np.argmax(predictions, axis=-1)


    # Display 10 predictions by the model
    display_predictions(predictions, y_validation, x_validation)

    # Generate a confusion matrix to show false positives and negatives
    generate_confusion_matrix(predictions, y_validation)

    # Repeat with test
    path = 'data/dataset/'
    x_test= np.load(path + 'x_test.npy')
    y_test= np.load(path + 'y_test.npy')

    print(len(y_test))
    print(len(x_test))

    # Normalize test data
    x_test = x_test / 255

    # Reshape validation data
    x_test = x_test.reshape(-1, img_size, img_size, 1)
    y_test = np.array(y_test)


    # Test using known inputs
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=-1)

    # Generate a confusion matrix to show false positives and negatives
    generate_confusion_matrix(predictions, y_test)

if __name__ == '__main__':
    main()
