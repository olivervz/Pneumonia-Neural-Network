import json
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential, model_from_json, load_model
from train import load_datasets, normalize_data, reshape_data


def print_example(test, model, r):
    plt.style.use("dark_background")
    fig = plt.figure()
    data = test[r].reshape(150, 150)
    plt.imshow(data)
    pred = model.predict(test[r].reshape(1, 150, 150, 1))[0][0]
    prediction = "NORMAL"
    if pred > 0.5:
        prediction = "PNEUMONIA"
    title = 'Prediction: ' + prediction + " " + str(pred)
    plt.title(title)
    plt.show()


def print_n_examples(test, model, n):
    for i in range(n):
        random_i = int(random.random() * len(test))
        print_example(test, model, random_i)


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
    # Reshape data
    x_train, y_train, x_test, y_test = load_datasets('data/dataset/')
    x_train, x_test = normalize_data(x_train, x_test)
    x_train, y_train, x_test, y_test = reshape_data(
        x_train, y_train, x_test, y_test)
    # Print 10 examples
    print_n_examples(x_test, model, 10)


if __name__ == '__main__':
    main()
