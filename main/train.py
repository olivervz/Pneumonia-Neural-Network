import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
import os

img_size = 250


def load_datasets(path):
    # Load each dataset from their .npy binary files
    x_train = np.load(path + 'x_train.npy')
    y_train = np.load(path + 'y_train.npy')
    x_test = np.load(path + 'x_test.npy')
    y_test = np.load(path + 'y_test.npy')
    return x_train, y_train, x_test, y_test


def normalize_data(x_train, x_test):
    # Since currently each value is between 0-255 int values for color
    # Normalize it to 0-1 float values
    x_train = x_train / 255
    x_test = x_test / 255
    return x_train, x_test


def reshape_data(x_train, y_train, x_test, y_test):
    # Prior to reshaping, the data is an array of images
    # Each image is a 2D 150x150 array of floats
    # After reshaping, the data is an array of images
    # Each image is an array of length 150
    # Each element is another array of length 150
    # Each element of that array is a float array of length 1

    # This allows it to be used with keras' training
    x_train = x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train)

    x_test = x_test.reshape(-1, img_size, img_size, 1)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test


def generate_model():
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3),
              input_shape=(img_size, img_size, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def save_model(model):
    model_json = model.to_json()
    with open("model/model.json", 'w') as json_file:
        json_file.write(model_json)
    model.save_weights("model/model.h5")


def main():
    # Load datasets
    print('Loading datasets...')
    x_train, y_train, x_test, y_test = load_datasets('data/dataset/')
    print('Completed loading datasets')

    # normalize data
    x_train, x_test = normalize_data(x_train, x_test)

    # resize data for deep learning
    x_train, y_train, x_test, y_test = reshape_data(
        x_train, y_train, x_test, y_test)

    # Create model
    model = generate_model()
    # Generate summary of layers
    model.summary()

    # Train model using datasets
    model.fit(x_train, y_train, epochs=1,
              validation_data=(x_test, y_test))

    print("Model Loss: ", model.evaluate(x_test, y_test)[0])
    print("Model Accuracy: ",
          model.evaluate(x_test, y_test)[1]*100, "%")

    # Save model
    save_model(model)


if __name__ == '__main__':
    main()
