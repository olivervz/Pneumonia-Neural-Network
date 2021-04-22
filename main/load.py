import os
import sys
import cv2
import numpy as np

# Two types of data, NORMAL and PNEUMONIA
xray_types = ["NORMAL", "PNEUMONIA"]

# Default size for image
img_size = 150


def main():
    # load the sets of training and testing data
    print('Loading training dataset...')
    training = load_dataset('data/train/')
    print('Finished loading training dataset')
    print('Loading Testing dataset...')
    testing = load_dataset('data/test/')
    print('Finished loading testing dataset')

    # generate the x and y sets of training and testing data
    (x_train, y_train) = construct_dataset(training)
    (x_test, y_test) = construct_dataset(testing)

    # Save data to numpy binary
    print('Saving x_train data...')
    np.save('data/dataset/x_train.npy', x_train)
    print('Saving y_train data...')
    np.save('data/dataset/y_train.npy', y_train)
    print('Saving x_test data...')
    np.save('data/dataset/x_test.npy', x_test)
    print('Saving y_test data...')
    np.save('data/dataset/y_test.npy', y_test)
    print("Complete")

# A user submitted a function for the same kaggle dataset which converts the
# images into numpy arrays that can be used by Keras.
# https://www.kaggle.com/madz2000/pneumonia-detection-using-cnn-92-6-accuracy
def load_dataset(dir):
    
    dataset = []
    for xray_type in xray_types:
        # Get path to the xray image
        path_to_images = os.path.join(dir, xray_type)
        # 0 for NORMAL, 1 for PNEUMONIA
        class_num = xray_types.index(xray_type)
        # List all images in each directory
        for xray_image in os.listdir(path_to_images):
            # Get path to each image
            path_to_image = os.path.join(path_to_images, xray_image)
            # Use the cv2 library to convert the GRAYSCALE image into an array
            img_arr = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
            # Resize the array
            resized_arr = cv2.resize(img_arr, (img_size, img_size))
            # Append each image array with it's type
            dataset.append([resized_arr, class_num])
    # Return the array as a numpy array
    return np.array(dataset, dtype=object)


def construct_dataset(dataset):
    x = []
    y = []

    # Construct training and testing sets
    for image_data, image_type in dataset:
        x.append(image_data)
        y.append(image_type)

    return (x, y)


if __name__ == '__main__':
    main()
