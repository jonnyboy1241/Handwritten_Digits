# This file has all of the functions that read the data from its original format

import numpy as np
import matplotlib.pyplot as plt

# This funciton returns the 60000 images as a 2D numpy array (60000 X 784 bytes)
# Each value can be viewed using matplotlib.pyplot - must be reshaped as (28, 28)
def get_training_data():
    with open('data/train-images-idx3-ubyte', 'rb') as file:
        magic_num = int.from_bytes(file.read(4), 'big')
        
        # Ensure the magic number matches
        assert magic_num == 0x803

        num_images = int.from_bytes(file.read(4), 'big')
        num_rows = int.from_bytes(file.read(4), 'big')
        num_cols = int.from_bytes(file.read(4), 'big')

        assert num_images == 60000
        assert num_rows == 28
        assert num_cols == 28

        # The training set image file consists of 60000 images
        # Each image is 28 x 28 pixels. Each pixel takes a greyscale
        # value in the range 0-255 inclusive.

        # Read the data
        image_data = file.read(60000 * 28 * 28)

        images = np.empty((60000, 28 * 28), dtype=np.uint8)

        for i in range(60000):
            images[i] = np.frombuffer(image_data, dtype=np.uint8, count=(28 * 28), offset=(i * 28 * 28))

        return images


# Returns the 60000 labels for the training data
def get_training_labels():
    with open('data/train-labels.idx1-ubyte', 'rb') as file:
        magic_num = int.from_bytes(file.read(4), 'big')

        # Ensure the magic number matches
        assert magic_num == 0x801

        num_labels = int.from_bytes(file.read(4), 'big')

        assert num_labels == 60000

        # Ths file includes 60000 unsigned bytes (np.uint8)
        # as labels for the training data
        raw_label_data = file.read(60000)
        label_data = np.frombuffer(raw_label_data, dtype=np.uint8, count=60000, offset=0)

        return label_data


# Returns the 10000 images in the test set in the same format
def get_test_data():
    with open('data/t10k-images-idx3-ubyte', 'rb') as file:
        magic_num = int.from_bytes(file.read(4), 'big')
        
        # Ensure the magic number matches
        assert magic_num == 0x803

        num_images = int.from_bytes(file.read(4), 'big')
        num_rows = int.from_bytes(file.read(4), 'big')
        num_cols = int.from_bytes(file.read(4), 'big')

        assert num_images == 10000
        assert num_rows == 28
        assert num_cols == 28

        # Read the data
        image_data = file.read(10000 * 28 * 28)

        images = np.empty((10000, 28 * 28), dtype=np.uint8)

        for i in range(10000):
            images[i] = np.frombuffer(image_data, dtype=np.uint8, count=(28 * 28), offset=(i * 28 * 28))

        return images


# Returns the 10000 labels for the test data
def get_test_labels():
    with open('data/t10k-labels-idx1-ubyte', 'rb') as file:
        magic_num = int.from_bytes(file.read(4), 'big')

        # Ensure the magic number matches
        assert magic_num == 0x801

        num_labels = int.from_bytes(file.read(4), 'big')

        assert num_labels == 10000

        # Ths file includes 10000 unsigned bytes (np.uint8)
        # as labels for the test data
        raw_label_data = file.read(10000)
        label_data = np.frombuffer(raw_label_data, dtype=np.uint8, count=10000, offset=0)

        return label_data


# Transforms the data into a 14x14 pixel image
# Not good for classification
def get_reduced_training_data():
    original_training_data = get_training_data()

    transformed_training_data = np.empty((60000, 14 * 14), dtype=np.uint8)

    for i in range(60000):
        image = original_training_data[i].reshape(28, 28)

        new_image = np.empty((196,), dtype=np.int8)

        curr_pixel_num = 0

        for row in range(0, 28, 2):
            for col in range(0, 28, 2):
                # TODO - Use a better algorithm to reduce the image
                new_image[curr_pixel_num] = np.uint8((np.uint16(image[row][col]) + np.uint16(image[row + 1][col]) + np.uint16(image[row][col + 1]) + np.uint16(image[row + 1][col + 1]))) // 4
                curr_pixel_num += 1
        
        transformed_training_data[i] = new_image
    
    return transformed_training_data


def get_reduced_test_data():
    original_test_data = get_test_data()

    transformed_test_data = np.empty((10000, 14 * 14), dtype=np.uint8)

    for i in range(10000):
        image = original_test_data[i].reshape(28, 28)

        new_image = np.empty((196,), dtype=np.int8)

        curr_pixel_num = 0

        for row in range(0, 28, 2):
            for col in range(0, 28, 2):
                # TODO - Use a better algorithm to reduce the image
                new_image[curr_pixel_num] = np.uint8((np.uint16(image[row][col]) + np.uint16(image[row + 1][col]) + np.uint16(image[row][col + 1]) + np.uint16(image[row + 1][col + 1]))) // 4
                curr_pixel_num += 1

        transformed_test_data[i] = new_image

    return transformed_test_data
