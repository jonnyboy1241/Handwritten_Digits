# This file has all of the functions that read the data from its original format

import numpy as np
import matplotlib.pyplot as plt

from view_image import *

# This funciton returns the 60000 images as 1D numpy arrays (784 bytes)
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

        # The training set image file consists of 600000 images
        # Each image is 28 x 28 pixels. Each pixel takes a greyscale
        # value in the range 0-255 inclusive.

        # Read the data
        image_data = file.read(60000 * 28 * 28)

        images = []

        for i in range(60000):
            image_data_arr = np.frombuffer(image_data, dtype=np.uint8, count=(28 * 28), offset=(i * 28 * 28))
            images.append(image_data_arr)

        return images
