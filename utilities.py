# This file includes various utilities that are helpful for visualization/accuracy computation

import matplotlib.pyplot as plt
import numpy as np
import time
import random

# This funciton displays an MNIST handwritten digit as a 28x28 pixel image
def view_image(image):
    image = image.reshape(28, 28)
    plt.gray()
    plt.imshow(image)
    plt.show()


# This function displays an MNIST handwritten digit as a reduced 14x14 pixel image
def view_reduced_image(image):
    image = image.reshape(14, 14)
    plt.gray()
    plt.imshow(image)
    plt.show()


# Calculate the accuracy of the classification
def compute_accuracy(classifications, labels):
    assert classifications.size == labels.size

    correct = np.int(0)
    for i in range(labels.size):
        if classifications[i] == labels[i]:
            correct += 1

    classification_accuracy = correct / labels.size
    print('Classification Accuracy = %6.4f' % (classification_accuracy))

    return classification_accuracy


def compute_time(begin_time):
    time_passed = time.time() - begin_time
    print('Time = %6.4f s' % (time_passed))
    return time_passed
