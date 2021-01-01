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


# Get a subset of the training set
# Get n images from each class (0 - 9)
def reduce_training_set(n, training_images, training_labels):
    assert 0 <= n and n <= 5000

    indices_of_selected_images = []
    selected_images = 0
    label_distribution = np.zeros((10), dtype=np.int)

    # While not the most efficient method for doing this, resulting datasets will be very random
    while selected_images < (n * 10):
        location = random.randint(0, training_labels.size - 1)

        label = training_labels[location]

        if label_distribution[label] >= n:
            continue

        indices_of_selected_images.append(label)
        selected_images += 1


    images = np.empty((n * 10, 28 * 28), dtype=np.uint8)
    labels = np.empty((n * 10,), dtype=np.uint8)

    for i in range(len(indices_of_selected_images)):
        images[i] = training_images[indices_of_selected_images[i]]
        labels[i] = training_labels[indices_of_selected_images[i]]
    
    return images, labels


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
