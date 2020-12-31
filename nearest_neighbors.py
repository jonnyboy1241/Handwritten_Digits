# nearest-neighbors classifiers and varients
import numpy as np

# Standard nearest neighbor classifier
#
# training_data - 2D np array that holds the training images
# training_labels - np array that holds the labels for the corresponding training images
# test_data - 2D np array to calculate the classification accuracy
# test_labels - true labels for the test data, used to calculate the classification accuracy
# k - # of neighbors to consider (1-NN, 3-NN, 5-NN, k-NN)
# p value to use for the minkowski distance - p = 1 is the Manhattan Distance, p = 2 is the Euclidean distance. Doesn't have to be an integer
def k_nearest_neighbors(training_data, training_labels, test_data, test_labels, k, minkowski_num):
    classification_labels = np.empty(test_labels.size, dtype=np.int8)

    for i in range(test_labels.size):
        classification_labels[i] = classify(test_data[i], training_data, training_labels, k, minkowski_num)

    return classification_labels
    

# Classifies a singular image
def classify(image_to_classify, training_data, training_labels, k , minkowski_num):
    distances = np.empty(training_data.shape[0])

    for i in range(training_data.shape[0]):
        distances[i] = minkowski_distance(image_to_classify, training_data[i], minkowski_num)

    class_label = get_class_label(distances, k, training_labels)

    return class_label


# Returns a single classification for the given distances vector
# Returns the first value in a tie
def get_class_label(distances, k, training_labels):
    k_nearest_neighbors_indices = np.argpartition(distances, k)[:k]
    k_nearest_neighbors_labels = [training_labels[x] for x in k_nearest_neighbors_indices]
    return max(k_nearest_neighbors_labels, key = k_nearest_neighbors_labels.count)


# Returns the minkowski distance between two images
def minkowski_distance(img_1, img_2, minkowski_num):
    diff_vect = np.abs(img_1 - img_2)
    sum_of_componants =  np.sum(np.power(diff_vect, minkowski_num))
    return (sum_of_componants ** (1 / minkowski_num))
