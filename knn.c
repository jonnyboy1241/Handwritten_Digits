// An implementation of the standard kNN algorithm using C for speed
//
// Compile with gcc using the command below
// gcc knn.c -o knn -lm
//
// Run with the following command
// ./knn <k_value> <p_value>
//      <k_value> is the number of neighbors to classify with (int)
//      <p_value> minkowski distance value (double) (NOTE, must be >= 1.0), if not,
//                the Minkowski Distance violates the triangle inequality principle
//                it is not a metric

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// Specialized data types for our data
typedef unsigned char PIXEL;
typedef unsigned char LABEL;

#define TRAINING_SIZE 60000
#define TEST_SIZE 10000
#define PIXELS_PER_IMAGE 784    // 28 X 28 pixel images

typedef struct IMAGE {
    PIXEL pixels[PIXELS_PER_IMAGE];
} IMAGE;

IMAGE *training_images = NULL;
LABEL *training_labels = NULL;

IMAGE *test_images = NULL;
LABEL *test_labels = NULL;


// Custom data type for the distance array. It's necessary to store
// the original index of the entry to reference the k nearest images
// in the training set later.
typedef struct DISTANCE_ENTRY
{
    int index;
    double distance;
} DISTANCE_ENTRY;


// Forward Function Declarations
void load_data();
void k_nearest_neighbors(unsigned int k, double minkowski_num);
LABEL classify(IMAGE image_to_classify, unsigned int k, double minkowski_num);
LABEL find_most_common_label(LABEL *labels, int num_labels);
void load_training_images();
void load_training_labels();
void load_test_images();
void load_test_labels();
void free_data();
double minkowski_distance(IMAGE img1, IMAGE img2, double minkowski_num);
int buffer_to_int(unsigned char buffer[4]);
int compare_distance(const void *p1, const void *p2);
int compare_label(const void *p1, const void *p2);
double compute_time(clock_t begin);
double classification_accuracy(LABEL *classificaitons);


int main(int argc, char** argv)
{
    if(argc != 3)
    {
        fprintf(stderr, "%s", "INCORRECT NUMBER OF COMMAND LINE ARGUEMENTS\n");
        return -1;
    }

    unsigned int k = (unsigned int) atoi(argv[1]);
    double minkowski_num = atof(argv[2]);

    load_data();
    k_nearest_neighbors(k, minkowski_num);
    free_data();

    return 0;
}


// Load all of the global data pointers with data from the input files
void load_data()
{
    load_training_images();
    load_training_labels();
    load_test_images();
    load_test_labels();

    assert(training_images != NULL);
    assert(training_labels != NULL);
    assert(test_images != NULL);
    assert(test_labels != NULL);
}


// main funciton to run the kNN algorithm
void k_nearest_neighbors(unsigned int k, double minkowski_num)
{
    // Assume that we're classifying All test images for now
    LABEL *classifications = (LABEL *)malloc(TEST_SIZE * sizeof(LABEL));

    clock_t time_began = clock();

    for(int i = 0; i < TEST_SIZE; i++)
    {
        classifications[i] = classify(test_images[i], k, minkowski_num);
    }

    compute_time(time_began);
    compute_accuracy(classifications);

    free(classifications);
    classifications = NULL;
}


/////////////////////////////////
/////   Helper Functions    /////
/////////////////////////////////


// This function classifies one image
LABEL classify(IMAGE image_to_classify, unsigned int k, double minkowski_num)
{
    DISTANCE_ENTRY *distances = (DISTANCE_ENTRY *)malloc(TEST_SIZE * sizeof(DISTANCE_ENTRY));

    for(int i = 0; i < TEST_SIZE; i++)
    {
        distances[i].index = i;
        distances[i].distance = minkowski_distance(image_to_classify, training_images[i], minkowski_num);
    }

    qsort((void *)distances, TEST_SIZE, sizeof(DISTANCE_ENTRY), compare_distance);

    LABEL *nearest_neighbors_labels = (LABEL *)malloc(k * sizeof(LABEL));

    for(int i = 0; i < k; i++)
    {
        nearest_neighbors_labels[i] = training_labels[distances[i].index];
    }

    qsort((void *)nearest_neighbors_labels, k, sizeof(DISTANCE_ENTRY), compare_label);

    LABEL most_common_label = find_most_common_label(nearest_neighbors_labels, k);

    free(distances);
    free(nearest_neighbors_labels);

    distances = NULL;
    nearest_neighbors_labels = NULL;

    return most_common_label;
}


LABEL find_most_common_label(LABEL *labels, int num_labels)
{
    int max_count = 1;
    int curr_count = 1;
    LABEL most_common_label = labels[0];

    for(int i = 1; i < num_labels; i++)
    {
        if(labels[i - 1] == labels[i])
        {
            curr_count++;
        }

        else
        {
            if(curr_count > max_count)
            {
                max_count = curr_count;
                most_common_label = labels[i - 1];
            }

            curr_count = 1;
        }
    }

    // Check the last value in the array
    if(curr_count > max_count)
    {
        most_common_label = labels[num_labels - 1];
    }

    return most_common_label;
}


// File format described at http://yann.lecun.com/exdb/mnist/
void load_training_images()
{
    FILE *fp = fopen("data/train-images-idx3-ubyte", "rb");

    unsigned char buffer[4];

    fread(buffer, sizeof(buffer), 1, fp);
    int magic_number = buffer_to_int(buffer);

    fread(buffer, sizeof(buffer), 1, fp);
    int num_images = buffer_to_int(buffer);

    fread(buffer, sizeof(buffer), 1, fp);
    int num_rows = buffer_to_int(buffer);

    fread(buffer, sizeof(buffer), 1, fp);
    int num_cols = buffer_to_int(buffer);

    // Validate values in the data file
    assert(magic_number == 2051);
    assert(num_images == TRAINING_SIZE);
    assert(num_rows == 28);
    assert(num_cols == 28);

    training_images = (IMAGE *)malloc(TRAINING_SIZE * sizeof(IMAGE));
    fread(training_images, sizeof(IMAGE), TRAINING_SIZE, fp);

    fclose(fp);
}


// File format described at http://yann.lecun.com/exdb/mnist/
void load_training_labels()
{
    FILE *fp = fopen("data/train-labels.idx1-ubyte", "rb");

    unsigned char buffer[4];

    fread(buffer, sizeof(buffer), 1, fp);
    int magic_number = buffer_to_int(buffer);

    fread(buffer, sizeof(buffer), 1, fp);
    int num_labels = buffer_to_int(buffer);

    // Validate values in the data file
    assert(magic_number == 2049);
    assert(num_labels == TRAINING_SIZE);

    training_labels = (LABEL *)malloc(TRAINING_SIZE * sizeof(LABEL));
    fread(training_labels, sizeof(LABEL), TRAINING_SIZE, fp);

    fclose(fp);
}


// File format described at http://yann.lecun.com/exdb/mnist/
void load_test_images()
{
    FILE *fp = fopen("data/t10k-images-idx3-ubyte", "rb");

    unsigned char buffer[4];

    fread(buffer, sizeof(buffer), 1, fp);
    int magic_number = buffer_to_int(buffer);

    fread(buffer, sizeof(buffer), 1, fp);
    int num_images = buffer_to_int(buffer);

    fread(buffer, sizeof(buffer), 1, fp);
    int num_rows = buffer_to_int(buffer);

    fread(buffer, sizeof(buffer), 1, fp);
    int num_cols = buffer_to_int(buffer);

    // Validate values in the data file
    assert(magic_number == 2051);
    assert(num_images == TEST_SIZE);
    assert(num_rows == 28);
    assert(num_cols == 28);

    test_images = (IMAGE *)malloc(TEST_SIZE * sizeof(IMAGE));
    fread(test_images, sizeof(IMAGE), TEST_SIZE, fp);

    fclose(fp);
}


// File format described at http://yann.lecun.com/exdb/mnist/
void load_test_labels()
{
    FILE *fp = fopen("data/t10k-labels-idx1-ubyte", "rb");

    unsigned char buffer[4];

    fread(buffer, sizeof(buffer), 1, fp);
    int magic_number = buffer_to_int(buffer);

    fread(buffer, sizeof(buffer), 1, fp);
    int num_labels = buffer_to_int(buffer);

    // Validate values in the data file
    assert(magic_number == 2049);
    assert(num_labels == TEST_SIZE);

    test_labels = (LABEL *)malloc(TEST_SIZE * sizeof(LABEL));
    fread(test_labels, sizeof(LABEL), TEST_SIZE, fp);

    fclose(fp);
}


void free_data()
{
    free(training_images);
    free(training_labels);
    free(test_images);
    free(test_labels);

    training_images = NULL;
    training_labels = NULL;
    test_images = NULL;
    test_labels = NULL;
}


// Calculate the Minkowski distance between two images
// In a lot of literature, the minkowski_num is represented as "p"
// p = 1 is the Manhattan Distance, p = 2 is the Euclidean Distance
double minkowski_distance(IMAGE img1, IMAGE img2, double minkowski_num)
{
    assert(minkowski_num > 0);

    double distance = 0;

    for(int i = 0; i < PIXELS_PER_IMAGE; i++)
    {
        double pixel_difference = abs((double) img1.pixels[i] - (double) img2.pixels[i]);
        distance += pow(pixel_difference, minkowski_num);
    }

    return pow(distance, 1 / minkowski_num);
}


// Helper function, I'm using a little-endian machine
// Could be extended to handle both big and little endian,
// but that's not my concern right now.
int buffer_to_int(unsigned char buffer[4])
{
    return (int)buffer[3] | (int)buffer[2] << 8 | (int)buffer[1] << 16 | (int)buffer[0] << 24;
}


// qsort compare function for the distances
int compare_distance(const void *p1, const void *p2)
{
    double d1 = ((DISTANCE_ENTRY *)p1) -> distance;
    double d2 = ((DISTANCE_ENTRY *)p2) -> distance;

    if(d1 > d2)
    {
        return 1;
    }

    else if (d1 < d2)
    {
        return -1;
    }
    
    return 0;
}


// qsort compare function for user-defined label (used to
// determine most frequent nearest neighbor)
int compare_label(const void *p1, const void *p2)
{
    LABEL l1 = *(LABEL *)p1;
    LABEL l2 = *(LABEL *)p2;

    if(l1 > l2)
    {
        return 1;
    }

    else if(l1 < l2)
    {
        return -1;
    }

    return 0;
}


// Used to compute the time it takes to run the algorithms
double compute_time(clock_t begin)
{
    double time_ellapsed = ((double) clock() - begin) / CLOCKS_PER_SEC;
    printf("Time = %6.4f s\n", time_ellapsed);
    printf("Time = %6.4f m\n", time_ellapsed / 60);
    return time_ellapsed;
}


// Calculate accuracy
double classification_accuracy(LABEL *classificaitons)
{
    int correct = 0;

    for(int i = 0; i < TEST_SIZE; i++)
    {
        if(classificaitons[i] == test_labels[i])
        {
            correct++;
        }
    }

    double accuracy_rate = correct / TEST_SIZE;

    printf("Classification Accuracy = %6.4f\n",accuracy_rate);
    return accuracy_rate;
}