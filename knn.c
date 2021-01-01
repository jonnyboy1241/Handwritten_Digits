#include <assert.h>
#include <stdio.h>
#include <stdlib.h>


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


// Function Declarations
void load_data();
void load_training_images();
void load_training_labels();
void load_test_images();
void load_test_labels();
void free_data();
int buffer_to_int(unsigned char buffer[4]);


int main()
{
    load_data();
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


// Helper function, I'm using a little-endian machine
int buffer_to_int(unsigned char buffer[4])
{
    return (int)buffer[3] | (int)buffer[2] << 8 | (int)buffer[1] << 16 | (int)buffer[0] << 24;
}