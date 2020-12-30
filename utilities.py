import matplotlib.pyplot as plt

def view_image(image):
    image = image.reshape(28, 28)
    plt.gray()
    plt.imshow(image)
    plt.show()
