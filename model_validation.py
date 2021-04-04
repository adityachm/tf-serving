from tensorflow.keras import datasets
from skimage import io
import matplotlib.pyplot as plt
import random



(train_images,train_labels), (test_images, test_labels) = datasets.mnist.load_data()

test_labels = test_labels[:1000]

test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
# function to display image
def show(idx, title):
    plt.figure()
    plt.imshow(test_images[idx].reshape(28,28))
    plt.axis('off')
    plt.title('\n\n{}'.format(test_labels[idx]), fontdict={'size': 16})

# generate a random index
r = random.randint(0,len(test_images)-1)

#
print("Random Number Generated: ", r, "Image Label : ", test_labels[r])
show(r, 'Image: {}'.format(test_images[r]))