import os
import numpy as np
import pandas as pd
import argparse
import h5py
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path to input image and label")
ap.add_argument("-wi", "--width",  type=int, default=500, help="width of image")
ap.add_argument("-hi", "--height", type=int, default=850, help="height of image")

args = vars(ap.parse_args())

path = args["path"]
height = args["height"]
width = args["width"]
num_channels = 3
figure_type = path.split('/')[-1]


def load_dataset(data_dir):
    img_dict = {}
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.bmp'):
                # load an image from file
                image = load_img(root + "/" + f, target_size=(height, width))
                # convert the image pixels to a numpy array
                image = img_to_array(image)
                # reshape data for the model
                # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                # prepare the image for the VGG model
                image = preprocess_input(image)
                img_dict[f.split('.bmp')[0]] = image

    labels = pd.read_csv(data_dir + "/labels.csv")
    id_sales = labels.values
    return img_dict, id_sales


# load dataset as image-dictionary and id-sales nd-array
image_dict, id_sales = load_dataset(path)
[num_data, _] = id_sales.shape
x = np.zeros(shape=[num_data, height, width, num_channels])
y = np.zeros(shape=[num_data, 1])

for i in range(num_data):
    id = id_sales[i, 0]
    y[i] = id_sales[i, 1]
    x[i, :, :, :] = image_dict[id]
# split train set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
h5f = h5py.File('data/' + figure_type + '/train_test_data.h5', 'w')
h5f.create_dataset('x_train', data=x_train)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('x_test', data=x_test)
h5f.create_dataset('y_test', data=y_test)

h5f.close()
