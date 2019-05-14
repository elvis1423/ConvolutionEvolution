from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,Callback
import h5py
import argparse
import tensorflow as tf


class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        _lr = tf.to_float(optimizer.lr, name='ToFloat')
        _decay = tf.to_float(optimizer.decay, name='ToFloat')
        _iter = tf.to_float(optimizer.iterations, name='ToFloat')
        lr = K.eval(_lr * (1. / (1. + _decay * _iter)))
        print('\nLR: {:.6f}\n'.format(lr))

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path to h5 data file")
ap.add_argument("-wi", "--width",  type=int, default=500, help="width of image")
ap.add_argument("-hi", "--height", type=int, default=850, help="height of image")

args = vars(ap.parse_args())

h5f = h5py.File('data/P01/train_test_data.h5', 'r')
x_train = h5f['x_train'][:]
y_train = h5f['y_train'][:]
x_test = h5f['x_test'][:]
y_test = h5f['y_test'][:]
h5f.close()

print("shape of x_train: " +  str(x_train.shape))
print("shape of y_train: " +  str(y_train.shape))
print("shape of x_test: " + str(x_test.shape))
print("shape of y_test: " +  str(y_test.shape))

[m, height, width, channel] = x_train.shape

model = VGG16(weights="imagenet", include_top=False, input_shape=(height, width, channel))
for layer in model.layers[:5]:
    layer.trainable = False
# Adding custom Layers
x = model.output
x = Flatten()(x)
predictions = Dense(1, activation="relu")(x)

# creating the final model
model_final = Model(input=model.input, output=predictions)
model_final.summary()

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
cbs = []

model_save_path = "model/P01/20190514/weights-improve-{epoch:05d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(model_save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
cbs.append(checkpoint)
tracker_lr = SGDLearningRateTracker()
cbs.append(tracker_lr)

model_final.compile(loss="mean_squared_error", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=['mae', 'acc'])
model_final.fit(x=x_train, y=y_train, validation_split=0.3, epochs=1, batch_size=1, callbacks=cbs)
test_predict = model_final.predict(x_test)
train_predict = model_final.predict(x_train)
print("test predict: " + str(test_predict))
print("train predict: " + str(train_predict))

# # load the model
# model = VGG16()
# # load an image from file
# image = load_img('images/P01_1_A01.bmp', target_size=(224, 224))
# # convert the image pixels to a numpy array
# image = img_to_array(image)
# # reshape data for the model
# image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# # prepare the image for the VGG model
# image = preprocess_input(image)
# # predict the probability across all output classes
# yhat = model.predict(image)
# # convert the probabilities to class labels
# label = decode_predictions(yhat)
# # retrieve the most likely result, e.g. highest probability
# label = label[0][0]
# # print the classification
# print('%s (%.2f%%)' % (label[1], label[2]*100))
