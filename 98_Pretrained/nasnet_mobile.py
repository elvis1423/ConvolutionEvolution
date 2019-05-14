from keras.applications.nasnet import NASNetMobile
from keras.preprocessing.image import load_img
from keras.applications.nasnet import preprocess_input
from keras.preprocessing.image import img_to_array


width = 256
height = 256
modelNASNet = NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
image = load_img('images/mug.jpg', target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)

yhat = modelNASNet.predict(image)


from keras.applications.nasnet import decode_predictions
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))