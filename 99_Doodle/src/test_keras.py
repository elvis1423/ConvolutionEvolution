import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Input
from keras.models import Model
import numpy as np
from numpy import genfromtxt


conv_shape = {
  'conv1': [64, 3, 7, 7]
}

def faceRecoModel(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)
    model = Model(inputs=X_input, outputs=X, name='FaceRecoModel')
    return model


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### END CODE HERE ###

    return loss


w1 = genfromtxt(fname='..\\conv1_w.csv', delimiter=',', dtype=None) # 7x7x3x64
w1 = np.reshape(w1, conv_shape['conv1'])
w1 = np.transpose(w1, (2, 3, 1, 0))
b1 = genfromtxt(fname='..\\conv1_b.csv', delimiter=',', dtype=None) # 64


# w1 = np.array([[[[1.0]],[[1.0]],[[1.0]]],[[[1.0]],[[1.0]],[[1.0]]],[[[1.0]],[[1.0]],[[1.0]]]])
# b1 = np.array([2.0])
FRmodel = faceRecoModel(input_shape=(10, 10, 3))
print("Total Params:", FRmodel.count_params())
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
FRmodel.get_layer('conv1').set_weights([w1,b1])
# x=np.array([[[[1.0],[1.0],[1.0]],[[1.0],[1.0],[1.0]],[[1.0],[1.0],[1.0]]]])
x=np.ones([1, 10, 10, 3])
print('x shape: ' + str(x.shape))
result=FRmodel.predict_on_batch(x)
print(result[0,:,:,0])
