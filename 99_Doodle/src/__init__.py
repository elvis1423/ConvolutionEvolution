import tensorflow as tf
from numpy import genfromtxt


def BatchNormalization(input=None, shape=None, name=None, epsilon=0.00001):
    bn_beta = tf.get_variable(name=name+'_w', shape=shape, trainable=False, initializer=tf.zeros_initializer())  # beta of batch normalization
    bn_gamma = tf.get_variable(name=name+'_b', shape=shape, trainable=False, initializer=tf.zeros_initializer())  # gamma of batch normalization
    bn_mean = tf.get_variable(name=name+'_m', shape=shape, trainable=False, initializer=tf.zeros_initializer())  # moving mean of batch normalization
    bn_variance = tf.get_variable(name=name+'_v', shape=shape, trainable=False, initializer=tf.zeros_initializer())  # moving variance of batch normalization
    bn = tf.nn.batch_normalization(x=input, variance_epsilon=epsilon, mean=bn_mean, variance=bn_variance,
                                   offset=bn_beta, scale=bn_gamma, name=name)
    return bn

a = tf.constant(-1.0, shape=[4, 3, 3])
b = tf.contrib.layers.flatten(a)
c = tf.nn.l2_normalize(b, dim=1)
d = tf.nn.l2_normalize(b, dim=0)
# b = tf.contrib.layers.fully_connected(inputs=flatten, num_outputs=128, activation_fn=tf.nn.relu)

w1 = tf.constant(1.0, shape=[3, 3, 1, 2])
b1 = tf.constant(2.0, shape=[2])
x = tf.constant(1.0, shape=[4, 5, 5, 1])

activation = tf.add(tf.nn.conv2d(input=x, filter=w1, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC'), b1)
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
activation = tf.pad(tensor=activation, paddings=paddings, mode='CONSTANT')
v = tf.get_variable(name='inception_3a_5x5_bn1_vv', shape=[16], initializer=tf.zeros_initializer())
bn_layer = BatchNormalization(input=v, shape=[16], name='inception_3a_5x5_bn1')

bn_w = genfromtxt(fname='..\\inception_3a_5x5_bn1_w.csv', delimiter=',', dtype=None)
bn_b = genfromtxt(fname='..\\inception_3a_5x5_bn1_b.csv', delimiter=',', dtype=None)
bn_m = genfromtxt(fname='..\\inception_3a_5x5_bn1_m.csv', delimiter=',', dtype=None)
bn_v = genfromtxt(fname='..\\inception_3a_5x5_bn1_v.csv', delimiter=',', dtype=None)

flat = tf.get_variable(name='flatten', shape=[1, 3], initializer=tf.zeros_initializer())
fc = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=4, activation_fn=None)


print(bn_v.shape)
init = tf.global_variables_initializer()
with tf.Session() as sess:

    # print(a)
    # result1 = sess.run(b)
    # print(result1)
    # result2 = sess.run(c)
    # print(result2)
    # result3 = sess.run(d)
    # print(result3)
    # print('-------------------------------------------------')
    # result4 = sess.run(activation)
    # print(result4)
    print('-------------------------------------------------')
    sess.run(init)
    print(v.eval())
    result5 = sess.run(v, feed_dict={v: bn_v})
    print(result5)