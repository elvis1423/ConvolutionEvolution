import tensorflow as tf
from numpy import genfromtxt
from keras import backend as K
from keras.layers.core import Lambda, Flatten, Dense

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
v = tf.get_variable(name='inception_3a_5x5_bn1_v', shape=[16], initializer=tf.zeros_initializer())
bn1_v = genfromtxt(fname='..\\inception_3a_5x5_bn1_v.csv', delimiter=',', dtype=None)
print(bn1_v.shape)
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
    result4 = sess.run(activation)
    print(result4)
    # print('-------------------------------------------------')
    # sess.run(init)
    # print(v.eval())
    # result5 = sess.run(v, feed_dict={v: bn1_v})
    # print(result5)