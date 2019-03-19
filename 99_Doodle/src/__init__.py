import tensorflow as tf
import numpy as np
from numpy import genfromtxt

conv_shape = {
  'conv1': [64, 3, 7, 7]
}


def Conv2D(input=None, filter_shape=None, strides=None, padding='VALID', data_format='NHWC', name=None):
    conv_w = tf.get_variable(name=name + '_w', shape=filter_shape, initializer=tf.zeros_initializer())
    if data_format == 'NHWC':
        conv_b = tf.get_variable(name=name + '_b', shape=[filter_shape[-1]], initializer=tf.zeros_initializer())
    else:
        conv_b = tf.get_variable(name=name + '_b', shape=[filter_shape[1]], initializer=tf.zeros_initializer())
    # conv_generation = tf.nn.conv2d(input=input, filter=conv_w, strides=strides, padding=padding,
    #                                data_format=data_format, name=name + '_generation',) + conv_b

    conv_generation = tf.layers.conv2d(inputs=input, filters=64, kernel_size=[7,7], strides=[2,2],padding='same',name=name)
    conv_generation = tf.reverse(tf.reverse(conv_generation, [-2]), [-3])
    return conv_generation


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

# w1 = tf.constant(1.0, shape=[3, 3, 1, 1])
# b1 = tf.constant(2.0, shape=[1])
# x = tf.constant(1.0, shape=[1, 3, 3, 1])


# activation = tf.add(tf.nn.conv2d(input=x, filter=w1, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC'), b1)
# paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
# activation = tf.pad(tensor=activation, paddings=paddings, mode='CONSTANT')
# v = tf.get_variable(name='inception_3a_5x5_bn1_vv', shape=[16], initializer=tf.zeros_initializer())
# bn_layer = BatchNormalization(input=v, shape=[16], name='inception_3a_5x5_bn1')
#
# bn_w = genfromtxt(fname='..\\inception_3a_5x5_bn1_w.csv', delimiter=',', dtype=None)
# bn_b = genfromtxt(fname='..\\inception_3a_5x5_bn1_b.csv', delimiter=',', dtype=None)
# bn_m = genfromtxt(fname='..\\inception_3a_5x5_bn1_m.csv', delimiter=',', dtype=None)
# bn_v = genfromtxt(fname='..\\inception_3a_5x5_bn1_v.csv', delimiter=',', dtype=None)
#
# flat = tf.get_variable(name='flatten', shape=[1, 3], initializer=tf.zeros_initializer())
# fc = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=4, activation_fn=None)
# x = tf.constant(1.0, shape=[1, 3, 3, 1])
# w1 = np.array([[[[1.0]],[[1.0]],[[1.0]]],[[[1.0]],[[1.0]],[[1.0]]],[[[1.0]],[[1.0]],[[1.0]]]])
# b1 = np.array([2.0])
x = tf.constant(1.0, shape=[1, 10, 10, 3])
w1 = genfromtxt(fname='..\\conv1_w.csv', delimiter=',', dtype=None) # 7x7x3x64
w1 = np.reshape(w1, conv_shape['conv1'])
w1 = np.transpose(w1, (2, 3, 1, 0))
w1 = np.flip(np.flip(w1, 0), 1)
# w1 = np.ones([7,7,3,64])
# b1 = genfromtxt(fname='..\\conv1_b.csv', delimiter=',', dtype=None) # 64
b1 = np.zeros([64])
generation = Conv2D(input=x, filter_shape=[7, 7, 3, 64], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC', name='conv1')
for variable in tf.global_variables():
    print(variable)
# print(bn_v)
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
with tf.Session(config=config) as sess:
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
    graph = tf.get_default_graph()
    # v_tensor = graph.get_tensor_by_name('inception_3a_5x5_bn1_vv:0')
    # feed_dictionary = {v_tensor: bn_v}
    # bn1_w_tensor = graph.get_tensor_by_name('inception_3a_5x5_bn1_w:0')
    # bn1_b_tensor = graph.get_tensor_by_name('inception_3a_5x5_bn1_b:0')
    # bn1_m_tensor = graph.get_tensor_by_name('inception_3a_5x5_bn1_m:0')
    # bn1_v_tensor = graph.get_tensor_by_name('inception_3a_5x5_bn1_v:0')
    # feed_dictionary[bn1_w_tensor] = bn_w
    # feed_dictionary[bn1_b_tensor] = bn_b
    # feed_dictionary[bn1_m_tensor] = bn_m
    # feed_dictionary[bn1_v_tensor] = bn_v
    # print(sess.run(v, feed_dict={v: bn_v}))
    # print(v.eval())
    #
    # # result5 = sess.run(bn_layer, feed_dict={v: bn_v})
    # result5 = sess.run(bn_layer, feed_dict=feed_dictionary)
    # print(result5)
    print('-------------------------------------------------')
    print('test for conv2d in tensor flow')
    conv_w_tensor = graph.get_tensor_by_name('conv1/kernel:0')
    conv_b_tensor = graph.get_tensor_by_name('conv1/bias:0')
    result = sess.run(generation, feed_dict={conv_w_tensor: w1, conv_b_tensor: b1})
    print(result[0,:,:,0])
