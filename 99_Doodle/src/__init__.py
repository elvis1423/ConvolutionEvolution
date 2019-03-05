import tensorflow as tf
from keras import backend as K
from keras.layers.core import Lambda, Flatten, Dense

a = tf.constant(-1.0, shape=[4, 3, 3])
b = tf.contrib.layers.flatten(a)
c = tf.nn.l2_normalize(b, dim=1)
d = tf.nn.l2_normalize(b, dim=0)
# b = tf.contrib.layers.fully_connected(inputs=flatten, num_outputs=128, activation_fn=tf.nn.relu)

with tf.Session() as sess:
    print(a)
    result1 = sess.run(b)
    print(result1)
    result2 = sess.run(c)
    print(result2)
    result3 = sess.run(d)
    print(result3)