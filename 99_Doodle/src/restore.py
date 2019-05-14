import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from numpy import genfromtxt
import numpy as np

chkp.print_tensors_in_checkpoint_file("../model/face_encoding.ckpt", tensor_name='', all_tensors=True)
tf.reset_default_graph()

ones = np.ones(shape=[64], dtype=float)
b1 = genfromtxt(fname='..\\conv1_b.csv', delimiter=',', dtype=None) # 64
b2 = genfromtxt(fname='..\\bn1_b.csv', delimiter=',', dtype=None) # 64

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph('../model/face_encoding.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint("../model/"))
    graph = tf.get_default_graph()
    for variable in tf.global_variables():
        print(variable)
    print("---------------------------------------------")
    input_X = graph.get_tensor_by_name('input_X:0')
    face_vector_tensor = graph.get_tensor_by_name('l2_norm:0')
    face_vector = sess.run(face_vector_tensor, feed_dict={input_X: ones})
    print(face_vector)
    print("---------------------------------------------")
    conv1_b_tensor = graph.get_tensor_by_name('conv1_b:0')
    result = sess.run(conv1_b_tensor)
    print(result)

