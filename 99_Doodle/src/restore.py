import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp


chkp.print_tensors_in_checkpoint_file("../model/face_encoding.ckpt", tensor_name='', all_tensors=True)
tf.reset_default_graph()
conv1_b_var = tf.get_variable("conv1_b", [64], initializer = tf.zeros_initializer)
bn1_b_var = tf.get_variable("bn1_b", [64], initializer = tf.zeros_initializer)
# face_vector_var = tf.get_variable('l2_normal', shape=[64])

saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
with tf.Session(config=config) as sess:
    conv1_b_var.initializer.run()

    saver.restore(sess, "../model/face_encoding.ckpt")
    print(conv1_b_var.eval())
    print('-----------------------------')
    print(bn1_b_var.eval())
    # print(face_vector_var.eval())
