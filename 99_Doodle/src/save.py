import tensorflow as tf
from numpy import genfromtxt


var1 = tf.get_variable(name="conv1_b", shape=[64], initializer=tf.zeros_initializer)
var2 = tf.get_variable(name="bn1_b", shape=[64], initializer=tf.zeros_initializer)

face_embedding = tf.nn.l2_normalize(x=var1, axis=-1, name='l2_normal')

b1 = genfromtxt(fname='..\\conv1_b.csv', delimiter=',', dtype=None) # 64
b2 = genfromtxt(fname='..\\bn1_b.csv', delimiter=',', dtype=None) # 64

weight_dict = {var1: b1, var2: b2}

graph = tf.get_default_graph()
conv1_b_tensor = graph.get_tensor_by_name('conv1_b:0')

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(init)
    # assign_op = var1.assign(b1)
    result = sess.run(face_embedding, feed_dict={conv1_b_tensor: b1})
    print('feed_dict to face_embedding:')
    print(result)
    for key, value in weight_dict.items():
        assign_op = key.assign(value)
        sess.run(assign_op)

    print('assign b1 to var:')
    print(var1.eval())
    print('assign b2 to var:')
    print(var2.eval())
    saver.save(sess, '../model/face_encoding.ckpt')
