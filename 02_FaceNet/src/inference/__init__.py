import model
import data
import tensorflow as tf
import numpy as np


data = data.Data()
weights_dict = data.load_weights()
input_x = data.get_camera_face_for_test()

younes = data.get_camera_face_by('../../images/younes.jpg')
benoit = data.get_camera_face_by('../../images/camera_2.jpg')

face_net = model.FaceNet()
face_vector = face_net.forward()

for variable in tf.global_variables():
    print(variable)

graph = tf.get_default_graph()
InputX = graph.get_tensor_by_name('InputX:0')
feed_diction = {InputX: input_x}
for key, value in weights_dict.items():
    if 'conv' in key:  # have conv_w and conv_b parameters
        conv_w = value[0]
        conv_b = value[1]
        conv_w_tensor = graph.get_tensor_by_name(key + '_w:0')
        conv_b_tensor = graph.get_tensor_by_name(key + '_b:0')
        feed_diction[conv_w_tensor] = conv_w
        feed_diction[conv_b_tensor] = conv_b
    elif 'bn' in key:  # have bn_w, bn_b, bn_m, bn_v parameters
        bn_w = value[0]
        bn_b = value[1]
        bn_m = value[2]
        bn_v = value[3]
        bn_w_tensor = graph.get_tensor_by_name(key + '_w:0')
        bn_b_tensor = graph.get_tensor_by_name(key + '_b:0')
        bn_m_tensor = graph.get_tensor_by_name(key + '_m:0')
        bn_v_tensor = graph.get_tensor_by_name(key + '_v:0')
        feed_diction[bn_w_tensor] = bn_w
        feed_diction[bn_b_tensor] = bn_b
        feed_diction[bn_m_tensor] = bn_m
        feed_diction[bn_v_tensor] = bn_v
    elif 'dense' in key:  # have dense_w, dense_b parameters
        fc_w = value[0]
        fc_b = value[1]
        fc_w_tensor = graph.get_tensor_by_name('fully_connected/weights:0')
        fc_b_tensor = graph.get_tensor_by_name('fully_connected/biases:0')
        feed_diction[fc_w_tensor] = fc_w
        feed_diction[fc_b_tensor] = fc_b

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(init)
    anchor = sess.run(face_vector, feed_dict=feed_diction)
    feed_diction[InputX] = younes
    positive = sess.run(face_vector, feed_dict=feed_diction)
    feed_diction[InputX] = benoit
    negative = sess.run(face_vector, feed_dict=feed_diction)

    for tensor, param in feed_diction.items():
        assign_op = tensor.assign(param)
        sess.run(assign_op)

    saver.save(sess, '../../model/face_encoding.ckpt')
    print(anchor)
    print('-------------------------------------------------')
    print(positive)
    print('-------------------------------------------------')
    print(negative)
    print('-------------------------------------------------')
    print('distance between anchor and positive: ' + str(np.linalg.norm(anchor-positive)))
    print('distance between anchor and negative: ' + str(np.linalg.norm(anchor-negative)))
