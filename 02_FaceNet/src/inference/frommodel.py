import data
import tensorflow as tf


data = data.Data()
younes = data.get_camera_face_by('../../images/younes.jpg')

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph('../../model/face_encoding.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('../../model/'))
    for variable in tf.all_variables():
        print(variable)
        print(variable.name)
    graph = tf.get_default_graph()
    InputX_tensor = graph.get_tensor_by_name('InputX:0')
    face_vector_tensor = graph.get_tensor_by_name('l2_normal:0')
    face_vector = sess.run(face_vector_tensor, feed_dict={InputX_tensor: younes})
    print(face_vector)
    print("---------------------------------------------")
    conv1_b_tensor = graph.get_tensor_by_name('conv1_b:0')
    result = sess.run(conv1_b_tensor)
    print(result)
