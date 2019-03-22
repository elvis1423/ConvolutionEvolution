import model
import data
import tensorflow as tf


data = data.Data()
younes = data.get_camera_face_by('../../images/younes.jpg')

# face_net = model.FaceNet()
# face_vector = face_net.forward()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('../../model/face_encoding.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('../../model/'))
    graph = tf.get_default_graph()
    inputX_tensor = graph.get_tensor_by_name('inputX:0')
    face_vector_tensor = graph.get_tensor_by_name('l2_normal:0')
    face_vector = sess.run(face_vector_tensor, feed_dict={inputX_tensor: younes})
    print(face_vector)
