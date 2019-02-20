import tensorflow as tf
import model
import data


h = 244
w = 244
c = 3
labels = 1000
learning_rate = 0.002
num_epochs = 10

# v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
# v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)
#
# inc_v1 = v1.assign(v1+1)
# dec_v2 = v2.assign(v2-1)
#
# # Add an op to initialize the variables.
# init_op = tf.global_variables_initializer()
#
# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()
#
# for variable in tf.all_variables():
#     print(variable)
#     print(variable.name)
#
# # Later, launch the model, initialize the variables, do some work, and save the
# # variables to disk.
# with tf.Session() as sess:
#     sess.run(init_op)
#     # Do some work with the model.
#     inc_v1.op.run()
#     dec_v2.op.run()
#     # Save the variables to disk.
#     save_path = saver.save(sess, "/tmp/model.ckpt")
#     print("Model saved in path: %s" % save_path)

data = data.Data()
x, label = data.get_test_img_and_label()

AlexNet = model.Model(h, w, c, labels)
AlexNet.train(x, label, learning_rate, num_epochs)

