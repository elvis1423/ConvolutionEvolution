import data
import model
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp


loader = data.Data()
test_img, label = loader.get_test_img_and_label()

# tf.reset_default_graph()
# # Create some variables.
# v1 = tf.get_variable("v1", shape=[3])
# v2 = tf.get_variable("v2", shape=[5])
# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()
# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# with tf.Session() as sess:
#     # Restore variables from disk.
#     saver.restore(sess, "/tmp/model.ckpt")
#     print("Model restored.")
#     # Check the values of the variables
#     print("v1 : %s" % v1.eval())
#     print("v2 : %s" % v2.eval())
#
# chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
with tf.Session(config=config) as sess:
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('D:/practice/tensorflow/01_AlexNet/model/classification.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('D:/practice/tensorflow/01_AlexNet/model/'))
    # for variable in tf.all_variables():
    #     print(variable)
    #     print(variable.name)
    #
    # # Access saved Variables directly
    # print(sess.run('bias1:0'))

    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data
    graph = tf.get_default_graph()
    inputX = graph.get_tensor_by_name('inputX:0')
    groundTruth = graph.get_tensor_by_name('groundTruth:0')
    prediction = graph.get_tensor_by_name('prediction:0')

    feed_dict = {inputX: test_img, groundTruth: label}
    result = sess.run(tf.argmax(input=prediction, axis=-1), feed_dict)
    print("the inference result is classified as label: %d" % result)
