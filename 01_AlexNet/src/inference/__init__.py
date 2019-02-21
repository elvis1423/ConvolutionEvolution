import data
import tensorflow as tf


loader = data.Data()
test_img, label = loader.get_test_img_and_label()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
with tf.Session(config=config) as sess:
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('../../model/classification.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('../../model/'))
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
    training = graph.get_tensor_by_name('training:0')

    feed_dict = {inputX: test_img, groundTruth: label, training: False}
    result = sess.run(tf.argmax(input=prediction, axis=-1), feed_dict)
    print(result)
