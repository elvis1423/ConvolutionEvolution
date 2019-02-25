import tensorflow as tf
import model
import data


h = 244
w = 244
c = 3
labels = 1000
learning_rate = 0.002
num_epochs = 10


def train(Model, X_train, Y_train, learning_rate=0.002, num_epochs=10, training=True):
    Y_logits = Model.forward_propagation()  # y_hat shape=(?, 1000)
    prediction = tf.nn.softmax(logits=Y_logits, name='prediction')  # Used for retrieving by name when inference
    cost = Model.cost(Y_logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=cost)

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            _, train_loss = sess.run([optimizer, cost], feed_dict={Model.X_holder: X_train,
                                                                   Model.Y_holder: Y_train, Model.Training: training})
            print('epoch %i cost: %f' % (epoch, train_loss))
        saver.save(sess, '../../model/classification.ckpt')


data = data.Data()
x, label = data.get_test_img_and_label()

AlexNet = model.Model(h, w, c, labels)
train(AlexNet, x, label, learning_rate, num_epochs)

