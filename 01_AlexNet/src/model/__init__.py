import tensorflow as tf

h = 227
w = 227
c = 3
labels = 1000


class Model:
    def __init__(self, height, width, channel, num_labels):
        print('This is the __init__ of class Model')
        tf.reset_default_graph()
        tf.set_random_seed(1)   # to keep results consistent (tensorflow seed)
        self.X_holder = tf.placeholder(dtype=tf.float32, shape=[None, height, width, channel], name='inputX')
        self.Y_holder = tf.placeholder(dtype=tf.float32, shape=[None, num_labels], name='groundTruth')
        self.Training = tf.placeholder(dtype=tf.bool, shape=[1], name='training')

    def forward_propagation(self):
        filter1 = tf.get_variable(name='filter1', shape=[11, 11, 3, 96],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=0))
        bias1 = tf.get_variable(name='bias1', shape=[96], initializer=tf.zeros_initializer())

        filter2 = tf.get_variable(name='filter2', shape=[5, 5, 96, 256],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=0))
        bias2 = tf.get_variable(name='bias2', shape=[256], initializer=tf.zeros_initializer())

        filter3 = tf.get_variable(name='filter3', shape=[3, 3, 256, 384],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=0))
        bias3 = tf.get_variable(name='bias3', shape=[384], initializer=tf.zeros_initializer())

        filter4 = tf.get_variable(name='filter4', shape=[3, 3, 384, 384],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=0))
        bias4 = tf.get_variable(name='bias4', shape=[384], initializer=tf.zeros_initializer())

        filter5 = tf.get_variable(name='filter5', shape=[3, 3, 384, 256],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=0))
        bias5 = tf.get_variable(name='bias5', shape=[256], initializer=tf.zeros_initializer())

        Activation1_0 = tf.nn.conv2d(input=self.X_holder, filter=filter1, strides=[1, 4, 4, 1], padding='VALID',
                                     data_format='NHWC', name='conv1')  # VALID here means no padding
        Activation1_0 = tf.nn.bias_add(value=Activation1_0, bias=bias1, data_format='NHWC', name='bias1_add')
        Activation1_0 = tf.nn.relu(Activation1_0, name='relu1')

        Activation1_1 = tf.nn.max_pool(value=Activation1_0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                                       data_format='NHWC', name='maxpool1')

        Activation2_0 = tf.nn.conv2d(input=Activation1_1, filter=filter2, strides=[1, 1, 1, 1], padding='SAME',
                                     data_format='NHWC', name='conv2')  # SAME here with stride=1 means no h/w shrinking
        Activation2_0 = tf.nn.bias_add(value=Activation2_0, bias=bias2, data_format='NHWC', name='bias2_add')
        Activation2_0 = tf.nn.relu(Activation2_0, name='relu2')

        Activation2_1 = tf.nn.max_pool(value=Activation2_0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                                       data_format='NHWC', name='maxpool2')

        Activation3_0 = tf.nn.conv2d(input=Activation2_1, filter=filter3, strides=[1, 1, 1, 1], padding='SAME',
                                     data_format='NHWC', name='conv3')
        Activation3_0 = tf.nn.bias_add(value=Activation3_0, bias=bias3, data_format='NHWC', name='bias3_add')
        Activation3_0 = tf.nn.relu(Activation3_0, name='relu3')

        Activation4_0 = tf.nn.conv2d(input=Activation3_0, filter=filter4, strides=[1, 1, 1, 1], padding='SAME',
                                     data_format='NHWC', name='conv4')
        Activation4_0 = tf.nn.bias_add(value=Activation4_0, bias=bias4, data_format='NHWC', name='bias4_add')
        Activation4_0 = tf.nn.relu(Activation4_0, name='relu4')

        Activation5_0 = tf.nn.conv2d(input=Activation4_0, filter=filter5, strides=[1, 1, 1, 1], padding='SAME',
                                     data_format='NHWC', name='conv5')
        Activation5_0 = tf.nn.bias_add(value=Activation5_0, bias=bias5, data_format='NHWC', name='bias5_add')
        Activation5_0 = tf.nn.relu(Activation5_0, name='relu5')

        Activation5_1 = tf.nn.max_pool(value=Activation5_0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                                       data_format='NHWC', name='maxpool5')
        flattened = tf.contrib.layers.flatten(Activation5_1)

        fc6 = tf.contrib.layers.fully_connected(inputs=flattened, num_outputs=4096, activation_fn=tf.nn.relu)
        fc6 = tf.layers.dropout(inputs=fc6, rate=0.5, seed=1, training=self.Training, name='dropout_fc6')
        fc7 = tf.contrib.layers.fully_connected(inputs=fc6, num_outputs=4096, activation_fn=tf.nn.relu)
        fc7 = tf.layers.dropout(input=fc7, rate=0.5, seed=2, training=self.Training, name='dropout_fc7')
        fc8 = tf.contrib.layers.fully_connected(inputs=fc7, num_outputs=1000, activation_fn=None)

        return fc8

    def cost(self, Y_logits):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_holder, logits=Y_logits)
        reduce_mean = tf.reduce_mean(cross_entropy)
        return reduce_mean

    def train(self, X_train, Y_train, learning_rate=0.002, num_epochs=10, training=True):

        Y_logits = self.forward_propagation()  # y_hat shape=(?, 1000)
        prediction = tf.nn.softmax(logits=Y_logits, name='prediction')
        cost = self.cost(Y_logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=cost)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            sess.run(init)
            for epoch in range(num_epochs):
                _, train_loss = sess.run([optimizer, cost], feed_dict={self.X_holder: X_train,
                                                                       self.Y_holder: Y_train, self.Training: training})
                print('epoch %i cost: %f' % (epoch, train_loss))
            saver.save(sess, '../../model/classification.ckpt')
