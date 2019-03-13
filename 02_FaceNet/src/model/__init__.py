import tensorflow as tf


def Conv2D(input=None, filter_shape=None, strides=None, padding='VALID', data_format='NHWC', name=None):
    conv_w = tf.get_variable(name=name + '_w', shape=filter_shape, initializer=tf.zeros_initializer())
    if data_format == 'NHWC':
        conv_b = tf.get_variable(name=name + '_b', shape=[filter_shape[-1]], initializer=tf.zeros_initializer())
    else:
        conv_b = tf.get_variable(name=name + '_b', shape=[filter_shape[1]], initializer=tf.zeros_initializer())
    conv_generation = tf.nn.conv2d(input=input, filter=conv_w, strides=strides, padding=padding,
                                   data_format=data_format, name=name + '_generation',) + conv_b
    return conv_generation


def BatchNormalization(input=None, shape=None, name=None, epsilon=0.00001):
    bn_beta = tf.get_variable(name=name+'_w', shape=shape, trainable=False, initializer=tf.zeros_initializer())  # beta of batch normalization
    bn_gamma = tf.get_variable(name=name+'_b', shape=shape, trainable=False, initializer=tf.zeros_initializer())  # gamma of batch normalization
    bn_mean = tf.get_variable(name=name+'_m', shape=shape, trainable=False, initializer=tf.zeros_initializer())  # moving mean of batch normalization
    bn_variance = tf.get_variable(name=name+'_v', shape=shape, trainable=False, initializer=tf.zeros_initializer())  # moving variance of batch normalization
    bn = tf.nn.batch_normalization(x=input, variance_epsilon=epsilon, mean=bn_mean, variance=bn_variance,
                                   offset=bn_beta, scale=bn_gamma, name=name)
    return bn


def Inception_block_1a(input=None):
    X_1x1 = Conv2D(input=input, filter_shape=[1, 1, 192, 64], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_3a_1x1_conv')
    X_1x1 = BatchNormalization(input=X_1x1, shape=[64], name='inception_3a_1x1_bn')
    X_1x1 = tf.nn.relu(features=X_1x1, name='inception_3a_1x1_relu')

    X_3x3 = Conv2D(input=input, filter_shape=[1, 1, 192, 96], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_3a_3x3_conv1')
    X_3x3 = BatchNormalization(input=X_3x3, shape=[96], name='inception_3a_3x3_bn1')
    X_3x3 = tf.nn.relu(features=X_3x3, name='inception_3a_3x3_relu1')
    X_3x3 = Conv2D(input=X_3x3, filter_shape=[3, 3, 96, 128], strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', name='inception_3a_3x3_conv2')
    X_3x3 = BatchNormalization(input=X_3x3, shape=[128], name='inception_3a_3x3_bn2')
    X_3x3 = tf.nn.relu(features=X_3x3, name='inception_3a_3x3_relu2')

    X_5x5 = Conv2D(input=input, filter_shape=[1, 1, 192, 16], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_3a_5x5_conv1')
    X_5x5 = BatchNormalization(input=X_5x5, shape=[16], name='inception_3a_5x5_bn1')
    X_5x5 = tf.nn.relu(features=X_5x5, name='inception_3a_5x5_relu1')
    X_5x5 = Conv2D(input=X_5x5, filter_shape=[5, 5, 16, 32], strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', name='inception_3a_5x5_conv2')
    X_5x5 = BatchNormalization(input=X_5x5, shape=[32], name='inception_3a_5x5_bn2')
    X_5x5 = tf.nn.relu(features=X_5x5, name='inception_3a_5x5_relu2')

    X_pool = tf.nn.max_pool(value=input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC', name='inception_3a_pool_max')
    X_pool = Conv2D(input=X_pool, filter_shape=[1, 1, 192, 32], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_3a_pool_conv')
    X_pool = BatchNormalization(input=X_pool, shape=[32], name='inception_3a_pool_bn')
    X_pool = tf.nn.relu(features=X_pool, name='inception_3a_pool_relu')
    paddings = tf.constant([[0, 0], [3, 4], [3, 4], [0, 0]])
    X_pool = tf.pad(tensor=X_pool, paddings=paddings, mode='CONSTANT')
    inception_1a = tf.concat(values=[X_1x1, X_3x3, X_5x5, X_pool], axis=-1, name='inception_3a_concat')
    return inception_1a


def Inception_block_1b(input=None):
    X_1x1 = Conv2D(input=input, filter_shape=[1, 1, 256, 64], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_3b_1x1_conv')
    X_1x1 = BatchNormalization(input=X_1x1, shape=[64], name='inception_3b_1x1_bn')
    X_1x1 = tf.nn.relu(features=X_1x1, name='inception_3b_1x1_relu')

    X_3x3 = Conv2D(input=input, filter_shape=[1, 1, 256, 96], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_3b_3x3_conv1')
    X_3x3 = BatchNormalization(input=X_3x3, shape=[96], name='inception_3b_3x3_bn1')
    X_3x3 = tf.nn.relu(features=X_3x3, name='inception_3b_3x3_relu1')
    X_3x3 = Conv2D(input=X_3x3, filter_shape=[3, 3, 96, 128], strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', name='inception_3b_3x3_conv2')
    X_3x3 = BatchNormalization(input=X_3x3, shape=[128], name='inception_3b_3x3_bn2')
    X_3x3 = tf.nn.relu(features=X_3x3, name='inception_3b_3x3_relu2')

    X_5x5 = Conv2D(input=input, filter_shape=[1, 1, 256, 32], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_3b_5x5_conv1')
    X_5x5 = BatchNormalization(input=X_5x5, shape=[32], name='inception_3b_5x5_bn1')
    X_5x5 = tf.nn.relu(features=X_5x5, name='inception_3b_5x5_relu1')
    X_5x5 = Conv2D(input=X_5x5, filter_shape=[5, 5, 32, 64], strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', name='inception_3b_5x5_conv2')
    X_5x5 = BatchNormalization(input=X_5x5, shape=[64], name='inception_3b_5x5_bn2')
    X_5x5 = tf.nn.relu(features=X_5x5, name='inception_3b_5x5_relu2')

    X_pool = tf.nn.avg_pool(value=input, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID', data_format='NHWC', name='inception_3b_pool_avg')
    X_pool = Conv2D(input=X_pool, filter_shape=[1, 1, 256, 64], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_3b_pool_conv')
    X_pool = BatchNormalization(input=X_pool, shape=[64], name='inception_3b_pool_bn')
    X_pool = tf.nn.relu(features=X_pool, name='inception_3b_pool_relu')
    paddings = tf.constant([[0, 0], [4, 4], [4, 4], [0, 0]])
    X_pool = tf.pad(tensor=X_pool, paddings=paddings, mode='CONSTANT')
    inception_1b = tf.concat(values=[X_1x1, X_3x3, X_5x5, X_pool], axis=-1, name='inception_3b_concat')
    return inception_1b


def Inception_block_1c(input=None):
    X_3x3 = Conv2D(input=input, filter_shape=[1, 1, 320, 128], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_3c_3x3_conv1')
    X_3x3 = BatchNormalization(input=X_3x3, shape=[128], name='inception_3c_3x3_bn1')
    X_3x3 = tf.nn.relu(features=X_3x3, name='inception_3c_3x3_relu1')
    X_3x3 = Conv2D(input=X_3x3, filter_shape=[3, 3, 128, 256], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC', name='inception_3c_3x3_conv2')
    X_3x3 = BatchNormalization(input=X_3x3, shape=[256], name='inception_3c_3x3_bn2')
    X_3x3 = tf.nn.relu(features=X_3x3, name='inception_3c_3x3_relu2')

    X_5x5 = Conv2D(input=input, filter_shape=[1, 1, 320, 32], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_3c_5x5_conv1')
    X_5x5 = BatchNormalization(input=X_5x5, shape=[32], name='inception_3c_5x5_bn1')
    X_5x5 = tf.nn.relu(features=X_5x5, name='inception_3c_5x5_relu1')
    X_5x5 = Conv2D(input=X_5x5, filter_shape=[5, 5, 32, 64], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC', name='inception_3c_5x5_conv2')
    X_5x5 = BatchNormalization(input=X_5x5, shape=[64], name='inception_3c_5x5_bn2')
    X_5x5 = tf.nn.relu(features=X_5x5, name='inception_3c_5x5_relu2')

    X_pool = tf.nn.max_pool(value=input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC', name='inception_3c_pool_max')
    paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
    X_pool = tf.pad(tensor=X_pool, paddings=paddings, mode='CONSTANT')
    inception_1c = tf.concat(values=[X_3x3, X_5x5, X_pool], axis=-1, name='inception_3c_concat')
    return inception_1c


def Inception_block_2a(input=None):
    X_1x1 = Conv2D(input=input, filter_shape=[1, 1, 640, 256], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_4a_1x1_conv')
    X_1x1 = BatchNormalization(input=X_1x1, shape=[256], name='inception_4a_1x1_bn')
    X_1x1 = tf.nn.relu(features=X_1x1, name='inception_4a_1x1_relu')

    X_3x3 = Conv2D(input=input, filter_shape=[1, 1, 640, 96], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_4a_3x3_conv1')
    X_3x3 = BatchNormalization(input=X_3x3, shape=[96], name='inception_4a_3x3_bn1')
    X_3x3 = tf.nn.relu(features=X_3x3, name='inception_4a_3x3_relu1')
    X_3x3 = Conv2D(input=X_3x3, filter_shape=[3, 3, 96, 192], strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', name='inception_4a_3x3_conv2')
    X_3x3 = BatchNormalization(input=X_3x3, shape=[192], name='inception_4a_3x3_bn2')
    X_3x3 = tf.nn.relu(features=X_3x3, name='inception_4a_3x3_relu2')

    X_5x5 = Conv2D(input=input, filter_shape=[1, 1, 640, 32], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_4a_5x5_conv1')
    X_5x5 = BatchNormalization(input=X_5x5, shape=[32], name='inception_4a_5x5_bn1')
    X_5x5 = tf.nn.relu(features=X_5x5, name='inception_4a_5x5_relu1')
    X_5x5 = Conv2D(input=X_5x5, filter_shape=[5, 5, 32, 64], strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', name='inception_4a_5x5_conv2')
    X_5x5 = BatchNormalization(input=X_5x5, shape=[64], name='inception_4a_5x5_bn2')
    X_5x5 = tf.nn.relu(features=X_5x5, name='inception_4a_5x5_relu2')

    X_pool = tf.nn.avg_pool(value=input, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID', data_format='NHWC', name='inception_4a_pool_avg')
    X_pool = Conv2D(input=X_pool, filter_shape=[1, 1, 640, 128], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_4a_pool_conv')
    X_pool = BatchNormalization(input=X_pool, shape=[128], name='inception_4a_pool_bn')
    X_pool = tf.nn.relu(features=X_pool, name='inception_4a_pool_relu')
    paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
    X_pool = tf.pad(tensor=X_pool, paddings=paddings, mode='CONSTANT')
    inception_2a = tf.concat(values=[X_1x1, X_3x3, X_5x5, X_pool], axis=-1, name='inception_4a_concat')
    return inception_2a


def Inception_block_2b(input=None):
    X_3x3 = Conv2D(input=input, filter_shape=[1, 1, 640, 160], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_4e_3x3_conv1')
    X_3x3 = BatchNormalization(input=X_3x3, shape=[160], name='inception_4e_3x3_bn1')
    X_3x3 = tf.nn.relu(features=X_3x3, name='inception_4e_3x3_relu1')
    X_3x3 = Conv2D(input=X_3x3, filter_shape=[3, 3, 160, 256], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC', name='inception_4e_3x3_conv2')
    X_3x3 = BatchNormalization(input=X_3x3, shape=[256], name='inception_4e_3x3_bn2')
    X_3x3 = tf.nn.relu(features=X_3x3, name='inception_4e_3x3_relu2')

    X_5x5 = Conv2D(input=input, filter_shape=[1, 1, 640, 64], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_4e_5x5_conv1')
    X_5x5 = BatchNormalization(input=X_5x5, shape=[64], name='inception_4e_5x5_bn1')
    X_5x5 = tf.nn.relu(features=X_5x5, name='inception_4e_5x5_relu1')
    X_5x5 = Conv2D(input=X_5x5, filter_shape=[5, 5, 64, 128], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC', name='inception_4e_5x5_conv2')
    X_5x5 = BatchNormalization(input=X_5x5, shape=[128], name='inception_4e_5x5_bn2')
    X_5x5 = tf.nn.relu(features=X_5x5, name='inception_4e_5x5_relu2')

    X_pool = tf.nn.max_pool(value=input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC', name='inception_4e_pool_max')
    paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
    X_pool = tf.pad(tensor=X_pool, paddings=paddings, mode='CONSTANT')
    inception_2b = tf.concat(values=[X_3x3, X_5x5, X_pool], axis=-1, name='inception_4e_concat')
    return inception_2b


def Inception_block_3a(input=None):
    X_1x1 = Conv2D(input=input, filter_shape=[1, 1, 1024, 256], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_5a_1x1_conv')
    X_1x1 = BatchNormalization(input=X_1x1, shape=[256], name='inception_5a_1x1_bn')
    X_1x1 = tf.nn.relu(features=X_1x1, name='inception_5a_1x1_relu')

    X_3x3 = Conv2D(input=input, filter_shape=[1, 1, 1024, 96], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_5a_3x3_conv1')
    X_3x3 = BatchNormalization(input=X_3x3, shape=[96], name='inception_5a_3x3_bn1')
    X_3x3 = tf.nn.relu(features=X_3x3, name='inception_5a_3x3_relu1')
    X_3x3 = Conv2D(input=X_3x3, filter_shape=[3, 3, 96, 384], strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', name='inception_5a_3x3_conv2')
    X_3x3 = BatchNormalization(input=X_3x3, shape=[384], name='inception_5a_3x3_bn2')
    X_3x3 = tf.nn.relu(features=X_3x3, name='inception_5a_3x3_relu2')

    X_pool = tf.nn.avg_pool(value=input, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID', data_format='NHWC', name='inception_5a_pool_avg')
    X_pool = Conv2D(input=X_pool, filter_shape=[1, 1, 1024, 96], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_5a_pool_conv')
    X_pool = BatchNormalization(input=X_pool, shape=[96], name='inception_5a_pool_bn')
    X_pool = tf.nn.relu(features=X_pool, name='inception_5a_pool_relu')
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    X_pool = tf.pad(tensor=X_pool, paddings=paddings, mode='CONSTANT')
    inception_2a = tf.concat(values=[X_1x1, X_3x3, X_pool], axis=-1, name='inception_5a_concat')
    return inception_2a


def Inception_block_3b(input=None):
    X_1x1 = Conv2D(input=input, filter_shape=[1, 1, 736, 256], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_5b_1x1_conv')
    X_1x1 = BatchNormalization(input=X_1x1, shape=[256], name='inception_5b_1x1_bn')
    X_1x1 = tf.nn.relu(features=X_1x1, name='inception_5b_1x1_relu')

    X_3x3 = Conv2D(input=input, filter_shape=[1, 1, 736, 96], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_5b_3x3_conv1')
    X_3x3 = BatchNormalization(input=X_3x3, shape=[96], name='inception_5b_3x3_bn1')
    X_3x3 = tf.nn.relu(features=X_3x3, name='inception_5b_3x3_relu1')
    X_3x3 = Conv2D(input=X_3x3, filter_shape=[3, 3, 96, 384], strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', name='inception_5b_3x3_conv2')
    X_3x3 = BatchNormalization(input=X_3x3, shape=[384], name='inception_5b_3x3_bn2')
    X_3x3 = tf.nn.relu(features=X_3x3, name='inception_5b_3x3_relu2')

    X_pool = tf.nn.max_pool(value=input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC', name='inception_5b_pool_avg')
    X_pool = Conv2D(input=X_pool, filter_shape=[1, 1, 736, 96], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='inception_5b_pool_conv')
    X_pool = BatchNormalization(input=X_pool, shape=[96], name='inception_5b_pool_bn')
    X_pool = tf.nn.relu(features=X_pool, name='inception_5b_pool_relu')
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    X_pool = tf.pad(tensor=X_pool, paddings=paddings, mode='CONSTANT')
    inception_3b = tf.concat(values=[X_1x1, X_3x3, X_pool], axis=-1, name='inception_5b_concat')
    return inception_3b


class FaceNet:
    def __init__(self):
        print('This is the __init__ func of class FaceNet')
        tf.reset_default_graph()
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 96, 96, 3], name='InputX')

    def forward(self):
        conv1_generation = Conv2D(input=self.X, filter_shape=[7, 7, 3, 64], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC', name='conv1')
        # check point 2
        conv1_bn = BatchNormalization(input=conv1_generation, shape=[64], name='bn1')
        conv1_activation = tf.nn.relu(features=conv1_bn, name='conv1_bn_relu')

        maxpool_conv1 = tf.nn.max_pool(value=conv1_activation, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC', name='maxpool_conv1')

        conv2_generation = Conv2D(input=maxpool_conv1, filter_shape=[1, 1, 64, 64], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='conv2')
        conv2_bn = BatchNormalization(input=conv2_generation, shape=[64], name='bn2')
        conv2_activation = tf.nn.relu(features=conv2_bn, name='conv2_bn_relu')

        conv3_generation = Conv2D(input=conv2_activation, filter_shape=[3, 3, 64, 192], strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', name='conv3')
        conv3_bn = BatchNormalization(input=conv3_generation, shape=[192], name='bn3')
        conv3_activation = tf.nn.relu(features=conv3_bn, name='conv3_bn_relu')

        maxpool_conv3 = tf.nn.max_pool(value=conv3_activation, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC', name='maxpool_conv3')
        # check point 1 not equal with expect
        inception_1a = Inception_block_1a(maxpool_conv3)
        inception_1b = Inception_block_1b(inception_1a)
        inception_1c = Inception_block_1c(inception_1b)

        inception_2a = Inception_block_2a(inception_1c)
        inception_2b = Inception_block_2b(inception_2a)

        inception_3a = Inception_block_3a(inception_2b)
        inception_3b = Inception_block_3b(inception_3a)

        avgpool_inception = tf.nn.avg_pool(value=inception_3b, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name='avgpool_inception')
        flattened = tf.contrib.layers.flatten(avgpool_inception)
        fully_connected = tf.contrib.layers.fully_connected(inputs=flattened, num_outputs=128, activation_fn=None)
        face_embedding = tf.nn.l2_normalize(x=fully_connected, axis=-1, name='l2_normal')
        return conv1_generation
