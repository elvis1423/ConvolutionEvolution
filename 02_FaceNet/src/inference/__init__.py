import model
import data
import tensorflow as tf
import numpy as np


data = data.Data()
weights_dict = data.load_weights()
input_x = data.get_camera_face_for_test()
# input_x=np.array([[[[ 1.62434536e+00, -4.47128565e-01, -4.00878192e-01],
#          [-6.11756414e-01,  1.22450770e+00,  8.24005618e-01],
#          [-5.28171752e-01,  4.03491642e-01, -5.62305431e-01],
#          [-1.07296862e+00,  5.93578523e-01,  1.95487808e+00],
#          [ 8.65407629e-01, -1.09491185e+00, -1.33195167e+00],
#          [-2.30153870e+00,  1.69382433e-01, -1.76068856e+00],
#          [ 1.74481176e+00,  7.40556451e-01, -1.65072127e+00],
#          [-7.61206901e-01, -9.53700602e-01, -8.90555584e-01],
#          [ 3.19039096e-01, -2.66218506e-01, -1.11911540e+00],
#          [-2.49370375e-01,  3.26145467e-02,  1.95607890e+00]],
#
#         [[ 1.46210794e+00, -1.37311732e+00, -3.26499498e-01],
#          [-2.06014071e+00,  3.15159392e-01, -1.34267579e+00],
#          [-3.22417204e-01,  8.46160648e-01,  1.11438298e+00],
#          [-3.84054355e-01, -8.59515941e-01, -5.86523939e-01],
#          [ 1.13376944e+00,  3.50545979e-01, -1.23685338e+00],
#          [-1.09989127e+00, -1.31228341e+00,  8.75838928e-01],
#          [-1.72428208e-01, -3.86955093e-02,  6.23362177e-01],
#          [-8.77858418e-01, -1.61577235e+00, -4.34956683e-01],
#          [ 4.22137467e-02,  1.12141771e+00,  1.40754000e+00],
#          [ 5.82815214e-01,  4.08900538e-01,  1.29101580e-01]],
#
#         [[-1.10061918e+00, -2.46169559e-02,  1.61694960e+00],
#          [ 1.14472371e+00, -7.75161619e-01,  5.02740882e-01],
#          [ 9.01590721e-01,  1.27375593e+00,  1.55880554e+00],
#          [ 5.02494339e-01,  1.96710175e+00,  1.09402696e-01],
#          [ 9.00855949e-01, -1.85798186e+00, -1.21974440e+00],
#          [-6.83727859e-01,  1.23616403e+00,  2.44936865e+00],
#          [-1.22890226e-01,  1.62765075e+00, -5.45774168e-01],
#          [-9.35769434e-01,  3.38011697e-01, -1.98837863e-01],
#          [-2.67888080e-01, -1.19926803e+00, -7.00398505e-01],
#          [ 5.30355467e-01,  8.63345318e-01, -2.03394449e-01]],
#
#         [[-6.91660752e-01, -1.80920302e-01,  2.42669441e-01],
#          [-3.96753527e-01, -6.03920628e-01,  2.01830179e-01],
#          [-6.87172700e-01, -1.23005814e+00,  6.61020288e-01],
#          [-8.45205641e-01,  5.50537496e-01,  1.79215821e+00],
#          [-6.71246131e-01,  7.92806866e-01, -1.20464572e-01],
#          [-1.26645989e-02, -6.23530730e-01, -1.23312074e+00],
#          [-1.11731035e+00,  5.20576337e-01, -1.18231813e+00],
#          [ 2.34415698e-01, -1.14434139e+00, -6.65754518e-01],
#          [ 1.65980218e+00,  8.01861032e-01, -1.67419581e+00],
#          [ 7.42044161e-01,  4.65672984e-02,  8.25029824e-01]],
#
#         [[-1.91835552e-01, -1.86569772e-01, -4.98213564e-01],
#          [-8.87628964e-01, -1.01745873e-01, -3.10984978e-01],
#          [-7.47158294e-01,  8.68886157e-01, -1.89148284e-03],
#          [ 1.69245460e+00,  7.50411640e-01, -1.39662042e+00],
#          [ 5.08077548e-02,  5.29465324e-01, -8.61316361e-01],
#          [-6.36995647e-01,  1.37701210e-01,  6.74711526e-01],
#          [ 1.90915485e-01,  7.78211279e-02,  6.18539131e-01],
#          [ 2.10025514e+00,  6.18380262e-01, -4.43171931e-01],
#          [ 1.20158952e-01,  2.32494559e-01,  1.81053491e+00],
#          [ 6.17203110e-01,  6.82551407e-01, -1.30572692e+00]],
#
#         [[ 3.00170320e-01, -3.10116774e-01, -3.44987210e-01],
#          [-3.52249846e-01, -2.43483776e+00, -2.30839743e-01],
#          [-1.14251820e+00,  1.03882460e+00, -2.79308500e+00],
#          [-3.49342722e-01,  2.18697965e+00,  1.93752881e+00],
#          [-2.08894233e-01,  4.41364444e-01,  3.66332015e-01],
#          [ 5.86623191e-01, -1.00155233e-01, -1.04458938e+00],
#          [ 8.38983414e-01, -1.36444744e-01,  2.05117344e+00],
#          [ 9.31102081e-01, -1.19054188e-01,  5.85662000e-01],
#          [ 2.85587325e-01,  1.74094083e-02,  4.29526140e-01],
#          [ 8.85141164e-01, -1.12201873e+00, -6.06998398e-01]],
#
#         [[-7.54397941e-01, -5.17094458e-01,  1.06222724e-01],
#          [ 1.25286816e+00, -9.97026828e-01, -1.52568032e+00],
#          [ 5.12929820e-01,  2.48799161e-01,  7.95026094e-01],
#          [-2.98092835e-01, -2.96641152e-01, -3.74438319e-01],
#          [ 4.88518147e-01,  4.95211324e-01,  1.34048197e-01],
#          [-7.55717130e-02, -1.74703160e-01,  1.20205486e+00],
#          [ 1.13162939e+00,  9.86335188e-01,  2.84748111e-01],
#          [ 1.51981682e+00,  2.13533901e-01,  2.62467445e-01],
#          [ 2.18557541e+00,  2.19069973e+00,  2.76499305e-01],
#          [-1.39649634e+00, -1.89636092e+00, -7.33271604e-01]],
#
#         [[-1.44411381e+00, -6.46916688e-01,  8.36004719e-01],
#          [-5.04465863e-01,  9.01486892e-01,  1.54335911e+00],
#          [ 1.60037069e-01,  2.52832571e+00,  7.58805660e-01],
#          [ 8.76168921e-01, -2.48634778e-01,  8.84908814e-01],
#          [ 3.15634947e-01,  4.36689932e-02, -8.77281519e-01],
#          [-2.02220122e+00, -2.26314243e-01, -8.67787223e-01],
#          [-3.06204013e-01,  1.33145711e+00, -1.44087602e+00],
#          [ 8.27974643e-01, -2.87307863e-01,  1.23225307e+00],
#          [ 2.30094735e-01,  6.80069840e-01, -2.54179868e-01],
#          [ 7.62011180e-01, -3.19801599e-01,  1.39984394e+00]],
#
#         [[-2.22328143e-01, -1.27255876e+00, -7.81911683e-01],
#          [-2.00758069e-01,  3.13547720e-01, -4.37508983e-01],
#          [ 1.86561391e-01,  5.03184813e-01,  9.54250872e-02],
#          [ 4.10051647e-01,  1.29322588e+00,  9.21450069e-01],
#          [ 1.98299720e-01, -1.10447026e-01,  6.07501958e-02],
#          [ 1.19008646e-01, -6.17362064e-01,  2.11124755e-01],
#          [-6.70662286e-01,  5.62761097e-01,  1.65275673e-02],
#          [ 3.77563786e-01,  2.40737092e-01,  1.77187720e-01],
#          [ 1.21821271e-01,  2.80665077e-01, -1.11647002e+00],
#          [ 1.12948391e+00, -7.31127037e-02,  8.09271010e-02]],
#
#         [[ 1.19891788e+00,  1.16033857e+00, -1.86578994e-01],
#          [ 1.85156417e-01,  3.69492716e-01, -5.68244809e-02],
#          [-3.75284950e-01,  1.90465871e+00,  4.92336556e-01],
#          [-6.38730407e-01,  1.11105670e+00, -6.80678141e-01],
#          [ 4.23494354e-01,  6.59049796e-01, -8.45080274e-02],
#          [ 7.73400683e-02, -1.62743834e+00, -2.97361883e-01],
#          [-3.43853676e-01,  6.02319280e-01,  4.17302005e-01],
#          [ 4.35968568e-02,  4.20282204e-01,  7.84770651e-01],
#          [-6.20000844e-01,  8.10951673e-01, -9.55425262e-01],
#          [ 6.98032034e-01,  1.04444209e+00,  5.85910431e-01]]]])
younes = data.get_camera_face_by('../../images/younes.jpg')
benoit = data.get_camera_face_by('../../images/camera_2.jpg')

print(input_x)
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
        print('conv_w shape' + str(conv_w.shape))
        print('conv_b shape' + str(conv_b.shape))
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
with tf.Session(config=config) as sess:
    sess.run(init)
    anchor = sess.run(face_vector, feed_dict=feed_diction)
    # feed_diction[InputX] = younes
    positive = sess.run(face_vector, feed_dict=feed_diction)
    # feed_diction[InputX] = benoit
    negative = sess.run(face_vector, feed_dict=feed_diction)

    # for i in range(48):
    #     print(anchor[0, i, :, 0])
    # print(anchor[0,:,:,0])
    print(anchor)
    print('-------------------------------------------------')
    print(positive)
    print('-------------------------------------------------')
    print(negative)
    print('-------------------------------------------------')
    print('distance between anchor and positive' + str(np.linalg.norm(anchor-positive)))
    print('distance between anchor and negative' + str(np.linalg.norm(anchor-negative)))


