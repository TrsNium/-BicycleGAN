import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

#leaky relu
def leaky_relu(features, alpha=0.2, name=None):
    """Compute the Leaky ReLU activation function.
    "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
    AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
    http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
    Args:
        features: A `Tensor` representing preactivation values. Must be one of
            the following types: `float16`, `float32`, `float64`, `int32`, `int64`.
        alpha: Slope of the activation function at x < 0.
        name: A name for the operation (optional).
    Returns:
        The activation value
    """
    with ops.name_scope(name, "LeakyRelu", [features, alpha]):
        features = ops.convert_to_tensor(features, name="features")
        if features.dtype.is_integer:
            features = math_ops.to_float(features)
            alpha = ops.convert_to_tensor(alpha, dtype=features.dtype, name="alpha")
        return math_ops.maximum(alpha * features, features)

#Generator 
def gen(x, Z, layer_num=4, first_depth=64,reuse=False):
    with tf.variable_scope('g'):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        depth = first_depth
        concat = lambda a,b: tf.concat([a,b], -1)

        features = []
        c = tf.identity(x)
        for d in range(layer_num):
            z = z2img(Z, tf.shape(c))[1:2]
            c = convs(concat(z, c), depth, 2, 'e{}'.format(d))
            
            if d+1 != layer_num:
                features.append(c)
                c = tf.layers.max_pooling2d(c, [2,2], 2)
                depth *= 2

        depth = depth//2
        for d, f in enumerate(features[::-1]):
            upsampling = tf.layers.conv2d_transpose(c, depth, [2,2], (2,2), padding='same', activation=tf.nn.relu, name='{}_convt'.format(d))
            c = tf.concat([f, upsampling], -1)
            c = convs(c, depth, 2, 'dc{}'.format(d))
            depth = depth//2

        r = tf.layers.conv2d(c, 3, [3,3], padding='same', name='out_conv')
    return r

def convs(x, filter_num, n, name_space):
    with tf.variable_scope(name_space):
        c = tf.identity(x)
        for n_ in range(n):
            c = tf.layers.conv2d(c, filter_num, [3,3], padding='same', activation=tf.nn.relu, name='{}_conv'.format(n_))
    return c

#Discriminator
def dis(x, layer_num=4, reuse=False):
    with tf.variable_scope('d'):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        depth = 64
        h = leaky_relu(tf.layers.conv2d(x, depth, [4,4], (2,2), padding='same', name='conv0'))
        
        for n in range(layer_num):
            h = tf.layers.conv2d(h, depth, [4,4], (2,2), padding='same', name='conv{}'.format(n+1))
            h = tf.layers.batch_normalization(h, name='norm{}'.format(n+1))
            h = leaky_relu(h)
            depth *= 2

        h = tf.layers.conv2d(h, depth, [4,4], padding='same', name='conv{}'.format(layer_num+1))
        h = tf.layers.batch_normalization(h, name='norm{}'.format(layer_num+1))
        h = leaky_relu(h)

        r = tf.layers.conv2d(h, 1, [4,4], padding='same', name='conv_out')
    return tf.nn.sigmoid(r)

#encoder
def enc(x, z_dim, reuse=False):
    with tf.variable_scope('decoder'):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        shape = x.get_shape().as_list()
        x_ = tf.layers.conv2d(x, 32, [3,3], (2,2), padding='same', activation=tf.nn.relu, name='conv1')
        x_ = tf.layers.conv2d(x_, 64, [3,3], (2,2), padding='same', activation=tf.nn.relu, name='conv2')
        x_ = tf.layers.conv2d(x_, 128, [3,3], (2,2), padding='same', activation=tf.nn.relu, name='conv3')
        flatten = tf.reshape(x, (-1, shape[1]//8))
        mu = tf.layers.dense(flatten, z_dim, name='m_dense')
        logvar =  tf.layers.dense(flatten, z_dim, name='l_dense')
    return mu, logvar

def random_z(batch_size, z_dim):
    z = tf.random_normal([batch_size, z_dim], dtype=tf.float32)
    return z

def kl(mu, logvar):
    return tf.reduce_sum(mu**2 - tf.log(logvar + 1e-16) + 1 + logvar) * -0.5

def z2img(z, img_shape):
    z_dim = tf.shape(z)[-1]
    splited = tf.split(z, [1]*z_dim, axis=1)
    imgs = []
    for z_ in splited:
        tiled = tf.tile(z_, [1, img_shape[1]*img_shape[2]])
        img = tf.reshape(tiled, (-1, img_shape[1], img_shape[2], 1))
        imgs.append(img)
    return tf.concat(imgs, -1)
