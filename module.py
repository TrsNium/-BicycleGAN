import tensorflow as tf

#Generator 
def gen(x, Z, reuse=False):
    with tf.variable_scope('g'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        ec1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(x, 32, [3,3], (1,1), padding='same', name='ec1'), name='eb1'))
        ec = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(ec1, 64, [4,4], (2,2), padding='same', name='ec2'), name='eb2'))
        ec2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(ec, 64, [3,3], (1,1), padding='same', name='ec3'), name='eb3'))
        ec = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(ec2, 128, [4,4], (2,2), padding='same', name='ec4'), name='eb4'))
        ec3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(ec, 128, [3,3], (1,1), padding='same', name='ec5'), name='eb5'))
                
        c = lambda a,b: tf.concat([a,b], -1)
                
        dx = c(ec3, ec)
        dct = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(dx, 128, [4,4], (2,2), padding='same', name='dct1'), name='db1'))
        dc = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(dct, 64, [3,3], (1,1), padding='same', name='dc1'), name='db2'))
        dct = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(c(dc, ec2), 64, [4,4], (2,2), padding='same', name='dct2'), name='db3'))
        dc = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(dct, 32, [3,3], (1,1), padding='same', name='dc2'), name='db4'))
        r = tf.layers.conv2d(c(ec1, dc), 3, [3,3], (1,1), padding='same', name='dc3')
    return r
        
#Discriminator
def dis(x, reuse=False):
    with tf.variable_scope('d'):
        if reuse:
            tf.get_variable_scope().reuse_variables()
                    
        x_ = tf.layers.conv2d(x, 32, [3,3], strides=(2,2), activation=tf.nn.relu, name='conv1')
        x_ = tf.layers.batch_normalization(tf.layers.conv2d(x_, 64, [3,3], strides=(2,2), activation=tf.nn.relu, name='conv2'), name='b1')
        x_ = tf.layers.batch_normalization(tf.layers.conv2d(x_, 128, [3,3], strides=(2,2), activation=tf.nn.relu, name='conv3'), name='b2')
        r = tf.layers.conv2d(x_, 1, [2,2], strides=(2,2), name='conv4', activation=tf.nn.sigmoid)
    return r

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
