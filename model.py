import tensorflow as tf
from module import gen, dis, dec
from util import *

def BicycleGAN():
    
    def __init__(self, args):
        z_dim = args['z_dim']
        im_size = args['im_size']
        channel_num = args['channel_num']
        
        self.A = tf.placeholder(tf.flaot32, [None, im_size, im_size, channel_num], name='A')
        self.B = tf.placeholder(tf.float32, [None, im_size, im_size, channel_num, name='B')
        self.Z = tf.placeholder(tf.float32, [None, z_dim], name='Z')

        self.B_ = gen(self.A, z)


