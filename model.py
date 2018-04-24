import tensorflow as tf
from module import gen, dis, dec
from util import *

class BicycleGAN():
    
    def __init__(self, args):
        z_dim = args['z_dim']
        im_size = args['im_size']
        channel_num = args['channel_num']
        
        self.A = tf.placeholder(tf.float32, [None, im_size, im_size, channel_num], name='A')
        self.B = tf.placeholder(tf.float32, [None, im_size, im_size, channel_num], name='B')
        self.Z = tf.placeholder(tf.float32, [None, z_dim], name='Z')

        # Fake Target B
        self.B_ = gen(self.A, self.Z)
        #print(self.B_.get_shape().as_list())

        # Estimate laten z
        z = dec(self.B_)

        # Discriminator outs
        dis_real = dis(tf.concat([self.A, self.B], -1), False)
        dis_fake = dis(tf.concat([self.A, self.B_], -1), True)

        #Losses
        laten = tf.abs(self.Z - z)
        
        

args = {'z_dim': 128, 'im_size':128, 'channel_num':3}
BicycleGAN(args)
