import tensorflow as tf
from module import gen, dis, enc, random_z, kl, z2img
from util import *

class BicycleGAN():
    
    def __init__(self, args):
        z_dim = args['z_dim']
        im_size = args['im_size']
        channel_num = args['channel_num']
        lambda_kl = args['lambda_kl']
        lambda_z = args['lambda_z']
        lambda_l1 = args['lambda_l1']

        self.A = tf.placeholder(tf.float32, [None, im_size, im_size, channel_num], name='A')
        self.B = tf.placeholder(tf.float32, [None, im_size, im_size, channel_num], name='B')
        batch_size = tf.shape(self.A)[0] 
        
        half_size = batch_size//2
        self.A_encoded = self.A[:half_size]
        self.A_random = self.A[half_size:]
        self.B_encoded = self.B[:half_size]
        self.B_random = self.B[half_size]
        
        z2img(random_z(batch_size, z_dim), tf.shape(self.A))

        #################
        #Part Of cLR-GAN#
        #################
        
        # Fake Target B
        r_size = tf.shape(self.A_random)[0]
        z = random_z(r_size, z_dim)
        self.B_ = gen(self.A, z, False)
        #print(self.B_.get_shape().as_list())


        # Estimate laten z
        mu, logvar = enc(self.B_, z_dim, False)

        # Discriminator outs
        dis_real = dis(tf.concat([self.A, self.B], -1), False)
        dis_fake = dis(tf.concat([self.A, self.B_], -1), True)

        # Losses
        laten = tf.reduce_mean(tf.abs(mu - z)) * lambda_z
        clr_g_loss = tf.reduce_mean(tf.ones_like(dis_fake) - dis_fake)
        clr_d_loss = tf.reduce_mean(tf.ones_like(dis_real) - dis_real) + tf.reduce_mean(dis_fake)
        
        ##################
        #Part Of cVAE-GAN#
        ##################

        # Estimate laten z
        mu, logvar = enc(self.B, z_dim, True)
        std = tf.log(logvar*.5 + 1e-16)
        eps = random_z(half_size, z_dim) 
        z = eps*(std) + mu

        # Fake Trarget B
        B_ = gen(self.A, z, True)

        # Discriminator outs
        dis_real = dis(tf.concat([self.A, self.B_], -1), True)
        dis_fake = dis(tf.concat([self.A, B_], -1), True)

        # Losses
        l1 = tf.reduce_mean(tf.abs(self.B - B_)) * lambda_l1
        kl_ = kl(mu, logvar) * lambda_kl
        vae_g_loss = tf.reduce_mean(tf.ones_like(dis_fake) - dis_fake) + l1 + kl_ 
        vae_d_loss = tf.reduce_mean(tf.ones_like(dis_real) - dis_real) + tf.reduce_mean(dis_fake)
           
        ##########
        # Hybrid #
        ##########

        self.e_loss = clr_g_loss + vae_g_loss
        self.d_loss = clr_d_loss + vae_d_loss
        self.g_loss = self.e_loss + laten

        trainable_vars = tf.trainable_variables()
        self.g_var = [var for var in trainable_vars if 'g' in var.name]
        self.d_var = [var for var in trainable_vars if 'd' in var.name]
        self.e_var = [var for var in trainable_vars if 'e' in var.name]

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list="0",
                allow_growth=True
            )
        )
        self.sess = tf.Session(config=config)    

    def fit(self, lr, iteration):
        optim_g = tf.train.AdamOptimizer(lr).minimize(self.g_loss, var_list=self.g_var)
        optim_e = tf.train.AdamOptimizer(lr).minimize(self.e_loss, var_list=self.e_var)
        optim_d = tf.train.AdamOptimizer(lr).minimize(self.d_loss, var_list=self.d_var)

        


args = {'z_dim': 128, 'im_size':128, 'channel_num':3, 'lambda_kl':1, 'lambda_z':1, 'lambda_l1':1}
BicycleGAN(args)
