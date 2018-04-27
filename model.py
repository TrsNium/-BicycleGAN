import tensorflow as tf
from module import gen, dis, enc, random_z, kl, z2img
from util import *

class BicycleGAN():
    
    def __init__(self, args):
        z_dim = args['z_dim']
        im_size = args['im_size']
        achannel_num = args['channel_num']
        channel_num = args['channel_num']
        lambda_kl = args['lambda_kl']
        lambda_z = args['lambda_z']
        lambda_l1 = args['lambda_l1']
        layer_num = args['layer_num']
        first_depth = args['first_depth']

        self.A = tf.placeholder(tf.float32, [None, im_size, im_size, achannel_num], name='A')
        self.B = tf.placeholder(tf.float32, [None, im_size, im_size, channel_num], name='B')
        batch_size = tf.shape(self.A)[0] 
        
        half_size = batch_size//2
        self.A_encoded = self.A[:half_size]
        self.A_random = self.A[half_size:]
        self.B_encoded = self.B[:half_size]
        self.B_random = self.B[half_size:]
        
    
        #################
        #Part Of cLR-GAN#
        #################
        
        # Fake Target B
        r_size = tf.shape(self.A_random)[0]
        z = random_z(r_size, z_dim)
        self.B_ = gen(self.A_random, z, layer_num, first_depth, channel_num, False)
        #print(self.B_.get_shape().as_list())


        # Estimate laten z
        mu, logvar = enc(self.B_, z_dim, reuse=False)

        # Discriminator outs
        dis_real = dis(tf.concat([self.A_random, self.B_random], -1), reuse=False)
        dis_fake = dis(tf.concat([self.A_random, self.B_], -1), reuse=True)

        # Losses
        laten = tf.reduce_mean(tf.abs(mu - z)) * lambda_z
        clr_g_loss = tf.reduce_mean(tf.ones_like(dis_fake) - dis_fake)
        clr_d_loss = tf.reduce_mean(tf.ones_like(dis_real) - dis_real) + tf.reduce_mean(dis_fake)
        
        ##################
        #Part Of cVAE-GAN#
        ##################

        # Estimate laten z
        mu, logvar = enc(self.B_encoded, z_dim, reuse=True)
        std = tf.log(logvar*.5 + 1e-16)
        eps = random_z(half_size, z_dim) 
        z = eps*(std) + mu

        # Fake Trarget B
        B_ = gen(self.A_encoded, z, layer_num, first_depth, channel_num, True)

        # Discriminator outs
        dis_real = dis(tf.concat([self.A_encoded, self.B_encoded], -1), reuse=True)
        dis_fake = dis(tf.concat([self.A_encoded, B_], -1), reuse=True)

        # Losses
        l1 = tf.reduce_mean(tf.abs(self.B_encoded - B_)) * lambda_l1
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

    def fit(self, epochs, generator, lr, log_interval=1000):
        optim_g = tf.train.AdamOptimizer(lr).minimize(self.g_loss, var_list=self.g_var)
        optim_e = tf.train.AdamOptimizer(lr).minimize(self.e_loss, var_list=self.e_var)
        optim_d = tf.train.AdamOptimizer(lr).minimize(self.d_loss, var_list=self.d_var)

        self.sess.run(tf.global_variables_initializer())
        for i, (epoch, x, y) in enumerate(generator()):
            fedd_dict = {
                self.A: x,
                self.B: y
            }
            
            g_loss,_ = self.sess.run([self.g_loss, optim_g], feed_dict=feed_dict)
            d_loss,_ = self.sess.run([self.d_loss, optim_d], feed_dict=feed_dict)
            _ = self.sess.run(optim_e, feed_dict=feed_dict)
            
            if i % log_interval ==0:
                print('epoch:', epoch, '\titeration:' , i, '\tg_loss:', g_loss, '\td_loss', d_loss)

            if epoch == epochs:
                print('*******finished training!*******') 
                return

    def pridict(self, x):
        pass

    def save(self, save_path):
        saver = tf.train.Saver(tf.global_variables())
        saver.save(self.sess, save_path)        

    def restore(self, save_path):
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, save_path)


if __name__ == '__main__':
    x, y = load_data('./data/edges2shoes/')
    gen = generator(32, x, y)
    
    args = {'z_dim': 16, 'im_size':256, 'achannel_num':1, 'channel_num':3, 'lambda_kl':1, 'lambda_z':1, 'lambda_l1':1, 'layer_num':4, 'first_depth':64}
    model_ = BicycleGAN(args)
    model_.fit(200, gen, .0002, )
