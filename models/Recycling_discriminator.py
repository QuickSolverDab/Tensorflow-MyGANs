#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from ops import *
from utils import *

class Recycling_discriminator(object):
    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, patch_depth, patch_size, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = 'celebA';
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.model_name = "Recycling_discriminator"     # name for checkpoint
        self.epoch = epoch
        """hyper_params"""
        self.n = patch_depth
        self.patch = patch_size
        self.z_dim = z_dim
        self.ge_iter = 1        #the num of generator updates per discriminator update
        self.noise_dist = 'Uniform'
        """Vis_parameters"""
        self.vis_num = 2
        self.sample_num = 36

        if dataset_name == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.output_height = 64
            self.output_width = 64
            self.c_dim = 3

            self.image_dims = [self.input_height, self.input_width, self.c_dim]
            self.latent_dim = self.n * self.patch**2
            self.z_dim = self.z_dim

            # initial depth
            self.depth = 64

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # Recycling_discriminator
            self.simul_co = True

            # load data list
            train_dir = # put train data directory here
            test_dir  = # put train data directory here
            self.train_list = glob(train_dir+"*.jpg")
            self.test_list  = glob(test_dir+"*.jpg")
            self.train_list = self.train_list
            self.num_batches = len(self.train_list) // self.batch_size
        else:
            raise NotImplementedError

    def discriminator(self, x, is_training=True, reuse=False, name='discriminator'):
        with tf.variable_scope(name, reuse=reuse):
            with tf.variable_scope(name+'_Org', reuse=reuse):
                net  = lrelu(   conv2d(x,    self.depth*1,  5, 5, 2, 2,  name='disc_conv1')) # 16*16*64
                net  = lrelu(bn(conv2d(net,  self.depth*2,  5, 5, 2, 2,  name='disc_conv2'), is_training=is_training, scope='disc_bn2')) #  8* 8*128
                net  = lrelu(bn(conv2d(net,  self.depth*4,  5, 5, 2, 2,  name='disc_conv3'), is_training=is_training, scope='disc_bn3')) #  4* 4*256
                net  = lrelu(bn(conv2d(net,  self.depth*8,  5, 5, 2, 2,  name='disc_conv4'), is_training=is_training, scope='disc_bn4')) #  4* 4*256
                ft   = tf.reshape(net,[self.batch_size,-1])
                log  = linear(ft,1,scope='disc_out_logit' )
        return log, ft

    def connector(self, z, is_training=True, reuse=False, name="connector"):
        with tf.variable_scope(name, reuse=reuse):
            z   = tf.reshape(z,[self.batch_size,-1])
            net = lrelu(linear(z,      1024, scope='conn_fc0'))
            net = lrelu(bn(linear(net, 1024, scope='conn_fc1'),is_training=is_training, scope='conn_fcbn1'))
            net = lrelu(bn(linear(net, 1024, scope='conn_fc2'),is_training=is_training, scope='conn_fcbn2'))
            out =       linear(net, self.z_dim, scope='conn_fc3')
        return out

    def generator(self, noise, is_training=True, reuse=False, name='generator'):
        with tf.variable_scope(name, reuse=reuse):
            net   = linear(noise, self.patch*self.patch*self.depth*16, scope='ge_fc_0')
            net   = tf.nn.relu(bn(tf.reshape(net,  [self.batch_size, self.patch, self.patch, self.depth*16]),            is_training=is_training, scope='ge_bn0')); # 4* 4*1024
            net   = tf.nn.relu(bn(deconv2d(net,    [self.batch_size, 8,  8 ,  self.depth*8], 5, 5, 2, 2, name='ge_dc1'), is_training=is_training, scope='ge_bn1')); # 8* 8* 512
            net   = tf.nn.relu(bn(deconv2d(net,    [self.batch_size, 16, 16,  self.depth*4], 5, 5, 2, 2, name='ge_dc2'), is_training=is_training, scope='ge_bn2')); #16*16* 256
            net   = tf.nn.relu(bn(deconv2d(net,    [self.batch_size, 32, 32,  self.depth*2], 5, 5, 2, 2, name='ge_dc3'), is_training=is_training, scope='ge_bn3')); #32*32* 128
            out   =               deconv2d(net,    [self.batch_size, 64, 64,    self.c_dim], 5, 5, 2, 2, name='ge_dc4')  #64*64*3
            out   = tf.tanh(out)
        return out

    def build_model(self):
        # some parameters
        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='real_images')
        # labels
        self.noise = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='noise')

        """ Network_in/out """
        # 1. Generator
        Ge_image = self.generator(self.noise, is_training=True, reuse=False, name='generator')
        # 2. Discriminator
        Dx_real_logits, ft_real  = self.discriminator(self.inputs, is_training=True, reuse=False, name='discriminator')
        Dx_fake_logits, ft_fake  = self.discriminator(Ge_image,    is_training=True, reuse=True , name='discriminator')
        # 3. Connector
        no_fake      = self.connector(ft_fake, is_training=True,  reuse=False, name='connector')

        """ Loss_compute """
        # Discriminator Loss of X
        dx_real  = binary_cross_entropy( Dx_real_logits,  tf.ones_like( Dx_real_logits))
        dx_fake  = binary_cross_entropy( Dx_fake_logits,  tf.zeros_like(Dx_fake_logits))
        self.dx_loss  = dx_real + dx_fake
        # Generator Loss
        self.ge_loss = binary_cross_entropy( Dx_fake_logits, tf.ones_like(Dx_fake_logits))
        self.co_loss = mse_loss(no_fake, self.noise)

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        ge_vars = [var for var in t_vars if 'generator' in var.name]
        dx_vars = [var for var in t_vars if 'discriminator' in var.name]
        co_vars = [var for var in t_vars if 'connector' in var.name]
        self.init_co_vars = tf.initialize_variables(co_vars)

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.ge_optm = tf.train.AdamOptimizer(self.learning_rate , beta1=self.beta1).minimize(self.ge_loss, var_list= ge_vars)
            self.dx_optm = tf.train.AdamOptimizer(self.learning_rate , beta1=self.beta1).minimize(self.dx_loss, var_list= dx_vars)
            self.co_optm = tf.train.AdamOptimizer(self.learning_rate , beta1=self.beta1).minimize(self.co_loss, var_list= co_vars)

        """" Testing """
        # for test
        self.fake_images  = self.generator(self.noise, is_training=False, reuse=True, name='generator')
        # Recon_by_connection
        _ft_real      = self.discriminator(self.inputs, is_training=False, reuse=True, name='discriminator')[1]
        self.no_real  = self.connector(_ft_real,    is_training=False, reuse=True, name='connector')
        self.recon_img = self.generator(self.no_real, is_training=False, reuse=True, name='generator')

        """ Summary """
        # Summary
        tf.summary.scalar("CelebA_dx_loss", self.dx_loss)
        tf.summary.scalar("CelebA_ge_loss", self.ge_loss)
        tf.summary.scalar("CelebA_co_loss", self.co_loss)
        self.summary_op = tf.summary.merge_all()

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()
        # saver to save model
        self.saver = tf.train.Saver()
        # summary writer
        name = "{}_init_depth_{}".format(self.model_name, self.depth)
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name + '/' + name, self.sess.graph)
        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] CKPT Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] CKPT Load failed...")

        """Train"""
        # loop for epoch
        start_time = time.time()
        out_train_list = self.train_list

        for epoch in range(start_epoch, self.epoch):
            random.shuffle(self.train_list); random.shuffle(self.test_list)
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                if (idx+1)*self.batch_size <= len(self.train_list):
                    batch_images = load_CelebA(self.train_list[idx*self.batch_size:(idx+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )[0]
                else:
                    pass
                # Prioir Distribution and Random Noise Distribution
                if self.noise_dist == 'Uniform':
                    batch_noise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                elif self.noise_dist == 'Normal':
                    batch_noise = np.random.normal(0.0,1.0, [self.batch_size, self.z_dim])

                "FeedDict"
                _feed_dict = {self.inputs: batch_images, self.noise: batch_noise}
                ## Training Part
                # update D network
                _ = self.sess.run(self.dx_optm, feed_dict=_feed_dict )
                # update C network
                if self.simul_co == True:
                    _ = self.sess.run(self.co_optm, feed_dict=_feed_dict )
                # update G network
                for gen_itr in range(self.ge_iter):
                    _ = self.sess.run(self.ge_optm, feed_dict=_feed_dict )

                # display training status
                if np.mod(idx, 50) == 0 or idx+1 == self.num_batches:
                    if self.simul_co == True:
                        _Dx,_Ge,_Co = self.sess.run([self.dx_loss, self.ge_loss, self.co_loss], feed_dict=_feed_dict)
                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, Loss:[Discriminator: %.4f, Generator:%.4f, Connector: %.4f]]"
                                % (epoch,idx,self.num_batches,time.time()-start_time,  _Dx, _Ge, _Co) )
                    else:
                        _Dx, _Ge = self.sess.run([self.dx_loss, self.ge_loss], feed_dict=_feed_dict)
                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, Loss:[Discriminator: %.4f, Generator:%.4f]"
                                % (epoch,idx,self.num_batches,time.time()-start_time,  _Dx, _Ge) )

                """Summary"""
                if np.mod(idx+1, np.ceil(self.num_batches/10)) == 0 or idx+1 == self.num_batches:
                    summary = self.sess.run(self.summary_op, feed_dict=_feed_dict )
                    self.writer.add_summary(summary,counter)
                    self.writer.flush()

                """Image SAVE"""
                # save training results for every 300 steps
                if np.mod(idx+1, np.ceil(self.num_batches/self.vis_num)) == 0 or idx+1 == self.num_batches:
                    test_ssim  = []; test_psnr  = []
                    train_ssim = []; train_psnr = []
                    test_ssim_naive  = []; test_psnr_naive  = []
                    train_ssim_naive = []; train_psnr_naive = []
                    for batch_idx in range(3):
                        random.shuffle(self.test_list)
                        if self.noise_dist == 'Uniform':
                            batch_noise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                        elif self.noise_dist == 'Normal':
                            batch_noise = np.random.normal(0.0,1.0, [self.batch_size, self.z_dim])
                        # Test_image
                        test_images  = load_CelebA(self.test_list[0*self.batch_size:(0+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )[0]
                        "FeedDict"
                        _feed_dict    = {self.inputs: test_images,  self.noise: batch_noise}
                        samples_rand = self.sess.run(self.fake_images, feed_dict=_feed_dict)

                        # save setting
                        tot_num_samples = min(self.sample_num, self.batch_size)
                        manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                        manifold_w = int(np.floor(np.sqrt(tot_num_samples)))

                        # Save images
                        save_images(samples_rand[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                    './' + check_folder(self.result_dir + '/' + self.model_dir + '/' + '{:02d}'.format(epoch))+ '/' +  '{:03d}_{:02d}_GE_samples_v36.png'.format(idx+1, batch_idx))
                        if self.simul_co:
                            recon_samp  = self.sess.run(self.recon_img, feed_dict=_feed_dict)
                            test_recon  = np.zeros_like(recon_samp)
                            for col in range(int(self.batch_size/2)):
                                col = col*2
                            test_recon[ col   ,...]  =  test_images[col,...]
                            test_recon[(col+1),...]  =  recon_samp[col,...]
                            save_images(test_recon[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                        './' + check_folder(self.result_dir + '/' + self.model_dir + '/' + '{:02d}'.format(epoch))+ '/' +  '{:03d}_{:02d}_GE_samples_v36.png'.format(idx+1, batch_idx))
                counter += 1
                start_batch_id=0
            self.save(self.checkpoint_dir, counter)

        """Reset"""
        _ = self.sess.run(self.init_co_vars)
        tot_num_samples = min(self.sample_num, self.batch_size)
        manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
        manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
        random.shuffle(self.test_list); random.shuffle(self.train_list)
        test_images  = load_CelebA(self.test_list[1*self.batch_size:(1+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )[0]
        train_images = load_CelebA(self.train_list[1*self.batch_size:(1+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )[0]

        for co_iter in range(100000):
            if np.mod(co_iter+1,1000) == 0:
                _Co = self.sess.run(self.co_loss, feed_dict=_feed_dict)
                print("iter: [%2d], Loss:[Connection_loss:%.4f]" % (co_iter+1, _Co) )

            elif np.mod(co_iter+1,2000) == 0:
                for num in range(5):
                    random.shuffle(self.test_list)
                    test_images = load_CelebA(self.test_list[1*self.batch_size:(1+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )[0]
                    _feed_dict  = {self.inputs: test_images, self.noise: batch_noise}
                    recon_samp  = self.sess.run(self.recon_img, feed_dict=_feed_dict)
                    test_recon  = np.zeros_like(recon_samp)
                    for col in range(int(self.batch_size/2)):
                        col = col*2
                        test_recon[ col   ,...]  =  test_images[col,...]
                        test_recon[(col+1),...]  =  recon_samp[col,...]
                    # Recon_64
                    save_images(test_recon[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir + '/' + 'Reset') + '/' + '{:02d}_{:02d}_recon_compare.png'.format(co_iter+1, num))
            batch_noise  = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
            _feed_dict  = {self.inputs: train_images, self.noise: batch_noise}
            _ = self.sess.run(self.co_optm, feed_dict=_feed_dict)

    @property
    def model_dir(self):
        return "{}_init_depth_{}".format(self.model_name, self.depth)


    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
