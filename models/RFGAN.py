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

class RFGAN(object):
    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, patch_depth, patch_size, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = 'celebA';
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.model_name = "RFGAN"     # name for checkpoint
        self.epoch = epoch
        """hyper_params"""
        self.n = patch_depth
        self.patch = patch_size
        self.z_dim = z_dim
        self.noise_dist = 'Uniform'
        """Vis_parameters"""
        self.vis_num = 2
        self.sample_num = 64  # number of generated images to be saved

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
            """ RFGAN hyper_params """
            self.pre_epoch = 10
            self.ge_iter = 2 #the num of generator updates per discriminator update

            # load data list
            train_dir = "./../../dataset/CelebA/splits/train/"
            test_dir  = "./../../dataset/CelebA/splits/test/"
            self.train_list = glob(train_dir+"*.jpg")
            self.test_list  = glob(test_dir+"*.jpg")
            self.train_list = self.train_list
            self.num_batches = len(self.train_list) // self.batch_size
        else:
            raise NotImplementedError

    def discriminator(self, x, encoder_layers, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            net  = lrelu(   conv2d(x,    self.depth*1,  5, 5, 2, 2,  name='disc_conv1')) # 16*16*64
            net  = lrelu(bn(conv2d(net,  self.depth*2,  5, 5, 2, 2,  name='disc_conv2'), is_training=is_training, scope='disc_bn2')) #  8* 8*128
            net  = lrelu(bn(conv2d(net,  self.depth*4,  5, 5, 2, 2,  name='disc_conv3'), is_training=is_training, scope='disc_bn3')) #  4* 4*256
            net  = lrelu(bn(conv2d(net,  self.depth*8,  5, 5, 2, 2,  name='disc_conv4'), is_training=is_training, scope='disc_bn4')) #  4* 4*256
            net = tf.concat([net,encoder_layers],axis=3)
            feature = tf.reshape(net,[self.batch_size,-1])
            out_logit = linear(feature,1,scope='disc_out_logit' )
            out = tf.sigmoid(out_logit)
        return out, out_logit

    def encoder(self, x, is_training=True, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):
            net = lrelu(   conv2d(x,   self.depth*1,  5, 5, 2, 2, name='en_conv1'))
            net = lrelu(bn(conv2d(net, self.depth*2,  5, 5, 2, 2, name='en_conv2'),  is_training=is_training, scope='en_bn2'))
            net = lrelu(bn(conv2d(net, self.depth*4,  5, 5, 2, 2, name='en_conv3'),  is_training=is_training, scope='en_bn3'))
            net = lrelu(bn(conv2d(net, self.depth*8,  5, 5, 2, 2, name='en_conv4'),  is_training=is_training, scope='en_bn4'))
            encoder_layers = net
            net =          conv2d(net, self.n, 1, 1, 1, 1, name='en_conv5');
            z   = tf.reshape(net, [self.batch_size, -1]);
        return z, encoder_layers

    def decoder(self, z, is_training=True, reuse=False, name='decoder'):
        with tf.variable_scope(name, reuse=reuse):
            z = tf.reshape(z, [self.batch_size, self.patch, self.patch, self.n]);
            z = tf.nn.relu(bn(conv2d(z, self.depth*4, 1, 1, 1, 1, name='de_dc0'), is_training=is_training, scope='de_bn0'));
            net = tf.nn.relu(bn(deconv2d(z,     [self.batch_size, 8,  8,  self.depth*4], 5, 5, 2, 2, name='de_dc1'), is_training=is_training, scope='de_bn1'));
            net = tf.nn.relu(bn(deconv2d(net,   [self.batch_size, 16, 16, self.depth*2], 5, 5, 2, 2, name='de_dc2'), is_training=is_training, scope='de_bn2'));
            net = tf.nn.relu(bn(deconv2d(net,   [self.batch_size, 32, 32, self.depth*1], 5, 5, 2, 2, name='de_dc3'), is_training=is_training, scope='de_bn3'));
            out =               deconv2d(net,   [self.batch_size, 64, 64,   self.c_dim], 5, 5, 2, 2, name='de_dc4');
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
        bs = self.batch_size
        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + self.image_dims, name='real_images')
        self.noisy_inputs = tf.placeholder(tf.float32, [bs] + self.image_dims, name='noisy_inputs')
        # labels
        self.noise = tf.placeholder(tf.float32, [bs, self.z_dim], name='noise')

        """ Network_in/out """
        ### 1. Autoencoder
        En, en_layer    = self.encoder(self.noisy_inputs, is_training=True, reuse=False)
        De              = self.decoder(En, is_training=True, reuse=False)
        ### 2 Generator
        Ge_image = self.generator(self.noise, is_training=True, reuse=False)

        ### 3. Discriminator
        en_layers_real = self.encoder(self.inputs, is_training=False, reuse=True)[1]
        en_layers_fake = self.encoder(Ge_image,    is_training=False, reuse=True)[1]
        Dr_real, Dr_real_logits = self.discriminator(self.inputs, en_layers_real, is_training=True, reuse=False)
        Dr_fake, Dr_fake_logits = self.discriminator(Ge_image,    en_layers_fake, is_training=True, reuse=True)

        """Loss_compute"""
        ### AE
        self.ae_loss  = l1_loss(De, self.inputs)
        ### Disc
        dr_loss_real  = binary_cross_entropy( Dr_real_logits,  tf.ones_like(Dr_real_logits))
        dr_loss_fake  = binary_cross_entropy( Dr_fake_logits,  tf.zeros_like(Dr_fake_logits))
        ## Ge
        self.gan_loss = binary_cross_entropy(Dr_fake_logits,   tf.ones_like(Dr_fake_logits))

        """Loss"""
        # get loss for discriminator
        self.dr_loss = dr_loss_real + dr_loss_fake
        # get loss for generator
        self.ge_loss = self.gan_loss

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        en_vars = [var for var in t_vars if 'encoder' in var.name]
        de_vars = [var for var in t_vars if 'decoder' in var.name]
        ge_vars = [var for var in t_vars if 'generator' in var.name]
        dr_vars = [var for var in t_vars if 'discriminator' in var.name]
        ae_vars = en_vars + de_vars

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.ae_optm = tf.train.AdamOptimizer(self.learning_rate,   beta1=self.beta1).minimize(self.ae_loss, var_list=ae_vars)
            self.ge_optm = tf.train.AdamOptimizer(self.learning_rate,   beta1=self.beta1).minimize(self.ge_loss, var_list=ge_vars)
            self.dr_optm = tf.train.AdamOptimizer(self.learning_rate,   beta1=self.beta1).minimize(self.dr_loss, var_list=dr_vars)

        """" Testing """
        # for test
        En_test          = self.encoder(self.inputs, is_training=False, reuse=True)[0]
        self.De_test     = self.decoder(En_test, is_training=False, reuse=True)
        self.fake_images = self.generator(self.noise, is_training=False, reuse=True)

        En_test_fake     = self.encoder(self.fake_images, is_training=False, reuse=True)[0]
        self.De_test_fake= self.decoder(En_test_fake, is_training=False, reuse=True)

        """ Summary """
        tf.summary.scalar("dr_loss", self.dr_loss)
        tf.summary.scalar("ge_loss", self.ge_loss)
        tf.summary.scalar("ae_loss", self.ae_loss)
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

            """PreTrain"""
            print('Pre-train start')
            start_time = time.time()
            for pre_epoch in range(self.pre_epoch):
                random.shuffle(self.train_list); random.shuffle(self.test_list)
                for pre_idx in range(0, self.num_batches):
                    if (pre_idx+1)*self.batch_size <= len(self.train_list):
                        batch_images, noisy_batch = load_CelebA(self.train_list[pre_idx*self.batch_size:(pre_idx+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )
                    else:
                        pass
                    # Prioir Distribution and Random Noise Distribution
                    if self.noise_dist == 'Uniform':
                        batch_noise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                    elif self.noise_dist == 'Normal':
                        batch_noise = np.random.normal(0.0,1.0, [self.batch_size, self.z_dim])
                    _feed_dict = {self.inputs: batch_images, self.noisy_inputs: noisy_batch, self.noise: batch_noise}
                    _ = self.sess.run(self.ae_optm, feed_dict=_feed_dict)
                    if np.mod(pre_idx, 10) == 0 or pre_idx == self.num_batches:
                        _AE = self.sess.run(self.ae_loss, feed_dict={self.inputs: batch_images, self.noisy_inputs: noisy_batch, self.noise: batch_noise})
                        print("Pre_Epoch: [%2d] [%4d/%4d] time: %4.4f, AE Loss: %.4f" \
                                % (pre_epoch,pre_idx,self.num_batches,time.time()-start_time,  _AE ))
            print('Pre-train is done')
            self.save(self.checkpoint_dir, 0)

        # loop for epoch
        start_time = time.time()
        # cnt_img = 0
        for epoch in range(start_epoch, self.epoch):
            random.shuffle(self.train_list); random.shuffle(self.test_list)
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                if (idx+1)*self.batch_size <= len(self.train_list):
                    batch_images, noisy_batch = load_CelebA(self.train_list[idx*self.batch_size:(idx+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )
                else:
                    pass

                # Prioir Distribution and Random Noise Distribution
                if self.noise_dist == 'Uniform':
                    batch_noise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                elif self.noise_dist == 'Normal':
                    batch_noise = np.random.normal(0.0,1.0, [self.batch_size, self.z_dim])
                "FeedDict"
                _feed_dict = {self.inputs: batch_images, self.noisy_inputs: noisy_batch, self.noise: batch_noise}
                # update Discriminator network
                _ = self.sess.run(self.dr_optm, feed_dict=_feed_dict )
                # update Generator network
                for x in range(self.ge_iter):
                    _ = self.sess.run(self.ge_optm, feed_dict=_feed_dict )


                # display training status
                if np.mod(idx, 50) == 0 or idx == self.num_batches:
                    _AE,_DR,_GE,_gan = self.sess.run([self.ae_loss,self.dr_loss,self.ge_loss,self.gan_loss], feed_dict=_feed_dict)
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, Loss:[AE:%.4f, DR:%.4f, GE:%.4f]" % (epoch,idx,self.num_batches,time.time()-start_time,  _AE,  _DR,  _GE ))

                """Summary"""
                if np.mod(idx+1, np.ceil(self.num_batches/10)) == 0 or idx+1 == self.num_batches:
                    summary = self.sess.run(self.summary_op, feed_dict=_feed_dict )
                    self.writer.add_summary(summary,counter)
                    self.writer.flush()

                """Visualization"""
                if np.mod(idx+1, np.ceil(self.num_batches/self.vis_num)) == 0 or idx+1 == self.num_batches:
                    for batch_idx in range(3):
                        if self.noise_dist == 'Uniform':
                            batch_noise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                        elif self.noise_dist == 'Normal':
                            batch_noise = np.random.normal(0.0,1.0, [self.batch_size, self.z_dim])
                        test_images, test_noisy_batch = load_CelebA(self.test_list[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )
                        "FeedDict"
                        _feed_dict={self.inputs: test_images, self.noisy_inputs: test_noisy_batch, self.noise: self.fixed_noise[batch_idx]}
                        samples_rand, samples_deco, samples_deco_fake = self.sess.run([self.fake_images,self.De_test,self.De_test_fake],feed_dict=_feed_dict)

                        tot_num_samples = min(self.sample_num, self.batch_size)
                        manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                        manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                        save_images(samples_rand[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                    './' + check_folder(self.result_dir + '/' + self.model_dir + '/' + '{:02d}'.format(epoch)) + '/' + '{:03d}_{:02d}_GE_samples.png'.format(idx+1, batch_idx))
                counter += 1
                start_batch_id=0
            # save model
            self.save(self.checkpoint_dir, counter)

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
