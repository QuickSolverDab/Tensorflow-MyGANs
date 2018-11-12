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

class ResembledGAN(object):
    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, patch_depth, patch_size, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name;
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.model_name = "ResembledGAN"     # name for checkpoint
        self.epoch = epoch
        """hyper_params"""
        self.n = patch_depth
        self.patch = patch_size
        self.z_dim = z_dim
        self.noise_dist = 'Uniform'
        """Vis_parameters"""
        self.vis_num = 2
        self.sample_num = 64  # number of generated images to be saved

        if dataset_name == 'cat2celebA':
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

            # Feature Matching(Average Feature Matching)
            self.image_dims = [self.input_height, self.input_width, self.c_dim]
            # train
            self.learning_rate_AE  = 1e-3
            self.learning_rate_GAN = 2e-4
            self.beta1 = 0.5
            # test
            self.sample_num = 64  # number of generated images to be saved

            """ ResembledGAN hyper_params """
            self.pre_epoch = 20
            self.w_similar = 1.
            # mean
            self.meanA = 0
            self.meanB = 0

            # load data list
            trainA_dir = "./../../dataset/CelebA/splits/train/"
            trainB_dir = "./../../dataset/CelebA/splits/train/" #"./../../dataset/CAT/train/"
            testA_dir  = "./../../dataset/CelebA/splits/test/"
            testB_dir  = "./../../dataset/CelebA/splits/test/" #"./../../dataset/CAT/test/"
            self.trainA_list = glob(trainA_dir+"*.jpg")
            self.trainB_list = glob(trainB_dir+"*.jpg")
            self.testA_list  = glob(testA_dir+"*.jpg")
            self.testB_list  = glob(testB_dir+"*.jpg")
            # A-B
            data_num = len(self.trainA_list)
            self.trainA_list = self.trainA_list[:data_num]
            self.trainB_list = self.trainB_list[:data_num]
            data_num = len(self.testA_list)
            self.testA_list = self.testA_list[:data_num]
            self.testB_list = self.testB_list[:data_num]
            # get number of batches for a single epoch
            self.num_batches = len(self.trainA_list) // self.batch_size
        else:
            raise NotImplementedError

    def encoder(self, x, is_training=True, reuse=False, name='encoder'):
        with tf.variable_scope(name, reuse=reuse):
            net = lrelu(   conv2d(x,   self.depth*1,  5, 5, 2, 2, name='en_conv1')) #64
            net = lrelu(bn(conv2d(net, self.depth*2,  5, 5, 2, 2, name='en_conv2'),  is_training=is_training, scope='en_bn2')) #32
            net = lrelu(bn(conv2d(net, self.depth*4,  5, 5, 2, 2, name='en_conv3'),  is_training=is_training, scope='en_bn3')) #16
            net = lrelu(bn(conv2d(net, self.depth*8,  5, 5, 2, 2, name='en_conv4'),  is_training=is_training, scope='en_bn4')) #8
            net =          conv2d(net, self.n,        1, 1, 1, 1, name='en_conv5') #4
            z   = tf.reshape(net, [self.batch_size, -1]);
        return z

    def decoder(self, z, is_training=True, reuse=False, name='decoder'):
        with tf.variable_scope(name, reuse=reuse):
            z = tf.reshape(z, [self.batch_size, self.patch, self.patch, self.n]);
            z = tf.nn.relu(bn(conv2d(z, self.depth*4, 1, 1, 1, 1, name='de_dc0'), is_training=is_training, scope='de_bn0'));
            net = tf.nn.relu(bn(deconv2d(z,     [self.batch_size, 8,  8,  self.depth*4], 5, 5, 2, 2, name='de_dc1'), is_training=is_training, scope='de_bn1'));
            net = tf.nn.relu(bn(deconv2d(net,   [self.batch_size, 16, 16, self.depth*2], 5, 5, 2, 2, name='de_dc2'), is_training=is_training, scope='de_bn2'));
            net = tf.nn.relu(bn(deconv2d(net,   [self.batch_size, 32, 32, self.depth*1], 5, 5, 2, 2, name='de_dc3'), is_training=is_training, scope='de_bn3'));
            out =               deconv2d(net,   [self.batch_size, 64, 64,   self.c_dim], 5, 5, 2, 2, name='de_dc4');
        return out

    def discriminator_x(self, x, is_training=True, reuse=False, name='discriminator_x'):
        with tf.variable_scope(name, reuse=reuse):
            net  = lrelu(   conv2d(x,    self.depth*1,  5, 5, 2, 2,  name='disc_conv1')) # 16*16*64
            net  = lrelu(bn(conv2d(net,  self.depth*2,  5, 5, 2, 2,  name='disc_conv2'), is_training=is_training, scope='disc_bn2')) #  8* 8*128
            net  = lrelu(bn(conv2d(net,  self.depth*4,  5, 5, 2, 2,  name='disc_conv3'), is_training=is_training, scope='disc_bn3')) #  4* 4*256
            net  = lrelu(bn(conv2d(net,  self.depth*8,  5, 5, 2, 2,  name='disc_conv4'), is_training=is_training, scope='disc_bn4')) #  4* 4*256
            feature = tf.reshape(net,[self.batch_size,-1])
            out_logit = linear(feature,1,scope='disc_out_logit' )
            out = tf.sigmoid(out_logit)
        return out, out_logit

    def discriminator_m(self, z, is_training=True, reuse=False, name='discriminator_m'):
        with tf.variable_scope(name, reuse=reuse):
            net = lrelu(linear(z,      1024, scope='disc_fc0'))
            net = lrelu(bn(linear(net, 1024, scope='disc_fc1'),is_training=is_training, scope='disc_fcbn1'))
            out_logit = linear(net, 1, scope='disc_fc2')
            out = tf.nn.sigmoid(out_logit)
            return out, out_logit

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

    def connector(self, z, is_training=True, reuse=False, name='connector'):
        with tf.variable_scope(name, reuse=reuse):
            net = lrelu(   linear(z,   1024, scope='conn_fc0'))
            net = lrelu(bn(linear(net, 1024, scope='conn_fc1'),is_training=is_training, scope='conn_fcbn1'))
            out_logit =    linear(net, self.z_dim, scope='conn_fc2')
            out = tf.nn.tanh(out_logit)
            return out

    def build_model(self):
        # some parameters
        bs = self.batch_size
        """ Graph Input """
        ## Images
        self.inputsA = tf.placeholder(tf.float32, [bs] + self.image_dims, name='real_imagesA')
        self.inputsB = tf.placeholder(tf.float32, [bs] + self.image_dims, name='real_imagesB')
        ## Salt and Pepper noise
        self.noisy_inputsA = tf.placeholder(tf.float32, [bs] + self.image_dims, name='noisy_inputsA')
        self.noisy_inputsB = tf.placeholder(tf.float32, [bs] + self.image_dims, name='noisy_inputsB')
        ## Noise
        self.noise = tf.placeholder(tf.float32, [bs, self.z_dim], name='noise')

        """ Network_in/out """
        ### 1. Autoencoder
        # A
        EnA   = self.encoder(self.noisy_inputsA, is_training=True, reuse=False, name='encoder')
        DeA   = self.decoder(EnA,                is_training=True, reuse=False, name='decoder')
        # B
        EnB   = self.encoder(self.noisy_inputsB, is_training=True, reuse=True,  name='encoder')
        DeB   = self.decoder(EnB,                is_training=True, reuse=True,  name='decoder')

        ### 2 Generator
        Ge_imageA = self.generator(self.noise, is_training=True, reuse=False, name='generatorA')
        Ge_imageB = self.generator(self.noise, is_training=True, reuse=False, name='generatorB')

        ### 3. Discriminator
        # A
        DxA_real_logits = self.discriminator_x(self.inputsA, is_training=True, reuse=False, name='discriminator_xA')[1]
        DxA_fake_logits = self.discriminator_x(Ge_imageA,    is_training=True, reuse=True,  name='discriminator_xA')[1]
        EnA_real        = self.encoder(self.inputsA, is_training=False, reuse=True, name='encoder')
        EnA_fake        = self.encoder(Ge_imageA   , is_training=False, reuse=True, name='encoder')
        dmA_real_logits = self.discriminator_m(EnA_real, is_training=True, reuse=False, name='discriminator_mA')[1]
        dmA_fake_logits = self.discriminator_m(EnA_fake, is_training=True, reuse=True,  name='discriminator_mA')[1]
        # B
        DxB_real_logits = self.discriminator_x(self.inputsB, is_training=True, reuse=False, name='discriminator_xB')[1]
        DxB_fake_logits = self.discriminator_x(Ge_imageB,    is_training=True, reuse=True,  name='discriminator_xB')[1]
        EnB_real        = self.encoder(self.inputsB, is_training=False, reuse=True, name='encoder')
        EnB_fake        = self.encoder(Ge_imageB   , is_training=False, reuse=True, name='encoder')
        dmB_real_logits = self.discriminator_m(EnB_real, is_training=True, reuse=False, name='discriminator_mB')[1]
        dmB_fake_logits = self.discriminator_m(EnB_fake, is_training=True, reuse=True,  name='discriminator_mB')[1]

        ### 4. Connector
        # Connector Objective
        nsA_fake        = self.connector(EnA_fake,   is_training=True,  reuse=False,  name='connector')
        nsB_fake        = self.connector(EnB_fake,   is_training=True,  reuse=True,   name='connector')
        # Similarity Loss
        nsA_real        = self.connector(EnA_real,   is_training=False, reuse=True,   name='connector')
        nsB_real        = self.connector(EnB_real,   is_training=False, reuse=True,   name='connector')
        Ge_imageRA      = self.generator(nsA_real,   is_training=True,  reuse=True,   name='generatorA')
        Ge_imageRB      = self.generator(nsB_real,   is_training=True,  reuse=True,   name='generatorB')
        # Trans loss
        DxRA_fake_logits = self.discriminator_x(Ge_imageRA, is_training=True, reuse=True,  name='discriminator_xA')[1]
        DxRB_fake_logits = self.discriminator_x(Ge_imageRB, is_training=True, reuse=True,  name='discriminator_xB')[1]

        """Loss_compute"""
        ### Autoencoder Loss
        # Autoencoder
        self.ae_loss  = l1_loss(DeA, self.inputsA) + l1_loss(DeB, self.inputsB)
        ### Discriminator Loss of X
        ## A
        # X
        dxA_loss_real   = binary_cross_entropy( DxA_real_logits,   tf.ones_like(DxA_real_logits))
        dxA_loss_fake   = binary_cross_entropy( DxA_fake_logits,   tf.zeros_like(DxA_fake_logits))
        dxRA_loss_fake  = binary_cross_entropy( DxRA_fake_logits,  tf.zeros_like(DxRA_fake_logits))
        self.dxA_loss   = dxA_loss_real + (dxA_loss_fake + dxRA_loss_fake)/2
        # z
        dmA_loss_real   = binary_cross_entropy( dmA_real_logits,   tf.ones_like(dmA_real_logits))
        dmA_loss_fake   = binary_cross_entropy( dmA_fake_logits,   tf.zeros_like(dmA_fake_logits))
        self.dmA_loss   = dmA_loss_real + dmA_loss_fake
        ## B
        # X
        dxB_loss_real   = binary_cross_entropy( DxB_real_logits,   tf.ones_like(DxB_real_logits))
        dxB_loss_fake   = binary_cross_entropy( DxB_fake_logits,   tf.zeros_like(DxB_fake_logits))
        dxRB_loss_fake  = binary_cross_entropy( DxRB_fake_logits,  tf.zeros_like(DxRB_fake_logits))
        self.dxB_loss   = dxB_loss_real + (dxB_loss_fake + dxRB_loss_fake)/2
        # z
        dmB_loss_real   = binary_cross_entropy( dmB_real_logits,   tf.ones_like(dmB_real_logits))
        dmB_loss_fake   = binary_cross_entropy( dmB_fake_logits,   tf.zeros_like(dmB_fake_logits))
        self.dmB_loss   = dmB_loss_real + dmB_loss_fake

        """ Loss Combine"""
        self.dx_loss = self.dxA_loss + self.dxB_loss
        self.dm_loss = self.dmA_loss + self.dmB_loss
        ### Generator Loss
        # X
        self.gxA_loss   = binary_cross_entropy( DxA_fake_logits,   tf.ones_like(DxA_fake_logits)) \
                        + binary_cross_entropy( DxRA_fake_logits,  tf.ones_like(DxRA_fake_logits))
        self.gmA_loss   = binary_cross_entropy( dmA_fake_logits,   tf.ones_like(dmA_fake_logits))
        self.geA_loss   = self.gxA_loss + self.gmA_loss
        # z
        self.gxB_loss   = binary_cross_entropy( DxB_fake_logits,   tf.ones_like(DxB_fake_logits)) \
                        + binary_cross_entropy( DxRB_fake_logits,  tf.ones_like(DxRB_fake_logits))
        self.gmB_loss   = binary_cross_entropy( dmB_fake_logits,   tf.ones_like(dmB_fake_logits))
        self.geB_loss   = self.gxB_loss + self.gmB_loss
        ### Similar Loss
        self.sim_loss = l1_loss((EnA_fake  - self.meanA), (EnB_fake  - self.meanB))
        self.sim_loss = self.w_similar*self.sim_loss

        """ Loss Combine"""
        self.geAB_loss = self.geA_loss + self.geB_loss + self.sim_loss
        ### Connector Loss
        self.co_loss  = mse_loss(nsA_fake, self.noise) + mse_loss(nsB_fake, self.noise)

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        ## Autoencoder
        en_vars = [var for var in t_vars if 'encoder' in var.name]
        de_vars = [var for var in t_vars if 'decoder' in var.name]
        ae_vars = en_vars + de_vars
        ## Discriminator
        dx_vars = [var for var in t_vars if 'discriminator_x'  in var.name]
        dm_vars = [var for var in t_vars if 'discriminator_m'  in var.name]
        ## Generator
        geA_vars = [var for var in t_vars if 'generatorA' in var.name]
        geB_vars = [var for var in t_vars if 'generatorB' in var.name]
        # Combine
        ge_vars = geA_vars + geB_vars
        ## Connector
        co_vars = [var for var in t_vars if 'connector' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            ## AE
            self.ae_optm  = tf.train.AdamOptimizer(self.learning_rate_AE,   beta1=self.beta1).minimize(self.ae_loss,   var_list=ae_vars)
            ## Discriminator
            self.dx_optm  = tf.train.AdamOptimizer(self.learning_rate_GAN , beta1=self.beta1).minimize(self.dx_loss,   var_list=dx_vars)
            self.dm_optm  = tf.train.AdamOptimizer(self.learning_rate_GAN , beta1=self.beta1).minimize(self.dm_loss,   var_list=dm_vars)
            ## Generator
            self.ge_optm  = tf.train.AdamOptimizer(self.learning_rate_GAN , beta1=self.beta1).minimize(self.geAB_loss, var_list=ge_vars)
            ## Connector
            self.co_optm  = tf.train.AdamOptimizer(self.learning_rate_GAN , beta1=self.beta1).minimize(self.co_loss,   var_list=co_vars)

        """" Testing """
        ### for test
        # AE
        EnA_test = self.encoder(self.inputsA,  is_training=False, reuse=True, name='encoder')
        self.DeA_test = self.decoder(EnA_test, is_training=False, reuse=True, name='decoder')
        EnB_test = self.encoder(self.inputsB,  is_training=False, reuse=True, name='encoder')
        self.DeB_test = self.decoder(EnB_test, is_training=False, reuse=True, name='decoder')
        # Mean
        self.batchmean_A, self.batchvar_A = tf.nn.moments(EnA_test, axes=0)
        self.batchmean_B, self.batchvar_B = tf.nn.moments(EnB_test, axes=0)

        # Generation
        self.fake_imagesA = self.generator(self.noise, is_training=False, reuse=True, name='generatorA')
        self.fake_imagesB = self.generator(self.noise, is_training=False, reuse=True, name='generatorB')
        # Reconstruction
        self.nsA_real     = self.connector(EnA_real,      is_training=False, reuse=True, name='connector')
        self.nsB_real     = self.connector(EnB_real,      is_training=False, reuse=True, name='connector')
        self.fake_reconA  = self.generator(self.nsA_real, is_training=False, reuse=True, name='generatorA')
        self.fake_reconAB = self.generator(self.nsA_real, is_training=False, reuse=True, name='generatorB')
        self.fake_reconB  = self.generator(self.nsB_real, is_training=False, reuse=True, name='generatorB')
        self.fake_reconBA = self.generator(self.nsB_real, is_training=False, reuse=True, name='generatorA')

        """ Summary """
        # Summary
        tf.summary.scalar("ae_loss", self.ae_loss)
        tf.summary.scalar("dx_loss", self.dx_loss)
        tf.summary.scalar("gxA_loss", self.gxA_loss)
        tf.summary.scalar("geA_loss", self.geA_loss)
        tf.summary.scalar("gxB_loss", self.gxB_loss)
        tf.summary.scalar("geB_loss", self.geB_loss)
        tf.summary.histogram("Encoder_A_Distribution_real", values=EnA)
        tf.summary.histogram("Encoder_B_Distribution_real", values=EnB)
        tf.summary.scalar("dm_loss", self.dm_loss)
        tf.summary.scalar("gmA_loss", self.gmA_loss)
        tf.summary.scalar("gmB_loss", self.gmB_loss)
        tf.summary.histogram("Encoder_A_Distribution_Real", values=EnA_real)
        tf.summary.histogram("Encoder_B_Distribution_Real", values=EnB_real)
        tf.summary.histogram("Encoder_A_Distribution_fake", values=EnA_fake)
        tf.summary.histogram("Encoder_B_Distribution_fake", values=EnB_fake)
        self.summary_op = tf.summary.merge_all()


    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()
        # saver to save model
        self.saver = tf.train.Saver()
        # summary writer
        name = "{}_sim_ratio_{}_init_depth_{}".format(self.model_name,self.w_similar, self.depth)
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
                random.shuffle(self.trainA_list); random.shuffle(self.trainB_list)
                random.shuffle(self.testA_list);  random.shuffle(self.testB_list)
                for pre_idx in range(0, self.num_batches):
                    if (pre_idx+1)*self.batch_size <= len(self.trainA_list):
                        batch_imagesA, noisy_batchA = load_CelebA(self.trainA_list[pre_idx*self.batch_size:(pre_idx+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )
                        batch_imagesB, noisy_batchB =    norm_img(self.trainB_list[pre_idx*self.batch_size:(pre_idx+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )
                    else:
                        pass
                    # Prioir Distribution and Random Noise Distribution
                    if self.noise_dist == 'Uniform':
                        batch_noise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                    elif self.noise_dist == 'Normal':
                        batch_noise = np.random.normal(0.0,1.0, [self.batch_size, self.z_dim])
                    _feed_dict  = {self.inputsA: batch_imagesA, self.noisy_inputsA: noisy_batchA, self.inputsB: batch_imagesB, self.noisy_inputsB: noisy_batchB, self.noise: batch_noise}
                    _ = self.sess.run(self.ae_optm, feed_dict=_feed_dict)
                    if np.mod(pre_idx, 50) == 0 or pre_idx == self.num_batches:
                        _AE = self.sess.run(self.ae_loss, feed_dict=_feed_dict)
                        print("Pre_Epoch: [%2d] [%4d/%4d] time: %4.4f, AE Loss: %.4f" % (pre_epoch,pre_idx,self.num_batches,time.time()-start_time,  _AE ))
            """MeanComputation"""
            print('compute Mean of each dataset')
            batchA = []; batchB = []
            random.shuffle(self.trainA_list); random.shuffle(self.trainB_list)
            random.shuffle(self.testA_list);  random.shuffle(self.testB_list)
            for pre_idx in range(0, self.num_batches):
                if (pre_idx+1)*self.batch_size <= len(self.trainA_list):
                    batch_imagesA, noisy_batchA = load_CelebA(self.trainA_list[pre_idx*self.batch_size:(pre_idx+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )
                    batch_imagesB, noisy_batchB =    norm_img(self.trainB_list[pre_idx*self.batch_size:(pre_idx+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )
                else:
                    pass
                # Prioir Distribution and Random Noise Distribution
                if self.noise_dist == 'Uniform':
                    batch_noise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                elif self.noise_dist == 'Normal':
                    batch_noise = np.random.normal(0.0,1.0, [self.batch_size, self.z_dim])
                _feed_dict  = {self.inputsA: batch_imagesA, self.noisy_inputsA: noisy_batchA, self.inputsB: batch_imagesB, self.noisy_inputsB: noisy_batchB, self.noise: batch_noise}
                _batchmean_A, _batchmean_B = self.sess.run([self.batchmean_A,self.batchmean_B], feed_dict=_feed_dict)
                batchA.append(_batchmean_A);        batchB.append(_batchmean_B)
            """Pre-train is done"""
            print('Pre-train is done')
            self.meanA = np.mean(batchA)
            self.meanB = np.mean(batchB)
            self.save(self.checkpoint_dir, counter)

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            random.shuffle(self.trainA_list); random.shuffle(self.testA_list)
            random.shuffle(self.trainB_list); random.shuffle(self.testB_list)
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                if (idx+1)*self.batch_size <= len(self.trainA_list):
                    batch_imagesA, noisy_batchA = load_CelebA(self.trainA_list[idx*self.batch_size:(idx+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )
                    batch_imagesB, noisy_batchB =    norm_img(self.trainB_list[idx*self.batch_size:(idx+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )
                else:
                    pass #batch_images, noisy_batch = load_CelebA(self.train_list[idx*self.batch_size:])
                # Prioir Distribution and Random Noise Distribution
                if self.noise_dist == 'Uniform':
                    batch_noise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                elif self.noise_dist == 'Normal':
                    batch_noise = np.random.normal(0.0,1.0, [self.batch_size, self.z_dim])
                "FeedDict"
                _feed_dict  = {self.inputsA: batch_imagesA, self.noisy_inputsA: noisy_batchA, self.inputsB: batch_imagesB, self.noisy_inputsB: noisy_batchB, self.noise: batch_noise}
                ## Training Part
                # update D network
                _,_ = self.sess.run([self.dx_optm, self.dm_optm], feed_dict=_feed_dict )#, self.dd_optm
                # update G & C network
                for x in range(1):
                    _ = self.sess.run(self.co_optm, feed_dict=_feed_dict )
                    _ = self.sess.run(self.ge_optm, feed_dict=_feed_dict )

                # display training status
                if np.mod(idx, 50) == 0 or idx+1 == self.num_batches:
                    _dxA,_dxB,_dmA,_dmB,_gxA,_gxB,_gmA,_gmB,_co,_sim = self.sess.run([self.dxA_loss,self.dxB_loss,self.dmA_loss,self.dmB_loss,
                                                                                      self.gxA_loss,self.gxB_loss,self.gmA_loss,self.gmB_loss,
                                                                                      self.co_loss, self.sim_loss], feed_dict=_feed_dict)
                    print("Epoch:[%2d] [%4d/%4d] time: %4.2f, Loss:[ Dx:%.4f/%.4f, Dz:%.4f/%.4f, Gx:%.4f/%.4f, Gm:%.4f/%.4f, Conn:%.4f, Sim:%.4f]"
                            % (epoch,idx,self.num_batches,time.time()-start_time,_dxA,_dxB,_dmA,_dmB,_gxA,_gxB,_gmA,_gmB,_co,_sim))

                if np.mod(idx+1, np.ceil(self.num_batches/self.vis_num)) == 0 or idx+1 == self.num_batches:
                    """Summary"""
                    summary = self.sess.run(self.summary_op, feed_dict=_feed_dict )
                    self.writer.add_summary(summary,counter)
                    self.writer.flush()
                    # self.save(self.checkpoint_dir, counter)
                    for batch_idx in range(5):
                        # if idx != 0:
                        if self.noise_dist == 'Uniform':
                            batch_noise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                        elif self.noise_dist == 'Normal':
                            batch_noise = np.random.normal(0.0,1.0, [self.batch_size, self.z_dim])
                        test_imagesA, test_noisy_batchA = load_CelebA(self.testA_list[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )
                        test_imagesB, test_noisy_batchB = norm_img(self.testB_list[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )
                        "FeedDict"
                        _feed_dict  = {self.inputsA: test_imagesA, self.noisy_inputsA: test_noisy_batchA, self.inputsB: test_imagesB, self.noisy_inputsB: test_noisy_batchB, self.noise: batch_noise}
                        samples_randA,samples_randB = self.sess.run([self.fake_reconA,self.fake_reconAB], feed_dict=_feed_dict)
                        reconA,reconAB,reconB,reconBA = self.sess.run([self.fake_reconA,self.fake_reconAB,self.fake_reconB,self.fake_reconBA], feed_dict=_feed_dict)
                        # Compare Random
                        comp_recon = np.zeros_like(samples_randA)
                        for col in range(int(np.floor(self.batch_size/2))):
                            col = col*2
                            comp_recon[ col   ,...] = samples_randA[col,...]
                            comp_recon[(col+1),...] = samples_randB[col,...]
                        # Compare Recon
                        test_reconA = np.zeros_like(samples_randB)
                        for col in range(int(np.floor(self.batch_size/3))):
                            col = col*3
                            test_reconA[ col   ,...] = test_imagesA[col,...]
                            test_reconA[(col+1),...] =       reconA[col,...]
                            test_reconA[(col+2),...] =      reconAB[col,...]
                        test_reconB = np.zeros_like(samples_randB)
                        for col in range(int(np.floor(self.batch_size/3))):
                            col = col*3
                            test_reconB[ col   ,...] = test_imagesB[col,...]
                            test_reconB[(col+1),...] =       reconB[col,...]
                            test_reconB[(col+2),...] =      reconBA[col,...]
                        # Recon_64
                        tot_num_samples = min(self.sample_num, self.batch_size)
                        manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                        manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                        save_images(comp_recon[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                    './' + check_folder(self.result_dir + '/' + self.model_dir + '/' + '{:02d}'.format(epoch)) + '/' + '{:03d}_{:02d}_rand_compare_v36.png'.format(idx+1, batch_idx))
                        save_images(test_reconA[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                    './' + check_folder(self.result_dir + '/' + self.model_dir + '/' + '{:02d}'.format(epoch)) + '/' + '{:03d}_{:02d}_reconA_compare_v36.png'.format(idx+1, batch_idx))
                        save_images(test_reconB[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                    './' + check_folder(self.result_dir + '/' + self.model_dir + '/' + '{:02d}'.format(epoch)) + '/' + '{:03d}_{:02d}_reconB_compare_v36.png'.format(idx+1, batch_idx))
                counter += 1
            # save model
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_sim_ratio_{}_init_depth_{}".format(self.model_name,self.w_similar, self.depth)

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
