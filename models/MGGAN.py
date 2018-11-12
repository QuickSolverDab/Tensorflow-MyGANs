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

class MGGAN(object):
    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, patch_depth, patch_size, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = 'celebA';
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.model_name = "MGGAN"     # name for checkpoint
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

            """ MGGAN hyper_params """
            self.pre_epoch  = 10
            self.loss_ratio = 0.5

            # train
            self.learning_rate_AE = 1e-3
            self.learning_rate_GAN  = 2e-4
            self.beta1 = 0.5
            # load data list
            train_dir = "./../../dataset/CelebA/splits/train/"
            test_dir  = "./../../dataset/CelebA/splits/test/"
            self.train_list = glob(train_dir+"*.jpg")
            self.test_list = glob(test_dir+"*.jpg")
            data_num = len(self.train_list)
            self.train_list = self.train_list[:data_num]
            # get number of batches for a single epoch
            self.num_batches = len(self.train_list) // self.batch_size
        else:
            raise NotImplementedError

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

    def discriminator_z(self, z, is_training=True, reuse=False, name='discriminator_z'):
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


    def build_model(self):
        # some parameters
        bs = self.batch_size
        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + self.image_dims, name='real_images')
        # Salt and Pepper noise
        self.noisy_inputs = tf.placeholder(tf.float32, [bs] + self.image_dims, name='noisy_inputs')
        # labels
        self.noise = tf.placeholder(tf.float32, [bs, self.z_dim], name='noise')

        """ Network_in/out """
        ### 1. Autoencoder
        En = self.encoder(self.noisy_inputs, is_training=True, reuse=False, name='encoder')
        De = self.decoder(En, is_training=True, reuse=False, name='decoder')

        ### 2 Generator
        Ge_image = self.generator(self.noise, is_training=True, reuse=False, name='generator')

        ### 3. Discriminator
        Dx_real, Dx_real_logits = self.discriminator_x(self.inputs, is_training=True, reuse=False, name='discriminator_x')
        Dx_fake, Dx_fake_logits = self.discriminator_x(Ge_image,    is_training=True, reuse=True, name='discriminator_x')

        En_real    = self.encoder(self.inputs, is_training=False, reuse=True, name='encoder')
        En_fake    = self.encoder(Ge_image,    is_training=False, reuse=True, name='encoder')
        Dz_real, Dz_real_logits = self.discriminator_z(En_real, is_training=True, reuse=False, name='discriminator_z')
        Dz_fake, Dz_fake_logits = self.discriminator_z(En_fake, is_training=True, reuse=True, name='discriminator_z')

        ### 4. Connector
        re_fake = self.connector(En_fake, is_training=True, reuse=False, name='connector')
        re_real = self.connector(En_real, is_training=False, reuse=True, name='connector')

        """Loss_compute"""
        ### Autoencoder Loss
        self.ae_loss  = l1_loss(De, self.inputs)
        ### Discriminator Loss of X
        dx_loss_real  = binary_cross_entropy( Dx_real_logits,  tf.ones_like(Dx_real_logits))
        dx_loss_fake  = binary_cross_entropy( Dx_fake_logits,  tf.zeros_like(Dx_fake_logits))
        self.dx_loss  = dx_loss_real + dx_loss_fake
        ### Discriminator Loss of Z
        dz_loss_real  = binary_cross_entropy( Dz_real_logits,  tf.ones_like(Dz_real_logits))
        dz_loss_fake  = binary_cross_entropy( Dz_fake_logits,  tf.zeros_like(Dz_fake_logits))
        self.dz_loss  = dz_loss_real + dz_loss_fake
        ### Generator Loss
        self.gx_loss  = binary_cross_entropy( Dx_fake_logits,   tf.ones_like(Dx_fake_logits))
        self.gz_loss  = binary_cross_entropy( Dz_fake_logits,   tf.ones_like(Dz_fake_logits))
        self.ge_loss = self.gx_loss + self.loss_ratio*self.gz_loss
        ### Connector
        self.co_loss  = l1_loss(re_fake,self.noise)


        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        en_vars = [var for var in t_vars if 'encoder' in var.name]
        de_vars = [var for var in t_vars if 'decoder' in var.name]
        ge_vars = [var for var in t_vars if 'generator' in var.name]
        dx_vars = [var for var in t_vars if 'discriminator_x' in var.name]
        dz_vars = [var for var in t_vars if 'discriminator_z' in var.name]
        co_vars = [var for var in t_vars if 'connector' in var.name]
        ae_vars = en_vars + de_vars

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.ae_optm = tf.train.AdamOptimizer(self.learning_rate_AE,   beta1=self.beta1).minimize(self.ae_loss, var_list=ae_vars)
            self.ge_optm = tf.train.AdamOptimizer(self.learning_rate_GAN,  beta1=self.beta1).minimize(self.ge_loss, var_list=ge_vars)
            self.dx_optm = tf.train.AdamOptimizer(self.learning_rate_GAN,  beta1=self.beta1).minimize(self.dx_loss, var_list=dx_vars)
            self.dz_optm = tf.train.AdamOptimizer(self.learning_rate_GAN,  beta1=self.beta1).minimize(self.dz_loss, var_list=dz_vars)
            self.co_optm = tf.train.AdamOptimizer(self.learning_rate_GAN,  beta1=self.beta1).minimize(self.co_loss, var_list=co_vars)

        """" Testing """
        # for test
        En_test = self.encoder(self.inputs, is_training=False, reuse=True, name='encoder')
        self.De_test = self.decoder(En_test, is_training=False, reuse=True, name='decoder')
        self.fake_images  = self.generator(self.noise, is_training=False, reuse=True, name='generator')
        self.recon_images = self.generator(re_real, is_training=False, reuse=True, name='generator')
        En_test_fake = self.encoder(self.fake_images, is_training=False, reuse=True, name='encoder')
        self.De_test_fake = self.decoder(En_test_fake, is_training=False, reuse=True, name='decoder')
        """ Summary """
        # Summary
        tf.summary.scalar("ae_loss", self.ae_loss)
        tf.summary.scalar("dx_loss", self.dx_loss)
        tf.summary.scalar("gx_loss", self.gx_loss)
        tf.summary.scalar("gz_loss", self.gz_loss)
        tf.summary.scalar("ge_loss", self.ge_loss)
        tf.summary.scalar("dz_loss", self.dz_loss)
        tf.summary.scalar("co_loss", self.co_loss)
        self.summary_op = tf.summary.merge_all()

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()
        # saver to save model
        self.saver = tf.train.Saver()
        # summary writer
        name = "{}_ratio_{}_init_depth_{}".format(self.model_name,self.loss_ratio, self.depth)
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
                random.shuffle(self.train_list)
                random.shuffle(self.test_list)
                for pre_idx in range(0, self.num_batches):
                    if (pre_idx+1)*self.batch_size <= len(self.train_list):
                        batch_images, noisy_batch = load_CelebA(self.train_list[pre_idx*self.batch_size:(pre_idx+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )
                    else:
                        pass #batch_images, noisy_batch = load_CelebA(self.train_list[pre_idx*self.batch_size:])
                    # Prioir Distribution and Random Noise Distribution
                    if self.noise_dist == 'Uniform':
                        batch_noise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                    elif self.noise_dist == 'Normal':
                        batch_noise = np.random.normal(0.0,1.0, [self.batch_size, self.z_dim])
                    _feed_dict = {self.inputs: batch_images, self.noisy_inputs: noisy_batch, self.noise: batch_noise}
                    _ = self.sess.run(self.ae_optm, feed_dict= _feed_dict)
                    if np.mod(pre_idx, 50) == 0 or pre_idx == self.num_batches:
                        _AE = self.sess.run(self.ae_loss, feed_dict={self.inputs: batch_images, self.noisy_inputs: noisy_batch, self.noise: batch_noise})
                        print("Pre_Epoch: [%2d] [%4d/%4d] time: %4.4f, AE Loss: %.4f" \
                                % (pre_epoch,pre_idx,self.num_batches,time.time()-start_time,  _AE ))
            print('Pre-train is done')
            self.save(self.checkpoint_dir, counter)

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            random.shuffle(self.train_list)
            random.shuffle(self.test_list)
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
                ## Training Part
                # update Discriminator
                _,_ = self.sess.run([self.dx_optm,self.dz_optm], feed_dict=_feed_dict )
                # update Generator & Connection network
                for x in range(1):
                    _,_ = self.sess.run([self.ge_optm,self.co_optm], feed_dict=_feed_dict )

                # display training status
                if np.mod(idx, 50) == 0 or idx+1 == self.num_batches:
                    _AE,_DR_x,_DR_z,_GE,_Gx,_Gz = self.sess.run([self.ae_loss,self.dx_loss,self.dz_loss,self.ge_loss,self.gx_loss,self.gz_loss], feed_dict=_feed_dict)
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, Loss:[AE:%.4f, DR_x:%.4f, DR_z:%.4f, GE:%.4f, Gx:%.4f, Gz:%.4f]"
                            % (epoch,idx,self.num_batches,time.time()-start_time,  _AE,  _DR_x, _DR_z,  _GE,  _Gx,  _Gz))


                """Image Save"""
                if np.mod(idx+1, np.ceil(self.num_batches/self.vis_num)) == 0 or idx+1 == self.num_batches:
                # if idx == 10:
                    """Summary"""
                    summary = self.sess.run(self.summary_op, feed_dict=_feed_dict )
                    self.writer.add_summary(summary,counter)
                    self.writer.flush()
                    # self.save(self.checkpoint_dir, counter)
                    test_ssim = []
                    test_psnr = []
                    for batch_idx in range(5):
                        # if idx != 0:
                        if self.noise_dist == 'Uniform':
                            batch_noise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                        elif self.noise_dist == 'Normal':
                            batch_noise = np.random.normal(0.0,1.0, [self.batch_size, self.z_dim])
                        test_images, test_noisy_batch = load_CelebA(self.test_list[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size],self.batch_size, self.input_height, self.input_width, self.c_dim )
                        "FeedDict"
                        _feed_dict={self.inputs: test_images, self.noisy_inputs: test_noisy_batch, self.noise: batch_noise}
                        samples_rand,recon_rand = self.sess.run([self.fake_images,self.recon_images], feed_dict=_feed_dict)
                        # Test vs Recon
                        test_recon = np.zeros_like(samples_rand)
                        for col in range(int(self.batch_size/2)):
                            col = col*2
                            test_recon[ col   ,...] = test_images[col,...]
                            test_recon[(col+1),...] =  recon_rand[col,...]
                        # Recon_36
                        tot_num_samples = min(36, self.batch_size)
                        manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                        manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                        save_images(samples_rand[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                    './' + check_folder(self.result_dir + '/' + self.model_dir + '/' + '{:02d}'.format(epoch)) + '/' + '{:03d}_{:02d}_GE_samples_v36.png'.format(idx+1, batch_idx))
                        save_images(test_recon[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                    './' + check_folder(self.result_dir + '/' + self.model_dir + '/' + '{:02d}'.format(epoch)) + '/' + '{:03d}_{:02d}_recon_compare_v36.png'.format(idx+1, batch_idx))
                counter += 1
                start_batch_id=0
            # save model
            self.save(self.checkpoint_dir, counter)


    @property
    def model_dir(self):
        return "{}_ratio_{}_init_depth_{}".format(self.model_name,self.loss_ratio, self.depth)

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
