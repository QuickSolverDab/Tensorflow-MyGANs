"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from numpy.lib.stride_tricks import as_strided as ast

import tensorflow.contrib.slim as slim
from utils import *
from scipy import signal
from scipy.ndimage.filters import convolve


# initializer
xavier_init = tf.contrib.layers.xavier_initializer()

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)
def ln(x, is_training, scope):
    return tf.contrib.layers.layer_norm(x,
                                        trainable=is_training,
                                        scope=scope)

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=xavier_init) #tf.truncated_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        conv = conv + biases
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", stddev=0.02, with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=xavier_init)

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, initializer=xavier_init)
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

"""Activation"""
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

"""Losses"""
def softmax_cross_entropy(_logits,_labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_logits, labels=_labels))

def binary_cross_entropy(_logits,_labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=_logits, labels=_labels))

def mse_loss(_logits, _labels):
    loss = tf.reduce_mean(tf.squared_difference(_logits, _labels))
    return loss

def l1_loss(_logits, _labels):
    loss = tf.reduce_mean(tf.losses.absolute_difference(_logits, _labels))
    return loss

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def cross_entropy(_logits,_labels):
    return  tf.reduce_mean(-tf.reduce_sum(tf.log(_logits + 1e-8) * _labels, axis=1))

def gram_matrix(tensor):
    """Compute the Gram Matrix for a set of feature maps"""
    batch_size, height, width, channels = tensor.get_shape().as_list()
    denominator = (height*width)
    feature_maps = tf.reshape(tensor,tf.stack([batch_size, height*width, channels]))
    gram = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
    gram_norm = gram/tf.constant(denominator,tf.float32)
    return gram_norm

def fc_noise(self, noise, channel, layer, is_training=True, name='fc_noise'):
    with tf.variable_scope(name):
        _patch = self.patch * np.power(2,layer)
        _noise = tf.nn.relu(bn(linear(noise, _patch*_patch*channel[layer], scope='ge_noise'), is_training=is_training, scope='ge_noise_bn'))
        _noise = tf.reshape(_noise, [self.batch_size, _patch, _patch, channel[layer]])
    return _noise

def add_gaussian_noise(input_img, std):
    noise = tf.random_normal(shape=tf.shape(input_img), mean=0.0, stddev=std, dtype=tf.float32)
    noisy_input = input_img + noise
    return tf.clip_by_value(noisy_input,clip_value_min = -1, clip_value_max = 1)

def conv2d_slim(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)
def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def residule_block_linear(x, out_dim, is_training =False, name='res_lin'):
    y = lrelu(bn(linear(x, out_dim, scope=name+'_c1'), is_training = is_training, scope = name+'_bn1'))
    y = bn(linear(x, out_dim, scope=name+'_c2'), is_training = is_training, scope = name+'_bn2')
    out = tf.concat([y,x],axis=1)
    return out

def slice_patch(x, offset_idx, patch_shape):
    _offset       = tf.concat([[0],offset_idx,[0]],axis=0)
    _patch_shape  = tf.concat([[-1],patch_shape,[-1]],axis=0)
    sliced_patch = tf.slice(x, _offset, _patch_shape)
    return sliced_patch

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
