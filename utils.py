"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
from __future__ import division
import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import range
import matplotlib.pyplot as plt
import os, gzip
import cv2, random

import tensorflow as tf
import tensorflow.contrib.slim as slim

## Preparing input images
def resize_image(batch_image, target_width, target_height):
    in_shape = batch_image.shape
    out_batch = np.zeros([in_shape[0],target_width, target_height,in_shape[3]])
    for num in range(in_shape[0]):
        out_batch[num,...] = np.expand_dims(cv2.resize(batch_image[num,...],(target_width,target_height)),axis=2)
    return out_batch

def shuffle_data(trX, trY, teX, teY):
    seed = np.random.seed()
    np.random.seed(seed);    np.random.shuffle(trX)
    np.random.seed(seed);    np.random.shuffle(teX)
    np.random.seed(seed);    np.random.shuffle(trY)
    np.random.seed(seed);    np.random.shuffle(teY)
    return trX, trY, teX, teY

def shuffle_half_data(trX):
    seed = np.random.seed()
    np.random.seed(seed);    np.random.shuffle(trX)
    return trX

def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

def load_fashion_mnist(dataset_name):
    data_dir = os.path.join("./data/fahsion", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

## Check graph
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

## Save/Load images
def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def save_images_bp(images, size, image_path):
    return imsave(inverse_transform_bp(images), size, image_path)

def save_images_YUV(images, size, image_path):
    return imsave(inverse_transform_YUV(images), size, image_path)

def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)
    # return cv2.imwrite(path, image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    images = (images+1.)/2.
    images = np.clip(images, 0, 1)
    return images

def inverse_transform_bp(images):
    images = np.clip(images, 0, 1)
    return images

def inverse_transform_YUV(images):
    images = (images+1.)/2.
    images = (np.clip(images, 0, 1)*255.).astype(np.uint8)
    if images.shape[3]==1:
        return images/255.
    elif images.shape[3] in (3,4):
        for idx, _image in enumerate(images):
            images[idx] = cv2.cvtColor(_image, cv2.COLOR_YCrCb2RGB)
        return images/255.
    else:
        ValueError('FUCK')


""" Drawing Tools """
# borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)
  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)


""" Image Normlaization between 0 and 1 """
def load_image(addr, batch_size, row, col, depth,noise_type='s&p',amount=0.25):
    batch_image = np.empty([batch_size, row, col, depth])
    noise_batch_image = np.copy(batch_image)
    for i in range(batch_size):
        if depth == 3:
            img = cv2.imread(addr[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            noise_image = noisy(noise_type,img,amount)
        else:
            img = cv2.imread(addr[i],0)
            img = np.expand_dims(img, axis=2)
            noise_image = noisy(noise_type,img,amount)
        img = img.astype(np.float32)/255.
        batch_image[i,...] = img
        noise_image = np.reshape(noise_image,[row,col,depth])
        noise_image = noise_image.astype(np.float32)/255.
        noise_batch_image[i,...] = noise_image
    return batch_image, noise_batch_image

def recon_image(norm):
    image = np.clip((norm*255.), 0, 255)
    return image.astype("uint8")

""" Image Normlaization between -1 and 1 """
def norm_img(addr, batch_size, row, col, depth,noise_type='s&p',amount=0.25):
    batch_image = np.empty([batch_size, row, col, depth])
    noise_batch_image = np.copy(batch_image)
    for i in range(batch_size):
        if depth == 3:
            img = cv2.imread(addr[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            noise_image = noisy(noise_type,img,amount)
        else:
            img = cv2.imread(addr[i],0)
            img = np.expand_dims(img, axis=2)
            noise_image = noisy(noise_type,img,amount)
        img = img.astype(np.float32)/127.5 - 1.
        batch_image[i,...] = img
        noise_image = np.reshape(noise_image,[row,col,depth])
        noise_image = noise_image.astype(np.float32)/127.5 - 1.
        noise_batch_image[i,...] = noise_image
    return batch_image, noise_batch_image
def denorm_img(norm):
    image = np.clip(((norm + 1.)*127.5), 0, 255)
    return image.astype("uint8")

"""Norm_img version gray"""
def norm_YCrCb_with_gray(addr, batch_size, row, col, depth,noise_type='s&p',amount=0.25):
    batch_image = np.empty([batch_size, row, col, depth])
    gray_batch_image = np.empty([batch_size, row, col, 1])
    for i in range(batch_size):
        if depth == 3:
            img = cv2.imread(addr[i])
            ycbcr_img  = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            _y,cb,cr   = cv2.split(ycbcr_img)
        else:
            img = cv2.imread(addr[i],0)
            img = np.expand_dims(img, axis=2)
            noise_image = noisy(noise_type,img,amount)
        ycbcr_img = ycbcr_img.astype(np.float32)/127.5 - 1.
        batch_image[i,...] = ycbcr_img
        y_img = _y.astype(np.float32)/127.5 - 1.
        gray_batch_image[i,...] = np.expand_dims(y_img,axis=3)
    return batch_image, gray_batch_image

""" CelebA Image Normlaization between -1 and 1 """
def load_CelebA(addr, batch_size, row, col, depth, noise_type='s&p',amount=0.25):
    batch_image       = np.empty([batch_size, row, col, depth])
    noise_batch_image = np.copy(batch_image)
    for i in range(batch_size):
        if depth == 3:
            img = cv2.imread(addr[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[50:50+128 , 25:25+128, :]
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
            noise_img = noisy(noise_type,img,amount)
        else:
            img = cv2.imread(addr[i],0)
            img = np.expand_dims(img, axis=2)
            noise_img = noisy(noise_type,img,amount)
        img = img.astype(np.float32)/127.5 - 1.
        batch_image[i,...] = img
        noise_img = np.reshape(noise_img,[row,col,depth])
        noise_img = noise_img.astype(np.float32)/127.5 - 1.
        noise_batch_image[i,...] = noise_img
    return batch_image, noise_batch_image

def recon_CelebA(norm):
    image = np.clip(((norm + 1.)*127.5), 0, 255)
    return image.astype("uint8")

""" Shuffle List"""
def shuffle_list(_color_list, _depth_list):
    temp_list = list(zip(_color_list, _depth_list))
    random.shuffle(temp_list)
    _color_list, _depth_list = zip(*temp_list)
    return _color_list, _depth_list

def noisy(noise_typ,image,amount):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        #row,col,ch = image.shape
        image_shape = image.shape
        if image_shape[2] == 1:
            image_shape = image_shape[:2]
        s_vs_p = 0.5
        amount = amount
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image_shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image_shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy
    else:
        raise NotImplementedError

def backprop_vis(target, target_grad_norm, img_grad):
    guided_backprop = np.zeros_like(img_grad)
    guided_grad_CAM = np.zeros_like(img_grad)
    batch_size = np.shape(target)[0]
    for idx in range(batch_size):
        guided_backprop[idx],guided_grad_CAM[idx] = visualize(target[idx], target_grad_norm[idx], img_grad[idx])
    return guided_backprop,guided_grad_CAM

def visualize(conv_output, target_grad_norm, img_grad):
    output = conv_output
    grads_val = target_grad_norm
    img_grad = np.dstack((
            img_grad[:, :, 2],
            img_grad[:, :, 1],
            img_grad[:, :, 0],
        ))
    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) # scale 0 to 1.0
    cam = scipy.misc.imresize(cam, (64,64))
    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)/255.

    # img_grad -= np.min(img_grad)
    # img_grad /= np.max(img_grad)
    img_grad -= np.mean(img_grad)
    img_grad /= np.std(img_grad) + 1e-5
    img_grad *= 0.1
    img_grad += 0.5
    img_grad = np.clip(img_grad,0,1)

    # grad_CAM = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

    guided_grad_CAM = np.dstack((
            img_grad[:, :, 0] * cam,
            img_grad[:, :, 1] * cam,
            img_grad[:, :, 2] * cam,
        ))
    return img_grad, cam_heatmap

def backprop_vis_grad(img_grad):
    guided_backprop = np.zeros_like(img_grad)
    batch_size = np.shape(img_grad)[0]
    for idx in range(batch_size):
        guided_backprop[idx] = visualize_grad(img_grad[idx])
    return guided_backprop

def visualize_grad(img_grad):
    img_grad = np.dstack((
            img_grad[:, :, 2],
            img_grad[:, :, 1],
            img_grad[:, :, 0],
        ))
    img_grad -= np.mean(img_grad)
    img_grad /= np.std(img_grad) + 1e-5
    img_grad *= 0.1
    img_grad += 0.5
    img_grad = np.clip(img_grad,0,1)
    return img_grad
