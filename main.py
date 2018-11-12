import os
import numpy as np

## GAN Variants
from models.RFGAN  import RFGAN
from models.MGGAN  import MGGAN
from models.ResembledGAN  import ResembledGAN
from models.Recycling_discriminator  import Recycling_discriminator

from utils import show_all_variables
from utils import check_folder

import tensorflow as tf
import argparse

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops


"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['RFGAN', 'MGGAN', 'ResembledGAN','Recycling_discriminator'] ,
                        help='The type of GAN', required=True)
    parser.add_argument('--dataset', type=str, default='celebA', choices=['celebA','celebA2celebA'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=30, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--patch_depth', type=int, default=8, help='depth of latent patches')
    parser.add_argument('--patch_size',  type=int, default=4, help='size of latent patches')
    parser.add_argument('--z_dim', type=int, default=128, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)
    # --result_dir
    check_folder(args.result_dir)
    # --result_dir
    check_folder(args.log_dir)
    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')
    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    # --z_dim
    try:
        assert args.z_dim >= 1
    except:
        print('dimension of noise vector must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        # declare instance for GAN
        if args.gan_type == 'RFGAN':
            gan = RFGAN(sess, epoch=args.epoch, batch_size=args.batch_size, z_dim=args.z_dim,
                        dataset_name=args.dataset, patch_depth=args.patch_depth, patch_size=args.patch_size,
                        checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'MGGAN':
            gan = MGGAN(sess, epoch=args.epoch, batch_size=args.batch_size, z_dim=args.z_dim,
                        dataset_name=args.dataset,  patch_depth=args.patch_depth, patch_size=args.patch_size,
                        checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'ResembledGAN':
            gan = ResembledGAN(sess, epoch=args.epoch, batch_size=args.batch_size, z_dim=args.z_dim,
                        dataset_name=args.dataset, patch_depth=args.patch_depth, patch_size=args.patch_size,
                        checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'Recycling_discriminator':
            gan = Recycling_discriminator(sess, epoch=args.epoch, batch_size=args.batch_size, z_dim=args.z_dim,
                        dataset_name=args.dataset, patch_depth=args.patch_depth, patch_size=args.patch_size,
                        checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        else:
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")


if __name__ == '__main__':
    main()
