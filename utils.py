# -*- coding: utf-8 -*
import scipy.stats as st
import tensorflow as tf
import numpy as np
import sys
from functools import reduce

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)  # 联乘 reduce(mul, [3,2,3], 1)=18，[1:]相当于是不要batch_size,只计算一个tensor中总的height*weidth*……

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1) # 在指定的间隔内返回均匀间隔的数字
    kern1d = np.diff(st.norm.cdf(x))  # st.norm.cdf正态累积分布函数，np.diff([1,3,4,7])=[2,1,3]是差分值
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def blur(x):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')
    # tf.nn.depthwise_conv2d() return A 4-D Tensor of shape [batch, out_height, out_width, in_channels * channel_multiplier]
    # tf.nn.conv2d() return A Tensor: Has the same type as input.

def process_command_args(arguments):

    # specifying default parameters

    batch_size = 50
    train_size = 30000
    learning_rate = 5e-4
    num_train_iters = 20000

    w_content = 10
    w_color = 0.5
    w_texture = 1
    w_tv = 2000

    dped_dir = 'dped/'
    vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    eval_step = 1000

    phone = ""

    for args in arguments:

        if args.startswith("model"):
            phone = args.split("=")[1]

        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("train_size"):
            train_size = int(args.split("=")[1])

        if args.startswith("learning_rate"):
            learning_rate = float(args.split("=")[1])

        if args.startswith("num_train_iters"):
            num_train_iters = int(args.split("=")[1])

        # -----------------------------------

        if args.startswith("w_content"):
            w_content = float(args.split("=")[1])

        if args.startswith("w_color"):
            w_color = float(args.split("=")[1])

        if args.startswith("w_texture"):
            w_texture = float(args.split("=")[1])

        if args.startswith("w_tv"):
            w_tv = float(args.split("=")[1])

        # -----------------------------------

        if args.startswith("dped_dir"):
            dped_dir = args.split("=")[1]

        if args.startswith("vgg_dir"):
            vgg_dir = args.split("=")[1]

        if args.startswith("eval_step"):
            eval_step = int(args.split("=")[1])


    if phone == "":
        print("\nPlease specify the camera model by running the script with the following parameter:\n")
        print("python train_model.py model={iphone,blackberry,sony}\n")
        sys.exit()

    if phone not in ["iphone", "sony", "blackberry"]:
        print("\nPlease specify the correct camera model:\n")
        print("python train_model.py model={iphone,blackberry,sony}\n")
        sys.exit()

    print("\nThe following parameters will be applied for CNN training:\n")

    print("Phone model:", phone)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)
    print("Training iterations:", str(num_train_iters))
    print()
    print("Content loss:", w_content)
    print("Color loss:", w_color)
    print("Texture loss:", w_texture)
    print("Total variation loss:", str(w_tv))
    print()
    print("Path to DPED dataset:", dped_dir)
    print("Path to VGG-19 network:", vgg_dir)
    print("Evaluation step:", str(eval_step))
    print()
    return phone, batch_size, train_size, learning_rate, num_train_iters, \
            w_content, w_color, w_texture, w_tv,\
            dped_dir, vgg_dir, eval_step


def process_test_model_args(arguments):

    phone = ""
    dped_dir = 'dped/'
    test_subset = "small"
    iteration = "all"
    resolution = "orig"
    use_gpu = "true"

    for args in arguments:

        if args.startswith("model"):
            phone = args.split("=")[1]

        if args.startswith("dped_dir"):
            dped_dir = args.split("=")[1]

        if args.startswith("test_subset"):
            test_subset = args.split("=")[1]

        if args.startswith("iteration"):
            iteration = args.split("=")[1]

        if args.startswith("resolution"):
            resolution = args.split("=")[1]

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

    if phone == "":
        print("\nPlease specify the model by running the script with the following parameter:\n")
        print("python test_model.py model={iphone,blackberry,sony,iphone_orig,blackberry_orig,sony_orig}\n")
        sys.exit()

    return phone, dped_dir, test_subset, iteration, resolution, use_gpu

def get_resolutions():

    # IMAGE_HEIGHT, IMAGE_WIDTH

    res_sizes = {}

    res_sizes["iphone"] = [1536, 2048]
    res_sizes["iphone_orig"] = [1536, 2048]
    res_sizes["blackberry"] = [1560, 2080]
    res_sizes["blackberry_orig"] = [1560, 2080]
    res_sizes["sony"] = [1944, 2592]
    res_sizes["sony_orig"] = [1944, 2592]
    res_sizes["high"] = [1260, 1680]
    res_sizes["medium"] = [1024, 1366]
    res_sizes["small"] = [768, 1024]
    res_sizes["tiny"] = [600, 800]

    return res_sizes

def get_specified_res(res_sizes, phone, resolution):

    if resolution == "orig":
        IMAGE_HEIGHT = res_sizes[phone][0]
        IMAGE_WIDTH = res_sizes[phone][1]
    else:
        IMAGE_HEIGHT = res_sizes[resolution][0]
        IMAGE_WIDTH = res_sizes[resolution][1]

    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3

    return IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE

def extract_crop(image, resolution, phone, res_sizes):

    if resolution == "orig":
        return image

    else:

        x_up = int((res_sizes[phone][1] - res_sizes[resolution][1]) / 2)
        y_up = int((res_sizes[phone][0] - res_sizes[resolution][0]) / 2)

        x_down = x_up + res_sizes[resolution][1]
        y_down = y_up + res_sizes[resolution][0]

        return image[y_up : y_down, x_up : x_down, :]

import argparse as Ap
def Complex_args():
    argp = Ap.ArgumentParser(
                             usage=None,
                             description=None,
                             epilog=None,
                             )
    argp.add_argument("-d", "--datadir", default=".", type=str,
                      help="Path to datasets directory.")
    argp.add_argument("-w", "--workdir", default=".", type=str,
                      help="Path to the workspace directory for this experiment.")
    argp.add_argument("-s", "--seed", default=0xe4223644e98b8e64, type=long,
                      help="Seed for PRNGs.")
    argp.add_argument("--summary", action="store_true",
                      help="""Print a summary of the network.""")
    argp.add_argument("--model", default="complex", type=str,
                      choices=["real", "complex"],
                      help="Model Selection.")
    argp.add_argument("--dataset", default="cifar10", type=str,
                      choices=["cifar10", "cifar100", "svhn"],
                      help="Dataset Selection.")
    argp.add_argument("--dropout", default=0, type=float,
                      help="Dropout probability.")
    argp.add_argument("-n", "--num-epochs", default=200, type=int,
                      help="Number of epochs")
    argp.add_argument("-b", "--batch-size", default=64, type=int,
                      help="Batch Size")
    argp.add_argument("--start-filter", "--sf", default=11, type=int,
                      help="Number of feature maps in starting stage")
    argp.add_argument("--num-blocks", "--nb", default=10, type=int,
                      help="Number of filters in initial block")
    argp.add_argument("--spectral-param", action="store_true",
                      help="""Use spectral parametrization.""")
    argp.add_argument("--spectral-pool-gamma", default=0.50, type=float,
                      help="""Use spectral pooling, preserving a fraction gamma of frequencies""")
    argp.add_argument("--spectral-pool-scheme", default="none", type=str,
                      choices=["none", "stagemiddle", "proj", "nodownsample"],
                      help="""Spectral pooling scheme""")
    argp.add_argument("--act", default="relu", type=str,
                      choices=["relu"],
                      help="Activation.")
    argp.add_argument("--aact", default="modrelu", type=str,
                      choices=["modrelu"],
                      help="Advanced Activation.")
    argp.add_argument("--no-validation", action="store_true",
                      help="Do not create a separate validation set.")
    argp.add_argument("--comp_init", default='complex_independent', type=str,
                      help="Initializer for the complex kernel.")

    optp = argp.add_argument_group("Optimizers", "Tunables for all optimizers")
    optp.add_argument("--optimizer", "--opt", default="nag", type=str,
                      choices=["sgd", "nag", "adam", "rmsprop"],
                      help="Optimizer selection.")
    optp.add_argument("--clipnorm", "--cn", default=1.0, type=float,
                      help="The norm of the gradient will be clipped at this magnitude.")
    optp.add_argument("--clipval", "--cv", default=1.0, type=float,
                      help="The values of the gradients will be individually clipped at this magnitude.")
    optp.add_argument("--l1", default=0, type=float,
                      help="L1 penalty.")
    optp.add_argument("--l2", default=0, type=float,
                      help="L2 penalty.")
    optp.add_argument("--lr", default=1e-3, type=float,
                      help="Master learning rate for optimizers.")
    optp.add_argument("--momentum", "--mom", default=0.9, type=float,
                      help="Momentum for optimizers supporting momentum.")
    optp.add_argument("--decay", default=0, type=float,
                      help="Learning rate decay for optimizers.")
    optp.add_argument("--schedule", default="default", type=str,
                      help="Learning rate schedule")
    optp = argp.add_argument_group("Adam", "Tunables for Adam optimizer")
    optp.add_argument("--beta1", default=0.9, type=float,
                      help="Beta1 for Adam.")
    optp.add_argument("--beta2", default=0.999, type=float,
                      help="Beta2 for Adam.")
    return argp





