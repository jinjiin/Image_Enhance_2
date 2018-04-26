# -*- coding: utf-8 -*
import tensorflow as tf
import complexnn
from   complexnn                             import ComplexBN,\
                                                    ComplexConv1D,\
                                                    ComplexConv2D,\
                                                    ComplexConv3D,\
                                                    ComplexDense,\
                                                    FFT,IFFT,FFT2,IFFT2,\
                                                    SpectralPooling1D,SpectralPooling2D
from complexnn import GetImag, GetReal
import h5py                                  as     H
import keras
from   keras.callbacks                       import Callback, ModelCheckpoint, LearningRateScheduler
from   keras.datasets                        import cifar10, cifar100
from   keras.initializers                    import Orthogonal
from   keras.layers                          import Layer, AveragePooling2D, AveragePooling3D, add, Add, concatenate, Concatenate, Input, Flatten, Dense, Convolution2D, BatchNormalization, Activation, Reshape, ConvLSTM2D, Conv2D
from   keras.models                          import Model, load_model, save_model
from   keras.optimizers                      import SGD, Adam, RMSprop
from   keras.preprocessing.image             import ImageDataGenerator
from   keras.regularizers                    import l2
from   keras.utils.np_utils                  import to_categorical
import keras.backend                         as     K
import keras.models                          as     KM
from   kerosene.datasets                     import svhn2
import logging                               as     L
import numpy                                 as     np
import os, pdb, socket, sys, time
import theano                                as     T
"""Learn initial imaginary component for input."""


def learnConcatRealImagBlock(I, filter_size, featmaps, stage, block, convArgs, bnArgs, d):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    O = BatchNormalization(name=bn_name_base + '2a', **bnArgs)(I)
    O = Activation(d.act)(O)
    O = Convolution2D(featmaps[0], filter_size,
                      name=conv_name_base + '2a',
                      padding='same',
                      kernel_initializer='he_normal',
                      use_bias=False,
                      kernel_regularizer=l2(0.0001))(O)

    O = BatchNormalization(name=bn_name_base + '2b', **bnArgs)(O)
    O = Activation(d.act)(O)
    O = Convolution2D(featmaps[1], filter_size,
                      name=conv_name_base + '2b',
                      padding='same',
                      kernel_initializer='he_normal',
                      use_bias=False,
                      kernel_regularizer=l2(0.0001))(O)

    return O


"""Get residual block."""


# O = getResidualBlock(O, 3, [64, 64], 2, '0', 'regular', convArgs, bnArgs, d)
def getResidualBlock(I, filter_size, featmaps, stage, block, shortcut, convArgs, bnArgs, d):
    activation = d.act
    drop_prob = d.dropout
    nb_fmaps1, nb_fmaps2 = featmaps
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    if K.image_data_format() == 'channels_first' and K.ndim(I) != 3:
        channel_axis = 1
    else:
        channel_axis = -1

    if d.model == "real":
        O = BatchNormalization(name=bn_name_base + '_2a', **bnArgs)(I)
    elif d.model == "complex":
        O = ComplexBN(name=bn_name_base + '_2a', **bnArgs)(I)
    O = Activation(activation)(O)

    if shortcut == 'regular' or d.spectral_pool_scheme == "nodownsample":
        if d.model == "real":
            O = Conv2D(nb_fmaps1, filter_size, name=conv_name_base + '2a', **convArgs)(O)
        elif d.model == "complex":
            O = ComplexConv2D(nb_fmaps1, filter_size, name=conv_name_base + '2a', **convArgs)(O)
    elif shortcut == 'projection':
        if d.spectral_pool_scheme == "proj":
            O = applySpectralPooling(O, d)
        if d.model == "real":
            O = Conv2D(nb_fmaps1, filter_size, name=conv_name_base + '2a', strides=(2, 2), **convArgs)(O)
        elif d.model == "complex":
            O = ComplexConv2D(nb_fmaps1, filter_size, name=conv_name_base + '2a', strides=(2, 2), **convArgs)(O)

    if d.model == "real":
        O = BatchNormalization(name=bn_name_base + '_2b', **bnArgs)(O)
        O = Activation(activation)(O)
        O = Conv2D(nb_fmaps2, filter_size, name=conv_name_base + '2b', **convArgs)(O)
    elif d.model == "complex":
        O = ComplexBN(name=bn_name_base + '_2b', **bnArgs)(O)
        O = Activation(activation)(O)
        O = ComplexConv2D(nb_fmaps2, filter_size, name=conv_name_base + '2b', **convArgs)(O)

    if shortcut == 'regular':
        O = Add()([O, I])
    elif shortcut == 'projection':
        if d.spectral_pool_scheme == "proj":
            I = applySpectralPooling(I, d)
        if d.model == "real":
            X = Conv2D(nb_fmaps2, (1, 1),
                       name=conv_name_base + '1',
                       strides=(2, 2) if d.spectral_pool_scheme != "nodownsample" else
                       (1, 1),
                       **convArgs)(I)
            O = Concatenate(channel_axis)([X, O])
        elif d.model == "complex":
            X = ComplexConv2D(nb_fmaps2, (1, 1),
                              name=conv_name_base + '1',
                              strides=(2, 2) if d.spectral_pool_scheme != "nodownsample" else
                              (1, 1),
                              **convArgs)(I)

            O_real = Concatenate(channel_axis)([GetReal()(X), GetReal()(O)])
            O_imag = Concatenate(channel_axis)([GetImag()(X), GetImag()(O)])
            O = Concatenate(1)([O_real, O_imag])

    return O


"""Perform spectral pooling on input."""


def applySpectralPooling(x, d):
    if d.spectral_pool_gamma > 0 and d.spectral_pool_scheme != "none":
        x = FFT2()(x)
        x = SpectralPooling2D(gamma=(d.spectral_pool_gamma,
                                     d.spectral_pool_gamma))(x)
        x = IFFT2()(x)
    return x

# d 是参数集
def comResnet(input, d):
    activation = d.act
    inputShape = (100, 100, 3)
    channelAxis = 1 if K.image_data_format() == 'channels_first' else -1
    convArgs = {
        "padding": "same",
        "use bias": False,
        "kernel_regularizer": l2(0.0001)
    }
    bnArgs = {
        "axis": channelAxis,
        "momentum": 0.9,
        "epsilon": 1e-04
    }
    convArgs.update({
        "spectral_parametrization": d.spectral_param,
        "kernel_initializer": d.comp_init
    })

    O = learnConcatRealImagBlock(input, (1, 1), (3, 3), 0, '0', convArgs, bnArgs, d)
    O = tf.concat(1, [input, O])
    O = ComplexConv2D(filters=64, kernel_size=9, name='conv1', **convArgs)(O)
    O = ComplexBN(name='bn conv1 2a', **bnArgs)(O)
    O = tf.nn.relu(O)

    # residual
    O = getResidualBlock(O, 3, [64, 64], 2, '0', 'regular', convArgs, bnArgs, d)
    O = getResidualBlock(O, 3, [64, 64], 2, '0', 'regular', convArgs, bnArgs, d)
    O = getResidualBlock(O, 3, [64, 64], 2, '0', 'regular', convArgs, bnArgs, d)
    O = getResidualBlock(O, 3, [64, 64], 2, '0', 'regular', convArgs, bnArgs, d)

    O = ComplexConv2D(filters=64, kernel_size=9, name='conv1', **convArgs)(O)
    O = ComplexBN(name='bn conv2 1a', **bnArgs)(O)
    O = tf.nn.relu(O)

    O = ComplexConv2D(filters=64, kernel_size=9, name='conv1', **convArgs)(O)
    O = ComplexBN(name='bn conv2 1a', **bnArgs)(O)
    O = tf.nn.relu(O)

    O = ComplexConv2D(filters=3, kernel_size=9, name='conv2', **convArgs)(O)
    O = ComplexBN(name='bn conv3 1a', **bnArgs)(O)
    O = tf.nn.tanh(O)
    return O

def resnet(input_image):
    with tf.variable_scope("generator"):
        W1 = weight_variable([9, 9, 3, 64], name="W1");
        b1 = bias_variable([64], name="b1");
        c1 = tf.nn.relu(conv2d(input_image, W1) + b1)

        # residual 1

        W2 = weight_variable([3, 3, 64, 64], name="W2");
        b2 = bias_variable([64], name="b2");
        c2 = tf.nn.relu(_instance_norm(conv2d(c1, W2) + b2))

        W3 = weight_variable([3, 3, 64, 64], name="W3");
        b3 = bias_variable([64], name="b3");
        c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3) + c1)

        # residual 2

        W4 = weight_variable([3, 3, 64, 64], name="W4");
        b4 = bias_variable([64], name="b4");
        c4 = tf.nn.relu(_instance_norm(conv2d(c3, W4) + b4))

        W5 = weight_variable([3, 3, 64, 64], name="W5");
        b5 = bias_variable([64], name="b5");
        c5 = tf.nn.relu(_instance_norm(conv2d(c4, W5) + b5) + c3)

        # residual 3

        W6 = weight_variable([3, 3, 64, 64], name="W6");
        b6 = bias_variable([64], name="b6");
        c6 = tf.nn.relu(_instance_norm(conv2d(c5, W6) + b6))

        W7 = weight_variable([3, 3, 64, 64], name="W7");
        b7 = bias_variable([64], name="b7");
        c7 = tf.nn.relu(_instance_norm(conv2d(c6, W7) + b7) + c5)

        # residual 4

        W8 = weight_variable([3, 3, 64, 64], name="W8");
        b8 = bias_variable([64], name="b8");
        c8 = tf.nn.relu(_instance_norm(conv2d(c7, W8) + b8))

        W9 = weight_variable([3, 3, 64, 64], name="W9");
        b9 = bias_variable([64], name="b9");
        c9 = tf.nn.relu(_instance_norm(conv2d(c8, W9) + b9) + c7)

        # Convolutional

        W10 = weight_variable([3, 3, 64, 64], name="W10");
        b10 = bias_variable([64], name="b10");
        c10 = tf.nn.relu(conv2d(c9, W10) + b10)

        W11 = weight_variable([3, 3, 64, 64], name="W11");
        b11 = bias_variable([64], name="b11");
        c11 = tf.nn.relu(conv2d(c10, W11) + b11)

        # Final

        W12 = weight_variable([9, 9, 64, 3], name="W12");
        b12 = bias_variable([3], name="b12");
        enhanced = tf.nn.tanh(conv2d(c11, W12) + b12) * 0.58 + 0.5

    return enhanced


def adversarial(image_):
    with tf.variable_scope("discriminator"):
        conv1 = _conv_layer(image_, 48, 11, 4, batch_nn=False)  # [100,100]->[25,25]
        conv2 = _conv_layer(conv1, 128, 5, 2)  # [25,25]->[13,13]
        conv3 = _conv_layer(conv2, 192, 3, 1)  # [13,13]->[13,13]
        conv4 = _conv_layer(conv3, 192, 3, 1)  # [13,13]->[13,13]
        conv5 = _conv_layer(conv4, 128, 3, 2)  # [13,13]->[7,7]

        flat_size = 128 * 7 * 7
        conv5_flat = tf.reshape(conv5, [-1, flat_size])

        W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))  # 1*1024

        fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)

        W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

        adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)

    return adv_out

def encode(input_image):
    with tf.variable_scope("encode"):
        W1 = weight_variable([5, 5, 3, 32], name="W1")
        b1 = bias_variable([32], name="b1")
        c1 = tf.nn.relu(conv2d(input_image, W1) + b1)

        p1 = max_pool_2x2(c1)

        W2 = weight_variable([5, 5, 3, 32], name="W2")
        b2 = bias_variable([32], name="b2")
        c2 = tf.nn.relu(conv2d(p1, W2) + b2)

        p2 = max_pool_2x2(c2)

        W_fc1 = weight_variable([25 * 25 * 32, 256]) # fc1的神经元个数是256
        b_fc1 = bias_variable([256])
        h_pool2_flat = tf.reshape(p2, [-1, 25 * 25 * 32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([256, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        return y_conv

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def _conv_layer(net, num_filters, filter_size, strides, batch_nn=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME') + bias
    net = leaky_relu(net)

    if batch_nn:
        net = _instance_norm(net)

    return net


def _instance_norm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)

    return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init
