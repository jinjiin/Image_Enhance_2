# -*- coding: utf-8 -*
# python train_model.py model={iphone,sony,blackberry} dped_dir=dped/ vgg_dir=vgg_pretrained/imagenet-vgg-verydeep-19.mat
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

import tensorflow as tf
from scipy import misc
import numpy as np
import sys

from load_dataset import load_test_data, load_batch
from ssim import MultiScaleSSIM
import model
import utils
import vgg

# define size of training imahe pathes

PATHCH_WEIDTH = 100
PATHCH_HEIGHT = 100
PATCH_SIZE = 100 * 100 * 3

phone, batch_size, train_size, learning_rate, num_train_iters, \
w_content, w_color, w_texture, w_tv, \
dped_dir, vgg_dir, eval_step = utils.process_command_args(sys.argv)

np.random.seed(0)  # 设置随机种子，则每次运行相同的代码生成的随机数都是一样的

# load training and test data
print("Loading test data...")
test_data, test_answ = load_test_data(phone, dped_dir, PATCH_SIZE)
print("Test data has been loaded.")
print("Loading training data...")
train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
print("Training data has been loaded.")

TEST_SIZE = test_data.shape[0]
num_test_batches = int(TEST_SIZE / batch_size)

# define system architecture
with tf.Graph().as_default(), tf.Session() as sess:
    phone_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    phone_image = tf.reshape(phone_, [-1, PATHCH_HEIGHT, PATHCH_WEIDTH, 3])

    dslr_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    dslr_image = tf.reshape(dslr_, [-1, PATHCH_HEIGHT, PATHCH_WEIDTH, 3])

    adv_0 = tf.placeholder(tf.float32, [None, 1])
    adv_1 = tf.placeholder(tf.float32, [None, 1])



