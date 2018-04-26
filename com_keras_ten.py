# -*- coding: utf-8 -*
from keras.layers import Dropout, Dense
from keras.losses import categorical_crossentropy
from keras import backend as K
import tensorflow as tf
from keras.metrics import categorical_accuracy as accuracy
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

def getAccuracy(v_xs,v_ys):
    global y_pre
    y_v = tf.Session().run(y_pre,feed_dict={x:v_xs})
    correct_prediction = tf.equal(tf.arg_max(y_v,1),tf.arg_max(v_ys,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = tf.Session().run(accuracy,feed_dict={x:v_xs,y:v_ys})
    return result

sess = tf.Session()
K.set_session(sess)
img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

x = Dense(128, activation='relu')(img)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(10, activation='softmax')(x)

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)  # TensorFlow model
print('Initializing variables')
sess.run(tf.global_variables_initializer())
with sess.as_default():

    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1],
                                  K.learning_phase(): 1})  # K.learning_phase()为1，表示为训练模式， K.learning_phase()为0，表示为测试模式

acc_value = accuracy(labels, preds)  # keras model
with sess.as_default():
    print(acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels,
                                    K.learning_phase(): 0}))