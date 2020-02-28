#!venv/bin/python
# -*- coding: utf-8 -*-

# import lib
import cv2
import glob
import numpy as np
import inception_resnet_v1
import tensorflow as tf
import tensorflow.contrib.slim as slim

__author__ = 'cristian'

#config paramss
nrof_age_classes = 10
nrof_gender_classes = 2
df_graph = tf.Graph()

with df_graph.as_default():
    image_batch = tf.identity(image_batch, 'input')
    label_age_batch = tf.identity(label_age_batch, 'label_age_batch')
    label_gender_batch = tf.identity(label_gender_batch, 'label_gender_batch')

with df_graph.as_default():

    phase_train_placeholder = tf.placeholder(tf.bool, name='Phase_train')
    age, gender, _ = inception_resnet_v1.inference(image_batch, keep_probability=0.8,
                                                   phase_train=True, bottleneck_layer_size=128)
    fn_age = slim.fully_connected(age, nrof_age_classes, activation_fn=None,
                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                  weights_regularizer=slim.l2_regularizer(1e-5),
                                  scopr='fn_age')
    fn_gender = slim.fully_connected(gender, nrof_gender_classes, activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     weights_regularizer=slim.l2_regularizer(1e-5),
                                     scopr='fn_gender')

with df_graph.as_default():

    cross_entropy_age = tf.nn.softmax_cross_entropy_with_logits(labels=label_age_batch,
                                                                logits=fn_age,
                                                                name='cross_entropy_age')
    cross_entropy_gender = tf.nn.softmax_cross_entropy_with_logits(labels=label_gender_batch,
                                                                   logits=fn_gender,
                                                                   name='cross_entropy_gender')
    tf.add_to_collection('age_losses', cross_entropy_age)
    tf.add_to_collection('gender_losses', cross_entropy_gender)

    correct_age_prediction = tf.cast(
        tf.equal(tf.argmax(fn_age, 1), tf.cast(label_age_batch, tf.int32)), tf.float32)

    correct_gender_prediction = tf.cast(
        tf.equal(tf.argmax(fn_gender, 1), tf.cast(label_gender_batch, tf.int32)), tf.float32)
