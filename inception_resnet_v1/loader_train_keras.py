#!venv/bin/python
# -*- coding: utf-8 -*-

import os
import cv2
import pathlib
import numpy as np
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from inceptionv4 import _inceptionv4
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import warnings
warnings.filterwarnings('ignore')

# tf.enable_eager_execution()


__author__ = 'cristian'


def load_img(list_path):
    imgs = []
    for img in list_path:
        i = cv2.imread(img)
        i_re = cv2.resize(i, (224, 224))
        imgs.append(i_re)
    return np.asarray(imgs)


def preprocess_image(image):
    """

    :param image:
    :return:
    """
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [224, 224])
    # image = tf.cast(image, dtype=tf.float32)
    image /= 255.0  # normalize to [0,1] range
    # print(image)
    return image


def load_and_preprocess_image(path_img):
    """

    :param path_img: path to img
    :return:
    """
    image = tf.read_file(path_img)
    return preprocess_image(image)


def get_path_images(path):
    """
    :param path:  path to datasets ( data )
    :return: list of path to image with format: data/{age_label}/{gender_label/img.*}
    """
    images_list = []
    images = {}
    age__ = os.listdir(path)
    for age_ in age__:
        age = os.path.join(path, age_)
        gender_ = os.listdir(age)
        for gender in gender_:
            path_image_gender = os.path.join(age, gender)
            for image in os.listdir(path_image_gender):
                path_to_img = os.path.join(path_image_gender, image)
                images['path_image'] = path_to_img
                images['age'] = age_
                images['gender'] = gender
                images_list.append(images)
                images = {}
    return images_list


def get_label_from_path(path_find_label, list_of_dict_image):
    """
    :param path_find_label: path of image with format: data/{age_label}/{gender_label/img.*
    :param list_of_dict_image: output of 'get_path_images' function
    :return: label of gender and age
    """
    for dict_image in list_of_dict_image:
        path_dict = dict_image['path_image']
        if path_dict is path_find_label:
            age_label = dict_image['age']
            gender_label = dict_image['gender']

            return age_label, gender_label


def label_gender(x):
    """
    :param x: label gender
    :return: one hot label
    """
    return list(tf.keras.utils.to_categorical(x, num_classes=2))


def label_age(x):
    """

    :param x: label age
    :return: one hot label
    """
    return list(tf.keras.utils.to_categorical(x, num_classes=10))


def change_range(image, age_label, gender_label):
    """

    :param image:
    :param age_label:
    :param gender_label:
    :return: [-1,1]
    """
    return 2*image-1, age_label, gender_label


# set up data loader
# some config
batch_size = 64
path = '../data'
train_mode = 'generator'

model = _inceptionv4(input_shape=(224, 224, 3),
                     dropout_keep=0.8,
                     weigth=1,
                     include_top=True,
                     nb_class_age=10,
                     nb_class_gender=2)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, min_delta=0.0001, min_lr=0)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0)
tf_board = TensorBoard(log_dir='logs',
                       histogram_freq=0,
                       batch_size=32,
                       write_graph=True,
                       write_grads=True,
                       update_freq='epoch')

callbacks = [reduce_lr, early_stop, tf_board]
opt = Adam(lr=0.01)
model.compile(optimizer=opt,
              loss=['categorical_crossentropy', 'categorical_crossentropy'],
              metrics=['accuracy', 'accuracy'])

list_dict_images = get_path_images(path)

random.shuffle(list_dict_images)

all_path_image = [dict_path['path_image'] for dict_path in list_dict_images]
# print(all_path_image)
image_count = len(all_path_image)

all_gender_label = [label_gender(pathlib.Path(each_of_all_path_image).parent.name)
                    for each_of_all_path_image in all_path_image]

all_age_label = [label_age(pathlib.Path(each_of_all_path_image).parent.parent.name)
                 for each_of_all_path_image in all_path_image]

if train_mode == 'DataLoader':
    path_ds = tf.data.Dataset.from_tensor_slices(all_path_image)

    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    age_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_age_label, tf.int32))

    gender_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_gender_label, tf.int32))

    image_label_ds = tf.data.Dataset.zip((image_ds, age_label_ds, gender_label_ds))

    # Setting a shuffle buffer size as large as the dataset ensures that the data is
    # completely shuffled.
    train_ds = image_label_ds.take(int(0.7 * image_count))
    val_ds = image_label_ds.skip(int(0.7 * image_count))

    # set up foe train ds
    train_ds = train_ds.cache(filename='./cache.tf-data-train')
    train_ds = train_ds.shuffle(buffer_size=image_count)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # setup for val ds
    val_ds = val_ds.cache(filename='./cache.tf-data-val')
    val_ds = val_ds.shuffle(buffer_size=image_count)
    val_ds = val_ds.repeat()
    val_ds = val_ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    train_ds = train_ds.map(change_range)
    val_ds = val_ds.map(change_range)

    model.fit(train_ds, epochs=5, steps_per_epoch=50, validation_data=val_ds, validation_steps=30, callbacks=callbacks)

elif train_mode == 'generator':

    aug = ImageDataGenerator(rotation_range=0.1,
                             width_shift_range=0.5,
                             height_shift_range=0.5,
                             zoom_range=0.3,
                             fill_mode='nearest')
    X_train = load_img(all_path_image[:20])
    print(X_train.shape)

    model.fit_generator(aug.flow(X_train, [all_age_label[:20], all_gender_label[:20]], batch_size=5, shuffle=True),
                        steps_per_epoch=len(X_train) / 5,
                        epochs=15,
                        callbacks=callbacks)
