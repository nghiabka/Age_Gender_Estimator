import os
import pathlib
from collections import Counter
from random import random

from dlib import image_gradients
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import numpy as np


def get_list_path(path):

    """
    :param path:  path to datasets ( data_agu )
    :return: list of path to image with format: data_agu/{age_label}/{gender_label/img.*}
    """
    list_path_train = []
    list_paths_test = []
    age_label = os.listdir(path)
    for age_ in age_label:
        list_paths_tmp = []
        list_label_gender = []
        age = os.path.join(path, age_)
        gender_label = os.listdir(age)
        for gender in gender_label:
            path_image_gender = os.path.join(age, gender)
            for image in os.listdir(path_image_gender):
                path_image = os.path.join(path_image_gender, image)
                list_paths_tmp.append(path_image)
                list_label_gender.append(gender)

        paths_train, paths_test, _,  _ = train_test_split(list_paths_tmp, list_label_gender, test_size=0.2,
                                                          shuffle=True, random_state=42)
        list_path_train += paths_train
        list_paths_test += paths_test
    return list_path_train, list_paths_test



# get path agu

def get_list_path2(path):
    """
    :param path:  path to datasets ( data )
    :return: list of path to image with format: data/{age_label}/{gender_label/img.*}
    """
    data_root = pathlib.Path(path)
    all_image_paths = data_root.glob('*/*/*')
    all_path_images = [str(path_img) for path_img in all_image_paths]
    return all_path_images

    if not os.path.exists(os.path.join(path_save, age)):
        os.mkdir(os.path.join(path_save, age))

    if not os.path.exists(os.path.join(path_save, age, gender)):
        os.mkdir(os.path.join(path_save, age, gender))

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)


    imageGen = datagen.flow(image, batch_size=1, save_to_dir=path_save,
                            save_prefix="{}/{}/{}".format(age, gender,name_image), save_format="jpg")

    total = 0
    for i in imageGen:
        # increment our counter
        total += 1

        # if we have reached the specified number of exampl"es, break from the loop
        if total == number_agu:
            break


def gen_data(list_paths, path_save):

    for path in list_paths:
        age_group = path.split("/")[-3]
        gender = path.split("/")[-2]
        age = age_group

        if age_group == "3" or age_group == "2" or age_group== "4" or age_group == "5":
            continue



