#!venv/bin/python
# -*- coding: utf-8 -*-

"""
Filename: config_label.py
Contains helper functions for copy datasets and fix path
"""

__author__ = 'cristian'

# import  necessary lib

import os
from shutil import copy
import glob
import ast

MALE = '111'
FEMALE = '112'

LABEL_MALE = '1'
LABEL_FEMALE = '0'


def fix_age_label(age):
    """
    labels = (1-12): 0,
             (13-18): 1,
             (19- 22): 2,
             (23-29): 3,
             (30-34): 4,
             (35-39): 5,
             (40-44): 6
             (45-50): 7
             (51-59): 8
             (60-..): 9
    :return: label of age corresponding
    """
    age = int(age)
    try:
        if 0 < age < 13:
            return str(0)

        elif 12 < age < 19:
            return str(1)

        elif 18 < age < 23:
            return str(2)

        elif 22 < age < 30:
            return str(3)

        elif 29 < age < 35:
            return str(4)

        elif 34 < age < 40:
            return str(5)

        elif 39 < age < 45:
            return str(6)
        elif 44 < age < 51:
            return str(7)

        elif 50 < age < 60:
            return str(8)

        else:
            return str(9)

    except Exception as e:
        raise Exception(" {}".format(e))


def move_images(path_to_move, list_images, label_age, label_gender):
    """
    param list_path_to_images(type: list):

    return: copy image to data dir
    """
    # print(os.path)
    path_move = os.path.join(path_to_move, label_age, label_gender)
    for image in list_images:
        try:
            # print(image)
            # print(path_move)
            copy(image, path_move)

            print("Copy file from {} --> {}".format(image, path_move))
        except Exception as ex:
            raise Exception("move image: {}".format(ex))
    return True


def copy_fad_to_data(data_dir, path_to_dataset):

    label_age = os.listdir(path_to_dataset)

    for _age in label_age:
        path_to_age = os.path.join(path_to_dataset, _age)
        genders = os.listdir(path_to_age)
        label_age = fix_age_label(_age)
        for gender in genders:
            if gender == MALE:
                path_to_images = os.path.join(path_to_age, MALE)
                images = glob.glob(path_to_images+'/*.jpg')
                move_images(data_dir, images, label_age, LABEL_MALE)

            elif gender == FEMALE:
                path_to_images = os.path.join(path_to_age, FEMALE)
                images = glob.glob(path_to_images + '/*.jpg')
                move_images(data_dir, images, label_age, LABEL_FEMALE)
    return True


def copy_aglined_to_data(data_dir, path_file_txt):

    txt_data = open(path_file_txt, 'r').read()
    dict_image = txt_data.split('\n')
    print(data_dir)
    for _dict in dict_image:

        if _dict != '':
            dt = ast.literal_eval(_dict)
            path_image_drive = dt['image']
            _age = dt['age']
            age_gr = fix_age_label(_age)
            gender = dt['label']
            idx = path_image_drive.find('ag')
            path_image = path_image_drive[idx:]
            path_to_images = os.path.join('datasets', path_image)
            path_move_data = os.path.join('data', age_gr, gender)

            try:
                copy(path_to_images, path_move_data)

                print("Copy file from {} --> {}".format(path_to_images, path_move_data))
            except Exception as ex:
                raise Exception("move image: {}".format(ex))


if __name__ == '__main__':

    path_dataset = 'datasets/AFAD-Full'
    path_data = 'data'
    copy_fad_to_data(path_data, path_dataset)
    print("===================================================================================\n")
    copy_aglined_to_data(path_data, 'datasets/aglined_datasets/data_aglined.txt')




