#!venv/bin/python
# -*- coding: utf-8 -*-

"""
Filename: data_utils.py
Contains helper functions for handling data
"""

__author__ = 'cristian'

import cv2
import glob
import numpy as np
import argparse
import run


def main(sess_, _age, sex, is_training, image, path_to_image):

    # load model and weights
    img_size = 160
    img = cv2.imread(path_to_image)
    img = cv2.resize(img, (img_size, img_size))
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = np.expand_dims(input_img, axis=0)
    # predict ages and genders of the detected faces
    ages, genders = sess_.run([_age, sex], feed_dict={face: image, is_training: False})
    print(path_image, ages, genders)


if __name__ == '__main__':

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("--model_path", "--M", default="./models", type=str, help="Model Path")

        parser.add_argument("--dir_images", "-DIR", default=None,
                            required=True, type=str,
                            help="path to images for predict")

        args = parser.parse_args()

        list_image = glob.glob(args.dir_images + '/*.jpg')

        sess, age, gender, train_mode, images_pl = run.load_network(args.model_path)

        for path_image in list_image:
            main(sess, age, gender, train_mode, images_pl, path_image)
    except Exception as err:
        with open('error_predict.log', 'w') as f:
            f.write(str(err))
