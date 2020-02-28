import os
import cv2
import dlib
import glob
import pathlib
import random
import argparse
<<<<<<< HEAD
=======
import tensorflow as tf
>>>>>>> 139ae1321e519e445118a9361e46b6c2197d51f5
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from collections import Counter
from keras import backend as K
from keras.models import model_from_json
from imutils.face_utils import FaceAligner

<<<<<<< HEAD
K.tensorflow_backend._get_available_gpus()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

=======
>>>>>>> 139ae1321e519e445118a9361e46b6c2197d51f5
# load model
def load_model(model_path):
    """ model path phase2"""
    json_file = open(model_path + '/inceptionv4.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(model_path + "/inceptionv4.h5")
    print("Loaded model from disk")

    return model


# thong ke tuoi trong tap test
def expoler_age(path_folder):
    list_age = []
    with open(path_folder, "r") as f:
        for line in f:
            age = int(line.strip())
            if age <= 12:
                age = "0"
            elif 13 <= age and age <= 18:
                age = "1"
            elif 19 <= age <= 22:
                age = "2"
            elif 23 <= age <= 29:
                age = "3"
            elif 30 <= age <= 34:
                age = "4"
            elif 35 <= age <= 39:
                age = "5"
            elif 40 <= age <= 44:
                age = "6"
            elif 45 <= age <= 50:
                age = "7"
            elif 51 <= age <= 59:
                age = "8"
            elif age >= 60:
                age = "9"
            list_age.append(age)

    return list_age


def crop_front_face(input_image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detected = detector(input_image, 1)

    img_h, img_w, _ = np.shape(input_image)
    faces = []
    for i, face_rect in enumerate(detected):
        width = face_rect.right() - face_rect.left()
        height = face_rect.bottom() - face_rect.top()

        #if cropface size < 64 , pass
        if width < 64 or height < 64:
            return None
        crop_face = input_image[face_rect.top()-15 :face_rect.top() + height, face_rect.left():face_rect.left()+width,:]
        faces.append(crop_face)
    return faces


def set_gender(path_folder_origin, path_foler_save, list_age, model, mode):
    """
    set label  gender for data megaage using model does not label gender
    :param path_folder_origin:
    :param path_foler_save:
    :param list_age:
    :param model: model is used to predict gender, this model is trained in phase2
    :return:
    """
    if not os.path.exists(path_foler_save):
        os.mkdir(path_foler_save)

    for i in tqdm(range(len(list_age))):
        try:
            img_path = os.path.join(path_folder_origin, str(i + 1) + ".jpg")
            img = cv2.imread(img_path)

            # image origin
            img_origin = img

            img = img / 255
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
        except:
            print("pass")
            continue
        if img is None:
            continue
        gender, _ = model.predict(img)
        gender = np.argmax(gender)

        # if age dir not exist , create
        age_dir_path = os.path.join(path_foler_save, str(list_age[i]))
        print(age_dir_path)
        if not os.path.exists(age_dir_path):
            os.mkdir(age_dir_path)

        #  if gender dir not exist, create
        gender_dir_path = os.path.join(age_dir_path, str(gender))
        print(gender_dir_path)
        if not os.path.exists(gender_dir_path):
            os.mkdir(gender_dir_path)
        #save image
        try:
            face_crop = crop_front_face(img_origin)[0]
        except:
            continue

         #save crop face
        cv2.imwrite(os.path.join(gender_dir_path, str(i) + ".jpg"), face_crop)

<<<<<<< HEAD
def crop_face_labeled(path_data):

    all_path_images = glob.glob(path_data+'/*/*/*/*.jpg')
    print(all_path_images)
=======

def crop_face_labeled(path_data: str):

    all_path_images = glob.glob(path_data+'*/*/*.jpg')
>>>>>>> 139ae1321e519e445118a9361e46b6c2197d51f5
    count = 0
    for path in tqdm(all_path_images):
        root_path = pathlib.Path(path)
        path_image = root_path.as_posix()
        name_img = root_path.name
        gender_label = root_path.parent.name
        age_label = root_path.parent.parent.name

        try:
            img_origin = cv2.imread(path_image)
            face_crop = crop_front_face(img_origin)[0]
            os.remove(path_image)
            cv2.imwrite(path_image, face_crop)
            count+=1
        except:
            continue
    print("number of images: {}".format(len(all_path_images)))
    print("number of images crop: {}".format(count))


def test():
    path_folder = "/media/nghiapi/Disk1/Inter_tdt/megaage_asian/test"
    path_save = "/home/nghiapi/Desktop/test_crop"

    list_image = os.listdir(path_folder)
    # ramdom choice
    sampling_image = np.random.choice(list_image, 200)
    for i, sample in enumerate(sampling_image):
        path_image = os.path.join(path_folder, sample)
        image = cv2.imread(path_image)
        try:
            face = crop_front_face(image)[0]
        except:
            continue
        cv2.imwrite(path_save + "/{}.jpg".format(str(i)), face)


if __name__ == '__main__':
    # arg = argparse.ArgumentParser()
    # arg.add_argument("-data", "--data", default=None, help="path to data")
    # args = vars(arg.parse_args())

    # set_gender("/media/nghiapi/Disk1/Inter_tdt/megaage_asian/train", )
    # list_age_test_megaage = expoler_age("/media/nghiapi/Disk1/Inter_tdt/megaage_asian/list/test_age.txt")
    # model = load_model("/media/nghiapi/Disk1/Inter_tdt/megaage_asian/model_v4_dropout")
    # set_gender(path_folder_origin="/media/nghiapi/Disk1/Inter_tdt/megaage_asian/test",
    #            path_foler_save="/media/nghiapi/Disk1/Inter_tdt/megaage_asian/data_process",
    #            model=model,
    #            list_age=list_age_test_megaage)

<<<<<<< HEAD
    crop_face_labeled('../data_megaage')
=======
    crop_face_labeled(path_data='../data_finetune/augment/')
>>>>>>> 139ae1321e519e445118a9361e46b6c2197d51f5






