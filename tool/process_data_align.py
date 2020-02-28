import os
import cv2
import numpy as np
from collections import Counter

import dlib
from tqdm import tqdm


def get_label(path_to_file_meta):
    """
    :param path_to_file_meta:
    :return: array of age, gender
    """
    list_age = []
    list_gender = []
    list_name_image = []
    with open(path_to_file_meta, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(" ")

            #add gender, age to list
            list_gender.append(line[1])
            list_name_image.append(line[0])

            age = int(line[0].split("A")[1].split(".")[0])
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

    return list_age, list_gender, list_name_image

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
        crop_face = input_image[face_rect.top()-15 :face_rect.top() + height + 5, face_rect.left()- 10:face_rect.left()+ width+ 10,:]
        faces.append(crop_face)
    return faces


def process_image_data_align(list_age, list_gender, list_name_image, path_origin_folder, path_save):
    """
    :param list_age:
    :param list_gender:
    :param list_name_image:
    :param path_origin_folder:
    :param path_save:
    :return: None
    """

    if not os.path.exists(path_save):
        os.mkdir(path_save)
    for i, image_name in tqdm(enumerate(list_name_image)):
        path_image = os.path.join(path_origin_folder, image_name)
        try:
            image = cv2.imread(path_image)
            crop_image = crop_front_face(image)[0]
        except:
            print("pass")
            continue

        #if age dir not exist, create
        age_dir_path = os.path.join(path_save, str(list_age[i]))
        if not os.path.exists(age_dir_path):
            os.mkdir(age_dir_path)

         #  if gender dir not exist, create
        gender_dir_path = os.path.join(age_dir_path, str(list_gender[i]))
        if not os.path.exists(gender_dir_path):
            os.mkdir(gender_dir_path)
        cv2.imwrite(os.path.join(gender_dir_path, str(i) + ".jpg"), crop_image)



if __name__ == '__main__':
    #process data train
    list_age_train, list_gender_train, list_name_image_train = get_label("/media/nghiapi/Disk1/Inter_tdt/aglined_datasets/labels/train.txt")
    path_origin_folder = "/media/nghiapi/Disk1/Inter_tdt/aglined_datasets/faces"
    path_save ="/media/nghiapi/Disk1/Inter_tdt/aglined_datasets/process_data/train"
    process_image_data_align(list_age_train, list_gender_train, list_name_image_train,path_origin_folder, path_save)

