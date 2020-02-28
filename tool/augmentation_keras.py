import os
import glob
from collections import Counter
from tqdm import tqdm
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


def data_generator(image_path, path_save, age, gender, number_agu):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = np.expand_dims(image, axis=0)

    # # create datagen2
    name_image = image_path.split("/")[-1].split("0")[0]

    if not os.path.exists(path_save):
        os.mkdir(path_save)

    if not os.path.exists(os.path.join(path_save, age)):
        os.mkdir(os.path.join(path_save, age))

    if not os.path.exists(os.path.join(path_save, age, gender)):
        os.mkdir(os.path.join(path_save, age, gender))

    # datagen = ImageDataGenerator(
    #     featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True)
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=25,
        width_shift_range=0.17,
        height_shift_range=0.18,
        brightness_range=[0.2,1.3],
        horizontal_flip=True,
        fill_mode="nearest")

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
    for path in tqdm(list_paths):
        age_group = path.split("/")[-3]
        gender = path.split("/")[-2]

        if age_group == "7":
            data_generator(path, path_save, age_group, gender, 1)
        elif age_group == "6":
            data_generator(path, path_save, age_group, gender, 1)
        elif age_group == "4":
            data_generator(path, path_save, age_group, gender, 1)
        elif age_group == "5":
            data_generator(path, path_save, age_group, gender, 1)
        elif age_group == "7":
            data_generator(path, path_save, age_group, gender, 1)
        elif age_group == "8":
            data_generator(path, path_save, age_group, gender, 1)
        elif age_group == "9":
            data_generator(path, path_save, age_group, gender, 1)


if __name__ == '__main__':
    list_path_train, list_path_test = get_list_path("/mnt/Disk1/Inter_tdt/data")
    print("train....")

    label_train_age = []
    label_train_gender = []
    for path in list_path_train:
        label_train_age.append(path.split("/")[-3])
        label_train_gender.append(path.split("/")[-2])
    print(Counter(label_train_gender))
    print(Counter(label_train_age))





