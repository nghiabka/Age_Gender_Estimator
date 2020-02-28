import os
import pickle

import cv2
import numpy as np
import random
import keras
import argparse
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from data_generator import DataGenerator
from training import get_list_path, remove_path_incorrect
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as K


K.tensorflow_backend._get_available_gpus()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def load_model(model_path):

    json_file = open(model_path + 'model_v4_dropout/inceptionv4.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(model_path + "model_v4_dropout/inceptionv4.h5")
    print("Loaded model from disk")

    return model


def save_model(model, history):

    # save model
    model_json = model.to_json()
    with open("model_v4_phase2/inceptionv4.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_v4_phase2/inceptionv4.h5")
    print("Saved model to disk")

    #save history train
    with open('model_v4_phase2/history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def train(model_path, list_paths_train, list_paths_test):

    model = load_model(model_path)
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss={'age_output': 'categorical_crossentropy',
                                       'gender_output': 'categorical_crossentropy'},
                  loss_weights={'age_output': 2.,
                                'gender_output': 1.},
                  metrics={'age_output': 'accuracy',
                           'gender_output': 'accuracy'})

    # Parameters
    params = {'batch_size': 64,
              'n_channels': 1,
              # 'n_classes': 2,
              'shuffle': True}
    train_generator = DataGenerator(list_paths_train, **params)
    test_generator = DataGenerator(list_paths_test, **params)
    checkpoint = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0,
                            save_best_only=True,
                            save_weights_only=False,
                            period=1)
    tf_board = keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0,
                                           batch_size=64, write_graph=True, write_grads=True,
                                           write_images=True,
                                           update_freq='epoch')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.001,
                                  min_lr=0)
    # fit
    history = model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        use_multiprocessing=True,
                        workers=8,
                        verbose=1,
                        callbacks=[reduce_lr, tf_board, checkpoint],
                        epochs=25)

    return model, history


if __name__ == '__main__':

    arg = argparse.ArgumentParser()
    arg.add_argument("-data_train", "--train", default='../data_finetune/data_train', help="path to data train")
    arg.add_argument("-data_test", "--test", default='../data_finetune/data_test', help="path to data test")
    args = vars(arg.parse_args())

    path_ds_train = glob.glob(args['train'] + '/*')
    path_ds_test = glob.glob(args['test'] + '/*')

    all_train_paths = get_list_path(path_ds_train)
    # data test
    all_test_paths = get_list_path(path_ds_test)
    # extend data training
    # shuffle data paths
    random.shuffle(all_train_paths)
    random.shuffle(all_test_paths)

    #  filtet path to remove path incorrect
    list_paths_train = remove_path_incorrect(all_train_paths)
    list_paths_test = remove_path_incorrect(all_test_paths)

    model_path = './model_v4_phase2/'
    train(model_path, list_paths_train, list_paths_test)

    