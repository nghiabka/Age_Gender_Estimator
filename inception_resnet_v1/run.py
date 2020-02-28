import os
import cv2
import dlib
from datetime import date
import time
import csv 
import numpy as np
import argparse
import inception_resnet_v1
import tensorflow as tf
from imutils.face_utils import FaceAligner


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.18-4.06.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def gen_csv(path):

    row = ['gender', 'age', "time"]
    with open(path, "a") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)


def save_csv(gender, age, path):

    FEMALE = 0
    with open(path, 'a') as writeFile:
        writer = csv.writer(writeFile)
        for i in range(len(gender)):
            age = str(int(age[i]))
            sex = 'FEMALE' if gender[i] == FEMALE else 'MALE'
            t = time.strftime('%H:%M:%S')
            writer.writerow([sex, age, t])


def main(sess,age,gender,train_mode,images_pl):
    args = get_args()
    depth = args.depth
    k = args.width

    # for face detection
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=160)

    # load model and weights
    img_size = 160

    # capture video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start_time = time.time()
    while True:
        # get video frame
        ret, img = cap.read()

        if not ret:
            print("error: failed to capture image")
            return -1

        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # input_img = cv2.flip(input_img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - 0.4 * w), 0)
            yw1 = max(int(y1 - 0.4 * h), 0)
            xw2 = min(int(x2 + 0.4 * w), img_w - 1)
            yw2 = min(int(y2 + 0.4 * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            faces[i, :, :, :] = fa.align(input_img, gray, detected[i])
        if len(detected) > 0:
            # predict ages and genders of the detected faces
            ages, genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})
            # print("estimates",ages, genders)
            time_ = (time.time()-start_time)
            if time_ > 30:
                path_out = "output"+"/{}.csv".format(date.today())
                if not os.path.exists(path_out):
                    gen_csv(path_out)
                save_csv(genders, ages, path_out)
                # dump_csv(genders, ages, path_out)
                print("dump....")
                start_time = time.time()
            # start_time = time.time()
            # print("estimates",ages, genders)

        # draw results
        for i, d in enumerate(detected):
            label = "{}, {}".format(int(ages[i]), "F" if genders[i] == 0 else "M")
            draw_label(img, (d.left(), d.top()), label)

        cv2.imshow("result", img)
        # cv2.imwrite("output/result{}.jpg".format(count), img)
        key = cv2.waitKey(1)

        if key == 27:
            break
        

def load_network(model_path):

    sess = tf.Session()
    images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
    images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
    train_mode = tf.placeholder(tf.bool)
    age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                 phase_train=train_mode,
                                                                 weight_decay=1e-5)
    gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
    age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
    age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restore model!")
    else:
        pass
    return sess, age, gender, train_mode, images_pl


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", "--M", default="./models", type=str, help="Model Path")
        args = parser.parse_args()
        sess, age, gender, train_mode, images_pl = load_network(args.model_path)
        # print(age, gender)
        main(sess, age, gender, train_mode, images_pl)
    except Exception as err:
        with open('error.log', 'w') as f:
            f.write(str(err))
