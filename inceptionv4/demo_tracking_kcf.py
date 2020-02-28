import numpy as np
import cv2
import sys
import time
import dlib
from tensorflow.python.keras.models import model_from_json

label_gender = {
    "0": "F",
    "1": "M"

}
label_age = {
        '0': '(1-12)',
        '1': '(13-18)',
        '2': '(19- 22)',
        '3': '(23-29)',
        '4': '(30-34)',
        '5': '(35-39)',
        '6': '(40-44)',
        '7': '(45-50)',
        '8': '(51-59)',
        '9': '(>60)'
}

def load_model(model_path):
    """
    :param model_path: path to folder'smodel
    :return: model
    """

    json_file = open(model_path + '/inceptionv4.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(model_path + "/inceptionv4.h5")
    print("Loaded model from disk")

    return model


def predict(model, image):
    """
    :param model:
    :param image:
    :return: gender, age result
    """
    image = cv2.resize(image, (224, 224))
    image = image/255
    image = image.reshape(1, 224, 224, 3)
    gender_arr, age_arr = model.predict(image)
    gender = np.argmax(gender_arr[0])
    age = np.argmax(age_arr[0])
    return gender, age


def detect_face():
    """detect face"""
    cap = cv2.VideoCapture(2)
    cv2.destroyAllWindows()
    while (1):
        ret, frame = cap.read()
        frame = np.array(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hog_face_detector = dlib.get_frontal_face_detector()
        faces = hog_face_detector(gray, 0)
        # print(len(faces))
        cv2.imshow('output', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

        facem = []
        if len(faces) != 0:

            # face1 is the image of extracted face
            for i, d in enumerate(faces):
                x, y, x2, y2, w, h = d.left(), d.top(), d.right() + 1, \
                                     d.bottom() + 1, d.width(), d.height()
                face1 = frame[y:y + h, x:x + w]
                facem.append(face1)

            # print("facem len:", len(facem))
            # cv2.imshow('face1', facem[0])
            break
    return faces, frame

def init_tracking(faces, frame):
    """init tracking"""
    trackerm = []
    okm = []
    for i, d in enumerate(faces):
        x, y, x2, y2, w, h = d.left(), d.top(), d.right() + 1, \
                             d.bottom() + 1, d.width(), d.height()
        tracker = cv2.TrackerMedianFlow_create()
        bbox = (x, y, w, h)
        ok = tracker.init(frame, bbox)
        trackerm.append(tracker)
        okm.append(ok)
    return trackerm, okm

def tracking(model):
    """tracking"""
    faces, frame = detect_face()
    trackerm, okm = init_tracking(faces, frame)
    fps = 0
    fps_counter = 0
    timer = time.time()
    frames = 0
    refind_timer = 0
    refind_flag = 0
    tmp_frame = 0
    tmp_faces = 0
    num_person = len(faces)
    cap = cv2.VideoCapture(2)

    while True:
        # Read a new frame
        # tmp_faces, tmp_frame = detect_face()
        ok, frame = cap.read(2)

        if not ok:
            break
        # Update tracker
        # ok, bbox = tracker.update(frame)
        if refind_timer == 3:
            refind_timer = 0
            frame = np.array(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hog_face_detector = dlib.get_frontal_face_detector()
            tmp_faces = hog_face_detector(gray, 0)
            num_person = len(tmp_faces)

        okm = []
        bboxm = []
        for tracker in trackerm:
            ok, bbox = tracker.update(frame)
            okm.append(ok)
            bboxm.append(bbox)

        for i, ok in enumerate(okm):
            if not ok:
                continue
            box_x, box_y, box_w, box_h = bboxm[i]
            p1 = (int(box_x), int(box_y))
            p2 = (int(box_x + box_w), int(box_y + box_h))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
            cv2.putText(frame, str(i), (p1[0] + 20, p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        #exit if esc press
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            cap.release()
            break
        # timer,fps, refind:finds the face and features again
        fps_counter = fps_counter + 1
        if (time.time() - timer > 1):
            # print(timer, ":", fps_counter)
            fps = fps_counter
            fps_counter = 0
            timer = time.time()
            refind_timer = refind_timer + 1

        # frames, refind:finds the face and features again
        frames = frames + 1
        if (frames > fps):
            frames = 0
            # refind()
        cv2.putText(frame, "Fps:" + str(fps) +" num_person:"+str(num_person), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        # output
        cv2.imshow('output', frame)

if __name__ == '__main__':
    # model = load_model("./model_v4_phase2")
    tracking(model=None)