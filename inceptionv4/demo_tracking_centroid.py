'''
    Author: Guido Diepen <gdiepen@deloitte.nl>
'''

#Import the OpenCV and dlib libraries
import os
from datetime import date

import cv2
import dlib
import csv
import numpy as np
from keras.models import model_from_json
import threading
import time
from collections import Counter


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


def predict(model, image, image_size):
    """
    :param model:
    :param image:
    :return: gender, age result
    """
    image = cv2.resize(image, (image_size, image_size))
    image = image/255
    image = image.reshape(1, image_size, image_size, 3)
    gender_arr, age_arr = model.predict(image)
    gender = np.argmax(gender_arr[0])
    age = np.argmax(age_arr[0])
    return gender, age


def gen_csv(path):
    """generate csv file"""
    row = ['gender', 'age', "time"]
    with open(path, "a", newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)


def save_csv( gender, age, path):
    """save to csv"""
    with open(path, 'a', newline='') as writeFile:
        writer = csv.writer(writeFile)
        t = time.strftime('%H:%M:%S')
        writer.writerow([label_gender[str(gender)], label_age[str(age)], t])


#Initialize a face detect using dlib

hog_face_detector = dlib.get_frontal_face_detector()

#The deisred output width and height
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600


#We are not doing really face recognition
def doRecognizePerson(faceNames,result_age, fid):
    faceNames[fid] = str(fid)
    result_age.append([fid])



def detectAndTrackMultipleFaces(model, imagesize):
    #Open the first webcame device
    capture = cv2.VideoCapture(0)

    #Create two opencv named windows
    # cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    #Position the windows next to eachother
    # cv2.moveWindow("base-image", 0, 100)
    cv2.moveWindow("result-image", 400, 100)

    #Start the window thread for the two windows we are using
    cv2.startWindowThread()

    #The color of the rectangle we draw around the face
    rectangleColor = (0, 165, 25)

    #variables holding the current frame number and the current faceid
    frameCounter = 0
    currentFaceID = 0
    flag_to_write_csv = 0
    result_age = []
    #Variables holding the correlation trackers and the name per faceid
    faceTrackers = {}
    faceNames = {}
    try:
        time_start = time.time()

        while True:
            #Retrieve the latest image from the webcam
            rc,fullSizeBaseImage = capture.read()

            #Resize the image to 320x240
            fullSizeBaseImage = cv2.flip(fullSizeBaseImage, 1)
            baseImage = fullSizeBaseImage;
            # baseImage = cv2.resize(fullSizeBaseImage, (1400, 1400));

            #Check if a key was pressed and if it was Q, then break
            #from the infinite loop
            pressedKey = cv2.waitKey(2)

            if pressedKey == ord('Q'):
                break

            #Result image is the image we will show the user, which is a
            #combination of the original image from the webcam and the
            #overlayed rectangle for the largest face
            resultImage = baseImage.copy()

            #Increase the framecounter
            frameCounter += 1

            #Update all the trackers and remove the onges for which the update
            #indicated the quality was not good enough
            fidsToDelete = []
            for fid in faceTrackers.keys():
                trackingQuality = faceTrackers[fid].update( baseImage )
                #If the tracking quality is good enough, we must delete
                if trackingQuality < 10:
                    fidsToDelete.append(fid)
            num_person = len(faceTrackers) - len(fidsToDelete)
            for fid in fidsToDelete:
                faceTrackers.pop(fid, None)

            #Every 10 frames, we will have to determine which faces
            #are present in the frame
            if (frameCounter % 10) == 0:

                gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2RGB)
                faces = hog_face_detector(gray, 1)
                for i, d in enumerate(faces):

                    x, y, x2, y2, w, h = d.left(), d.top(), d.right() + 1,\
                                           d.bottom() + 1, d.width(), d.height()

                    #calculate the centerpoint
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h

                    matchedFid = None;

                    #Now loop over all the trackers and check if the
                    #centerpoint of the face is within the box of a tracker
                    for fid in faceTrackers.keys():
                        tracked_position = faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())

                        #calculate th   e centerpoint
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        #check if the centerpoint of the face is within the
                        #rectangleof a tracker region. Also, the centerpoint
                        #of the tracker region must be within the region
                        #detected as a face. If both of these conditions hold
                        #we have a match
                        if ((t_x <= x_bar <= (t_x + t_w)) and
                            (t_y <= y_bar <= (t_y + t_h)) and
                            (x <= t_x_bar <= (x + w)) and
                            (y <= t_y_bar <= (y + h))):
                            matchedFid = fid

                    #If no matched fid, then we have to create a new tracker
                    if matchedFid is None:
                        # print("Creating new tracker " + str(currentFceID))

                        #Create and store the tracker
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(baseImage,
                                            dlib.rectangle(x, y, x+w, y+h))

                        faceTrackers[currentFaceID] = tracker

                        #Start a new thread that is used to simulate
                        #face recognition. This is not yet implemented in this
                        #version :)
                        t = threading.Thread(target=doRecognizePerson,
                                               args=(faceNames, result_age, currentFaceID))
                        t.start()

                        #Increase the currentFaceID counter
                        currentFaceID += 1

            #tracking
            time_wirte_csv = time.time()
            if time_wirte_csv - time_start > 3:
                flag_to_write_csv = 1
            for fid in faceTrackers.keys():
                tracked_position = faceTrackers[fid].get_position()
                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())
                image = baseImage[t_y -int(t_h/2):t_y+t_h-5, t_x-5:t_x+ t_w+5, :]
                try:
                    # if flag_to_write_csv:
                    cv2.imshow("image", image)
                    gender, age = predict(model, image, imagesize)
                except:
                    continue

                cv2.rectangle(resultImage, (t_x, t_y),
                                        (t_x + t_w, t_y + t_h),
                                        rectangleColor, 2)

                if fid in faceNames.keys():
                    result_age[fid].append(age)
                    if len(result_age[fid]) > 10:
                        age = Counter(result_age[fid][-10:-1]).most_common()[0][0]

                    if flag_to_write_csv:
                        # print(Counter(result_age[fid]))
                        # print(result_age)
                        path_out = "output" + "/{}.csv".format(date.today())
                        if not os.path.exists(path_out):
                            gen_csv(path_out)
                        save_csv(gender, age, path_out)

                    cv2.putText(resultImage," " + label_gender[str(gender)] + ",{}".format(
                        label_age[str(age)]),
                                (int(t_x + t_w/5), int(t_y)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 1)

                else:
                    cv2.putText(resultImage, "Detecting..." ,
                                (int(t_x + t_w/2), int(t_y)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)

            if flag_to_write_csv:
                    flag_to_write_csv = 0
                    time_start = time.time()

            largeResult = cv2.resize(resultImage,
                                     (OUTPUT_SIZE_WIDTH,OUTPUT_SIZE_HEIGHT))

            #Finally, we want to show the images on the screen
            # cv2.imshow("base-image", baseImage)
            cv2.imshow("result-image", largeResult)


    except KeyboardInterrupt as e:
        pass

    #Destroy any OpenCV windows and exit the application
    cv2.destroyAllWindows()
    exit(0)


if __name__ == '__main__':
    try:
        # create output folder and logs folder
        if not os.path.exists(("./output")):
            os.mkdir("./output")

        if not os.path.exists("./logs"):
            os.mkdir("./logs")
         #load model
        model = load_model("./model_v4_160")
        detectAndTrackMultipleFaces(model, imagesize=160)
    except Exception as err:
        with open('logs/error.log', 'w+') as f:
            f.write(str(err))
            f.write("\n")
