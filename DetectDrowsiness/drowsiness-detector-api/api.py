from imutils.video import VideoStream
from flask import Response
from flask import Flask, jsonify
from flask import render_template
from flask_session import Session
from flask_cors import CORS
import threading
import argparse
import datetime
import imutils
import time
import cv2
import glob
import numpy as np
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,MaxPooling2D)
from collections import deque
from sequenceextractor import SequenceExtractor
from modelloader import ResearchModels

TESTTING_IMAGE_PATH = "./Output/"
SEQ = 50
WEIGHT_PATH = "./model/rm.model.h5"
GRAPH_PATH = "./model/output_graph.pb"
SEQUENCE_PATH = './data/sequences/testing/'

outputFrame = None
lock = threading.Lock()
data_lock = threading.Lock()
lock2 = threading.Lock()
test_data = []
data_flag = 0
extract_flag = 0
app = Flask(__name__)
SESSION_TYPE = 'filesystem'
Session(app)


# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})


vs = VideoStream(src=0).start()
time.sleep(2.0)

protoPath = "./utils/deploy.prototxt.txt"
modelPath = "./utils/res10_300x300_ssd_iter_140000.caffemodel"
confidenceBoundary = 0.95
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
outputFolder = "./Output/"

model = ResearchModels(2, 50, input_shape=(50, 2048)).loadModelFromPath(WEIGHT_PATH)

extractor = SequenceExtractor(SEQ)

def load_one_sequence(path):
    if os.path.isfile(path):
        return np.load(path)
    else:
        return None


def detect_face(frameCount):

    #generate testing image
    global vs, outputFrame, lock, net, lock2, extract_flag
    frameCount = 0
    time.sleep(1.0)
    while True:
        #read another frame
        frame = vs.read()
        frame = imutils.resize(frame, width = 400)
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < confidenceBoundary:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            frameCount+=1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
    
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            croppedImg = frame[startY:endY, startX:endX]
            if frameCount <= 50:
                data_name = outputFolder +str(frameCount)+".jpg"
                cv2.imwrite(data_name, croppedImg)
            with lock:
                outputFrame = frame.copy()
            break
        if frameCount > 50:
            with lock2: 
                extract_flag = 1
            
def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encodedImage) + b'\r\n')

def get_prediction():
    global test_data, data_lock, data_flag
    
    while True:
        with data_lock:
            if data_flag != 1:
                continue
            X_test = test_data
            print(X_test.shape)
            numberOfSample, sizeOfSample = X_test.shape[0], X_test.shape[1]
            X_test = np.ravel(X_test)
            X_test = X_test.reshape((numberOfSample, sizeOfSample, -1))
            predictions = model.predict(X_test)
            print(predictions.shape)
            print(predictions)
            return predictions


@app.route("/face_detector")
def video_feed():
    return Response(generate(), 
            mimetype = "multipart/x-mixed-replace; boundary=frame")


@app.route("/get_result")
def get_result():
    global lock2, extract_flag
    while extract_flag != 1: 
        continue
    X_test = []
    print("Start extracting features")
    extractor.extractFeaturesFromImagePath(TESTTING_IMAGE_PATH, SEQUENCE_PATH)
    #load saved sequences
    allSequences = glob.glob(os.path.join(SEQUENCE_PATH + '/*npy'))
    for sequence in allSequences:
        sequence = load_one_sequence(sequence)
        print(sequence.shape)
        if sequence is None:
            print("cannot load any sequence named: " + sequence)
            raise
        X_test.append(sequence)
    X_test = np.array(X_test)
    print("done extracting test data")
    numberOfSample, sizeOfSample = X_test.shape[0], X_test.shape[1]
    X_test = np.ravel(X_test)
    X_test = X_test.reshape((numberOfSample, sizeOfSample, -1))
    predictions = model.predict(X_test)
    print(predictions)
    alertCounter = 0
    drowsyCounter = 0
    resultMessage = 'You are awake'
    for i in predictions:
        if i[0] > i[1]:
            drowsyCounter+=1
        else:
            alertCounter+=1
    if drowsyCounter > alertCounter:
        resultMessage = 'You are drowsy'
    return jsonify({
        'result': resultMessage
    })

    
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_face, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

vs.stop()