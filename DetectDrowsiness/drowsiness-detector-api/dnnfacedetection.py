from imutils.video import VideoStream, FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import random as rnd

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True)
ap.add_argument("-o", "--output", required=True)

args = vars(ap.parse_args())

print("[INFO] loading model and video ... ")
protoPath = "./drowsiness-detector-model/face-detection/deploy.prototxt.txt"
modelPath = "./drowsiness-detector-model/face-detection/res10_300x300_ssd_iter_140000.caffemodel"
outputFolder = args["output"]
confidenceBoundary = 0.95
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
vs = cv2.VideoCapture(args["video"])
# totalFrame = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
# clip = rnd.randint(1500, 3000)
# startFrame = int((clip/totalFrame))
# vs.set(cv2.CV_CAP_PROP_POS_FRAMES,startFrame)

frameCount = 0

time.sleep(1.0)

while True:
    #read another frame
    ret, frame = vs.read()
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
        data_name = outputFolder +str(frameCount)+".jpg"
        cv2.imwrite(data_name, croppedImg)
        break
    
    if(frameCount == 1000):
        break


