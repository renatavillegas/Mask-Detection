import cv2
import imutils
import statistics
import argparse
import sys
sys.path.append('../dataTable')
import countTable
import sqlite3
from sqlite3 import Error
from datetime import datetime 
from client import *
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from math import ceil
def setArgs():
	all_args = argparse.ArgumentParser()
	all_args.add_argument("-v", "--video", required=False,
   						  help="Video Stream. Path to the video.")

	all_args.add_argument("-i", "--image", required=False,
   						  help="Image. Path to the image")
	args = vars(all_args.parse_args())
	return args 

def importModelAndClassLabel():
	confg_file= "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
	frozen_model= "frozen_inference_graph.pb"

	model = cv2.dnn_DetectionModel(frozen_model, confg_file)

	classLabel=[]
	file_name = "labels.txt" 
	with open(file_name,'rt') as fpt:
		classLabel= fpt.read().rstrip("\n").split("\n")

	model.setInputSize(320, 320)
	model.setInputScale(1.0/127.5)
	model.setInputMean(127.5)
	model.setInputSwapRB(True)

	return model

def importFaceModelDetector():
	prototxtPath ="deploy.prototxt"
	weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
	return faceNet
def importMaskModelDetector():
	maskNet = load_model("mask_detector.model")
	return maskNet

def detectAndCount(model,frame):
	ClassIndex, confidence, bbox = model.detect(frame,confThreshold = 0.48)	
	#From the labels.txt file, the Classindex of a person is 1 
	count =0
	for index in ClassIndex:
		if index ==1: 
			count +=1
	return count

def detectVideo(model,client, faceDetector, maskDetector, path=None):
	if path==None:
		video = cv2.VideoCapture(0)
	else:
		video = cv2.VideoCapture(path)
	totalPeople =[]
	errorMask = []
	frames = 0
	while(video.isOpened()):
		ret, frame = video.read()
		if ret== True: 
			frame = imutils.resize(frame,128*4, 64*4)
			frames +=1
			if (frames%25==0):
				totalPeople.append(detectAndCount(model,frame))
				(locs, preds) = detect_and_predict_mask(frame, faceDetector, maskDetector)
				for (box, pred) in zip(locs, preds):
					# unpack the bounding box and predictions
					(startX, startY, endX, endY) = box
					(mask, withoutMask) = pred
					if mask > withoutMask:
						print("Mask!")
						errorMask.append(0)
					else:
						print("No Mask!!")
						errorMask.append(1)
			if (len(totalPeople)==5):
				med = statistics.median(totalPeople)
				medErrors=0
				print ("Number of people detected =" + str(med))
				if len(errorMask) >0:
					medErrors = ceil(statistics.median(errorMask))
				errorPeople = 0
				if med >=15: 
					print ("Exceeded the counter limit!!!")
					errorPeople = 1
				payload = composeDataSample(med, medErrors, errorPeople)
				send_data_sample(client, payload)
				totalPeople = []
				errorMask =[]
			cv2.imshow('video',frame)
			if cv2.waitKey(1)==27:
				break
		else: 
			 break
	video.release()
	cv2.destroyAllWindows()
	return 

def detectImage(model, path):
	img = cv2.imread(path)
	if img is None: 
		print("---(!) Image not found")
		return
	print("Number of people detected =" + detectAndCount(model, img))

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (128*4, 64*4),
		(104.0, 117.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.35:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=10)
		print (preds)
	print ("len faces=", len(faces))
	return (locs, preds)


def main():
	
	args= setArgs()
	model = importModelAndClassLabel()
	faceDetector= importFaceModelDetector()
	maskDetector = importMaskModelDetector()
	database = r"dataBase.db"
	conn = countTable.create_connection(database)
	# create table
	if conn is not None:
		table = countTable.create_table(conn)
	else:
		print("Error! cannot create the database connection.")
#Connect to server 
	mclient = connectToServer()
	if args['image'] is not None:
		path = args['image']
		detectImage(model, path)
		return
	elif args['video'] is not None:
		path = args['video']
		detectVideo(model,mclient, faceDetector, maskDetector, path)
		countTable.read_from_db(conn)
		return
	else:
		detectVideo(model,mclient)
		countTable.read_from_db(conn)
		return		

	exit(0)

if __name__ == "__main__":
    main()



