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

def detectAndCount(model,frame):
	ClassIndex, confidence, bbox = model.detect(frame,confThreshold = 0.48)	
	#From the labels.txt file, the Classindex of a person is 1 
	count =0
	for index in ClassIndex:
		if index ==1: 
			count +=1
	return count

def detectVideo(model, conn, mclient, path=None):
	if path==None:
		video = cv2.VideoCapture(0)
	else:
		video = cv2.VideoCapture(path)
	totalPeople =[]
	frames = 0
	while(video.isOpened()):
		ret, frame = video.read()
		if ret== True: 
			frame = imutils.resize(frame,128*4, 64*4)
			frames +=1
			if (frames%20==0):
				totalPeople.append(detectAndCount(model,frame))
			if (len(totalPeople)==5):
				med = statistics.median(totalPeople)
				print ("Number of people detected =" + str(med))
				error = 0
				if med >=15: 
					print ("Exceeded the counter limit!!!")
					error = 1
				now = datetime.now()
				# dd/mm/YY H:M:S
				dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
				countTable.insert_table(conn, dt_string, med, error)
				totalPeople = []
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
# Reading the Image
#path ="../images/aLotOfPeople.jpg"
#image = cv2.imread(path)
def main():
	
	args= setArgs()
	model = importModelAndClassLabel()
	database = r"dataBase.db"
	conn = countTable.create_connection(database)
	# create table
	if conn is not None:
		table = countTable.create_table(conn)
	else:
		print("Error! cannot create the database connection.")
#Connect to server 
	if args['image'] is not None:
		path = args['image']
		detectImage(model, path)
		return
	elif args['video'] is not None:
		path = args['video']
		detectVideo(model, conn, path)
		countTable.read_from_db(conn)
		return
	else:
		detectVideo(model, conn)
		countTable.read_from_db(conn)
		return		

	exit(0)

if __name__ == "__main__":
    main()



