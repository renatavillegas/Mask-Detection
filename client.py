import argparse
import json
import logging
import os
import random
import signal
import string
import sys
import time
import paho.mqtt.client as mqtt

DEFAULT_KPC_HOST = os.getenv('DEFAULT_KPC_HOST', 'mqtt.cloud.kaaiot.com')
DEFAULT_KPC_PORT = os.getenv('DEFAULT_KPC_PORT', 1883)

DEFAULT_METADATA_REQUEST_ID= 1
DEFAULT_DATA_COLLECTION_REQUEST_ID = 2

EPMX_INSTANCE_NAME = os.getenv('EPMX_INSTANCE_NAME', 'epmx')
DCX_INSTANCE_NAME = os.getenv('DCX_INSTANCE_NAME', 'dcx')

def killhandle(signum, frame):
  print("SIGTERM detected, shutting down")
  disconnectFromServer(client=client, host=host, port=port)
  sys.exit(0)

signal.signal(signal.SIGINT, killhandle)
signal.signal(signal.SIGTERM, killhandle)

APP_VERSION = "c2svemmgul2q7qujul8g-v1"
APP_TOKEN = "raspberry"

def composeTopicMetadata():
# METADATA section ------------------------------------
# Compose KP1 topic for metadata
	topic_metadata = "kp1/{application_version}/{service_instance}/{resource_path}".format(
		application_version=APP_VERSION,
		service_instance=EPMX_INSTANCE_NAME,
		resource_path="{token}/update/keys/{metadata_request_id}".format(token=APP_TOKEN,
			metadata_request_id=DEFAULT_METADATA_REQUEST_ID))
	print("Composed metadata topic: {}".format(topic_metadata))
	return topic_metadata

def composeTopicDataCollection():
# TELEMETRY section --------------------------------------
# Compose KP1 topic for data collection
	topic_data_collection = "kp1/{application_version}/{service_instance}/{resource_path}".format(
								application_version=APP_VERSION,
								service_instance=DCX_INSTANCE_NAME,
								resource_path="{token}/json/{data_request_id}".format(token=APP_TOKEN, data_request_id=DEFAULT_DATA_COLLECTION_REQUEST_ID))
	print("Composed data collection topic: {}".format(topic_data_collection))
	return topic_data_collection


def connectToServer():
	client_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
	client = mqtt.Client(client_id=client_id)
	host = DEFAULT_KPC_HOST
	port = DEFAULT_KPC_PORT
	client.connected_flag = False  # create flag in class
	client.on_connect = on_connect  # bind call back function
	client.on_message = on_message
	# Start the loop
	client.loop_start()
	client.connect(host, port, 60)
	while not client.connected_flag:  # wait in loop
		print("Waiting for connection with MQTT server")
		time.sleep(1)

	print("Successfully connected!")
	# Send metadata once on the first connection
	metadata = composeMetadata(version="v0.0.1")
	topic_metadata = composeTopicMetadata()
	client.publish(topic=topic_metadata, payload=metadata)
	print("Sent metadata: {0}\n".format(metadata))
	return client

def disconnectFromServer(client):
	print("Disconnecting from server at {0}:{1}.".format(DEFAULT_KPC_HOST, DEFAULT_KPC_PORT))
	time.sleep(4)  # wait
	client.loop_stop()  # stop the loop
	client.disconnect()
	print

def composeMetadata(version):
  return json.dumps(
    {
      "model": "webcam",
      "fwVersion": version,
      "customer": "CounterApp",
    }
  )

def composeDataSample(people, errorMask, errorPeople):
  payload = [
    {
      "timestamp": int(round(time.time() * 1000)),
      "people": people,
      "errorMask": errorMask,
      "errorPeople": errorPeople
    }
  ]
  return json.dumps(payload)

def on_connect(client, userdata, flags, rc):
  if rc == 0:
    client.connected_flag = True  # set flag
    print("Successfully connected to MQTT server")
  else:
    print("Failed to connect to MQTT code. Returned code=", rc)

def on_message(client, userdata, message):
	print("Message received: topic [{}]\nbody [{}]".format(message.topic, str(message.payload.decode("utf-8"))))

def send_data_sample(client, dataSample):
	topic_data_collection = composeTopicDataCollection()
	result = client.publish(topic=topic_data_collection, payload=dataSample)
	if result.rc != 0:
		print("Server connection lost, attempting to reconnect")
		connectToServer()
	else:
		print("{0}: Sent next data: {1}".format(APP_TOKEN, dataSample))