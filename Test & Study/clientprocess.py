#Client
import socket
import cv2
import numpy
from queue import LifoQueue
from _thread import *
import time

cascadefile = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascadefile)

queue = LifoQueue()

def detect(gray, frame):
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y, w, h), (0, 255, 0), 3)
        
    return frame

def webcam(queue):
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        process=detect(gray, frame)
        
        if ret == False:
            continue

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, imgencode = cv2.imencode('.jpg', process, encode_param)

        data = numpy.array(imgencode)
        stringData = data.tobytes()

        queue.put(stringData)

        cv2.imshow('CLIENT', process)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

TCP_IP = '165.229.125.89'
TCP_PORT = 8000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((TCP_IP, TCP_PORT))
print('connect')

start_new_thread(webcam, (queue,))

while True:
    stringData = queue.get()
    #time.sleep(0.2)
    sock.send(str(len(stringData)).ljust(16).encode())
    sock.send(stringData)
    print('send image')

sock.close()
cv2.destroyAllWindows()
