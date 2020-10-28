# Server
import socket
import cv2
import numpy
import time
from queue import Queue
from _thread import *
import threading

lock = threading.Lock()


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


TCP_IP = ''
TCP_PORT = 8000


enclosure_queue = Queue()


def threaded(conn, addr, queue, array):
    cnt = 1
    i=0
    print('Connect by :', addr[0], ':', addr[1])
    
    while True:

        for i in range(len(array)):
            if addr[0] == array[i]:
                start = time.time()
                length = recvall(conn, 16)
                stringData = recvall(conn, int(length))
                data = numpy.frombuffer(stringData, dtype='uint8')
                decimg = cv2.imdecode(data, 1)
                cv2.imshow('SERVER'+str(i), decimg)
                end = time.time()
                difftime = end - start
                lock.acquire()
                print(array[i],", time : ", format(difftime,'.6f'))
                lock.release()

    
        

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((TCP_IP, TCP_PORT))
server_socket.listen()
print('Listening')

array=[]

while True:
    conn, addr = server_socket.accept()
    array.append(addr[0])
    print(array)
    start_new_thread(threaded, (conn, addr, enclosure_queue, array,))
    

server_socket.close()
cv2.destroyAllWindows()
