# Server
import socket
# import cv2
# import numpy
import time
from queue import Queue
from _thread import *
import threading
import pymysql
from datetime import datetime
import math

lock = threading.Lock()
# db_conn = pymysql.connect(host='116.89.189.36', user='root', passwd='4556',
#                           db='location', charset='utf8')
# curs = db_conn.cursor()

'''
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

'''
TCP_IP = ''
TCP_PORT = 8000

location=[]
enclosure_queue = Queue()
total_point=[]
index = ['61']
cli = ['0']
a = 0
b = 1

lock1 = threading.Lock()
lock2 = threading.Lock()

def combine_thread(point_now,point_pre):
    thread = threading.Thread(target=combine,args=(point_now,point_pre))
    thread.daemon = True
    thread.start()

def combine(point_now,point_pre):
    global location

    lock1.acquire()
    print('--combine')

    point_now=point_now
    point_pre=point_pre
    print('point_now:',point_now)
    print('point_pre:',point_pre)
    total_point = []


    now = len(point_now) / 2
    pre = len(point_pre) / 2

    flag_pre = [0 for _ in range((int)(pre))]
    flag_now = [0 for _ in range((int)(now))]
    cnt = 1
    i=0
    j=0


    for i in range((int)(now)):

      #  print('i', i)

        for j in range((int)(pre)):
           # print('j', j)

            distance = math.sqrt(
                float(((float)(point_now[2 * i]) - (float)(point_pre[2 * j])) ** 2 + (
                        (float)(point_now[2 * i + 1]) - (float)(point_pre[2 * j + 1])) ** 2))

            if distance < 0.00003:
                print('distance:', distance)


                x = ((float)(point_now[2 * i]) + (float)(point_pre[2 * j])) / 2
                y = ((float)(point_now[(2 * i) + 1]) + (float)(point_pre[(2 * j) + 1])) / 2
                total_point.append((str)(x))
                total_point.append((str)(y))

                if cnt==1:
                    flag_now[i] = cnt
                    flag_pre[j] = cnt
                elif flag_pre[j]!=flag_now[i]:
                    cnt=cnt+1
                    if flag_now[i]<flag_pre[j]:
                        flag_pre[j]=cnt
                        flag_now[i]=flag_pre[j]
                    else:
                        flag_now[i]=cnt
                        flag_pre[j]=flag_now[i]
                #print('total_point',total_point)
                # print('flag_now', flag_now)
                # print('flag_pre', flag_pre)

    i = 0
    j = 0

    for i in range((int)(now)):
        if flag_now[i]==0:
            total_point.append((point_now[2*i]))
            total_point.append((point_now[2 * i+1]))

    for j in range((int)(pre)):
        if flag_pre[j] == 0:
            total_point.append((point_pre[2 * j]))
            total_point.append((point_pre[2 * j + 1]))

    # print(flag_now)
    # print(flag_pre)
    #
    #
    #
    print('total', total_point)
    location=total_point

    lock1.release()


def threaded(conn, addr, queue, array):
    global index, a, b,location

    point_pre = []
    point_now = []

    cnt = 1
    i = 0
    j = 0
    print('Connect by :', addr[0], ':', addr[1])

    while True:

        for i in range(len(array)):
            if addr[0] == array[i]:
                start = time.time()
                # sql_sel = "select ip from person where ip=%s"
                # var_sel = (addr[0] + '%')
                # curs.execute(sql_sel, var_sel)
                # rows = curs.rowcount

                # length = recvall(conn, 16)
                # stringData = recvall(conn, int(length))
                # data = numpy.frombuffer(stringData, dtype='uint8')
                data = conn.recv(1024)
                point = data.decode()
                print(point)

                index_tmp = point.find('S')
                index.append(point[index_tmp + 1:])

                index_tmp2 = point.find('[')
                cli.append(point[index_tmp2:index_tmp2 + 3])

                index_tmp2 = point.find('[')

                print('index a',index[a])
                print('index b', index[b])
                print('cli a', cli[a])
                print('cli b', cli[b])

                point = point[:index_tmp2]
                point_now2 = point.split()

                if index[a] == index[b] and cli[a] != cli[b]:
                    lock2.acquire()
                    point_now=point_now2
                    lock2.release()

                    combine_thread(point_now,point_pre)

                point_pre = point_now
                print('Location:', location)

                a = a + 1
                b = b + 1

                #

                # decimg = cv2.imdecode(data, 1)
                # cv2.imshow('SERVER'+str(i), decimg)
                end = time.time()
                difftime = end - start
                #print(difftime)
                lock.acquire()
                # print(array[i],", time : ", format(difftime,'.6f'))

                lock.release()

                now = datetime.now()

                # sql_del = "delete from person where ip like %s"
                # val_del = (addr[0] + '%')

                # curs.execute(sql_del, val_del)
                # db_conn.commit()
                # print(len(location))
                len_tmp = len(location)
                length = len_tmp / 2

                # for j in range(int(length)):
                # sql_ins = "insert into person values(%s, %s, %s, %s)"
                # val_ins = (addr[0] + '/' + str(j), now, location[2 * j], location[2 * j + 1])

                # curs.execute(sql_ins, val_ins)
                # db_conn.commit()


'''
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
'''

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((TCP_IP, TCP_PORT))
server_socket.listen()
print('Listening')

array = []

while True:
    conn, addr = server_socket.accept()
    msg = conn.recv(1024)
    print(msg.decode('utf-8'))
    array.append(addr[0])
    print(array)
    start_new_thread(threaded, (conn, addr, enclosure_queue, array,))

server_socket.close()
# cv2.destroyAllWindows()
