# Client
import socket
import cv2
import numpy as np
from queue import Queue
from _thread import *
import time
from sympy import Symbol, solve
import math
import datetime

resolution = (854, 480)
img = 'big.mp4'

# GPS list
gps_list = []

# Right up 0,0
gps_list.append((35.832909, 128.754458))  # Left Up
gps_list.append((35.832842, 128.754476))  # Right Up
gps_list.append((35.832850, 128.754155))  # Left Down
gps_list.append((35.832776, 128.754171))  # Right Down
'''
#Left down 0,0
gps_list.append((35.832776,128.754171)) # Left Up
gps_list.append((35.832850,128.754155)) # Right Up
gps_list.append((35.832842,128.754476)) # Left Down
gps_list.append((35.832909,128.754458)) # Right Down
'''
# 픽셀 간 위도 경도 비
gps_width = (math.sqrt((gps_list[0][0] - gps_list[1][0]) ** 2 + (gps_list[0][1] - gps_list[1][1]) ** 2) + math.sqrt(
    (gps_list[2][0] - gps_list[3][0]) ** 2 + (gps_list[2][1] - gps_list[3][1]) ** 2)) / 2
gps_length = (math.sqrt((gps_list[0][0] - gps_list[2][0]) ** 2 + (gps_list[0][1] - gps_list[2][1]) ** 2) + math.sqrt(
    (gps_list[1][0] - gps_list[3][0]) ** 2 + (gps_list[1][1] - gps_list[3][1]) ** 2)) / 2
pixel_width = resolution[0]
pixel_length = resolution[1]
x_rate = gps_width / pixel_width
y_rate = gps_length / pixel_length

# 원본 이미지
capture = cv2.VideoCapture(img)
ret, img_original = capture.read()
img_original = cv2.resize(img_original, (resolution[0], resolution[1]))

# Perspective Point
point_list = []

point_list.append((333, 70))
point_list.append((559, 76))
point_list.append((157, 426))
point_list.append((623, 452))
'''

def mouse_callback(event, x, y, flags, param):
    global point_list, count, img_original


    # 마우스 왼쪽 버튼 누를 때마다 좌표를 리스트에 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        print("(%d, %d)" % (x, y))
        point_list.append((x, y))

        print(point_list)
        cv2.circle(img_original, (x, y), 3, (0, 0, 255), -1)



cv2.namedWindow('original')
cv2.setMouseCallback('original', mouse_callback)


while(True):

    cv2.imshow("original", img_original)


    height, width = img_original.shape[:2]


    if cv2.waitKey(1)&0xFF == 32: # spacebar를 누르면 루프에서 빠져나옵니다.
        break
cv2.destroyAllWindows()
'''

# Detect Person .xml
# cascadefile = "haarcascade_frontalface_default.xml"
# cascade = cv2.CascadeClassifier(cascadefile)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize
queue = Queue()
point = ""


def gps_conversion(foot_x, foot_y):
    a = Symbol('a')
    b = Symbol('b')

    equation1 = (gps_list[0][1] - a) ** 2 + (gps_list[0][0] - b) ** 2 - (foot_x * x_rate) ** 2
    equation2 = ((gps_list[0][0] - gps_list[1][0]) / (gps_list[0][1] - gps_list[1][1])) * (a - gps_list[0][1]) + \
                gps_list[0][0] - b
    res = solve((equation1, equation2), dict=True)
    # print(res[1])

    x = Symbol('x')
    y = Symbol('y')

    equation1 = (res[1][a] - x) ** 2 + (res[1][b] - y) ** 2 - (foot_y * y_rate) ** 2
    equation2 = (-(gps_list[0][1] - gps_list[1][1]) / (gps_list[0][0] - gps_list[1][0])) * (x - res[1][a]) + res[1][
        b] - y
    res = solve((equation1, equation2), dict=True)
    res1 = res[0][y]
    res2 = res[0][x]
    print('res: ', (res1, res2))
    if type(res1) is complex:
        res1 = res1.real
        print('!!!')
    if type(res2) is complex:
        res2 = res2.real
        print('!!!')
    gps = str(res1) + ' ' + str(res2) + ' '
    return gps


# Detect Function
def detect(frame):
    global point
    point_tmp = ""
    detected, _ = hog.detectMultiScale(frame)

    for (x, y, w, h) in detected:
        cv2.rectangle(frame, (x, y, w, h), (0, 255, 0), 3)
        foot_x = x + (w / 2)
        foot_y = y + (h - 60)
        point_tmp += perspective(frame, foot_x, foot_y)
        print(point_tmp)

    point = point_tmp
    return frame


def perspective(process, f_x, f_y):
    height, width = process.shape[:2]

    pts1 = np.float32([list(point_list[0]), list(point_list[1]), list(point_list[2]), list(point_list[3])])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # print(pts1)
    # print(pts2)

    M = cv2.getPerspectiveTransform(pts1, pts2)

    x = np.mat([[M[0][0], M[0][1], M[0][2]]])
    y = np.mat([[M[1][0], M[1][1], M[1][2]]])
    b = np.mat([[M[2][0], M[2][1], M[2][2]]])

    c = np.mat([[f_x],
                [f_y],
                [1]])

    x_map = int((x * c) / (b * c))
    y_map = int((y * c) / (b * c))

    # print((x_map, y_map))

    point = gps_conversion(x_map, y_map)
    return point

    # img_result = cv2.warpPerspective(process, M, (width,height))
    # return img_result


# Show webcam
# capture.set(cv2.CV_CAP_PROP_FPS,24)
real_ret, real_frame = capture.read()


def webcam(queue):
    while True:
        frame = real_frame
        ret = real_ret

        if ret == False:
            continue

        frame = detect(frame)

        cv2.imshow('CLIENT', frame)
        # frame=perspective(frame)

        # cv2.imshow('test', process)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break


def frame_drop():
    cap = cv2.VideoCapture(img)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    global real_frame
    global real_ret

    while True:
        real_ret, real_frame = cap.read()
        real_frame = cv2.resize(real_frame, (resolution[0], resolution[1]))

        if real_ret == False:
            continue

        cv2.imshow('c2', real_frame)

        key = cv2.waitKey(int(1000 / FPS))
        if key == ord('q'):
            break





# TCP_IP = '10.100.201.132'
# TCP_PORT = 10002

TCP_IP = '127.0.0.1'
TCP_PORT = 8000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((TCP_IP, TCP_PORT))
print('connect')
start_time = datetime.datetime.now()

start_new_thread(frame_drop, ())
start_new_thread(webcam, (queue,))
# webcam(queue)


while True:
    # stringData = queue.get()
    # time.sleep(1)
    # sock.send(str(len(stringData)).ljust(16).encode())

    if point != "":
        sock.send(point.encode())
        # print('send point')
        point = ""

    # time.sleep(0.5)

sock.close()
cv2.destroyAllWindows()
