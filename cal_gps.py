import cv2
import numpy as np
import time

i=0
width=480
length=852
gps_list=[(35.832909,128.754458),(35.832842,128.754476),(35.832852,128.754121),(35.832776,128.754171)]

human_list=[(254,427)] #warping 사람 픽셀 list
human_warp_list=[]
pixel_list=[(198,298),(343,300),(69,625),(386,634)]

def warp(frame_rotate):
    # Locate points of the documents or object which you want to transform
    pts1 = np.float32(pixel_list)
    pts2 = np.float32([[0, 0],[480, 0],[0,852],[480, 852]])
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame_rotate, matrix, (width, length))
    cv2.imshow("frame_rotate",result)

def cal():
    global human_warp_list,human_list,width,length,i 

    human_lat=gps_list[0][0]-human_list[i][0]*lat_pixel
    human_long=gps_list[0][1]-human_list[i][1]*long_pixel
    print(human_lat)
    print(human_long)
    human_warp_list[i][0].append(human_lat)
    human_warp_list[i][1].append(human_long) 
   
filepath=r'D:\git repos\NetCC\NetCC_-Image-recognition\python\gps1..jpg'

frame = cv2.imread(filepath)
ret = cv2.imread(filepath)
frame=cv2.resize(frame,dsize=(480,852),interpolation=cv2.INTER_AREA)
cv2.imshow('original',frame)

warp(frame)
lat_mean=((gps_list[0][0]-gps_list[1][0])+(gps_list[2][0]-gps_list[3][0]))/2
long_mean=((gps_list[0][1]-gps_list[2][1])+(gps_list[1][1]-gps_list[3][1]))/2

lat_pixel=lat_mean/width
print(lat_pixel)
long_pixel=long_mean/length
print(long_pixel)

for i in range(len(human_list)):
    cal()
