import cv2
import numpy as np
import time

def warp(frame_rotate):
    # Locate points of the documents or object which you want to transform
    pts1 = np.float32(pixel_list)
    pts2 = np.float32([[0, 0],[480, 0],[0,852],[480, 852]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame_rotate, matrix, (width, length))
    cv2.imshow("frame_rotate",result)


filepath=r'C:\Users\gwons\Desktop\netcc\gps.mp4'
cap=cv2.VideoCapture(filepath)
width=480
length=852
#코너 gpt list
gps_list=[(35.832909,128.754458),(35.832842,128.754476),(35.832850,128.754155),(35.832776,128.754171)]
human_list=[] #warping 전 사람의 pixel list
pixel_list=[(199,312),(337,317),(82,522),(365,519)]

while(cap.isOpened()):
    ret,frame=cap.read()
    frame_rotate=cv2.transpose(frame)
    frame_rotate=cv2.flip(frame_rotate,1)
    time.sleep(0.026)
    if ret:
        cv2.imshow('video',frame_rotate)
        warp(frame_rotate)
        
        #픽셀 간 위도 경도 비
        lat_pixel=((gps_list[0][0]-gps_list[1][0])+(gps_list[2][0]-gps_list[3][0])/2)/width
        long_pixel=((gps_list[0][1]-gps_list[2][1])+(gps_list[1][1]-gps_list[3][1])/2)/length
        
        for x,y in human_list:
            human_warp_list[x]=(width*human_list[x][0])*lat_pixel #최종 인식된 사람 위도
            human_warp_list[y]=(length*human_list[x][1])*long_pixel #최종 인식된 사람 경도
        

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
