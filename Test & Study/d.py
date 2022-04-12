import cv2

while True:
   capture = cv2.VideoCapture(r"C:\Users\gwons\Desktop\netcc\gps.mp4")

   if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        capture.open(r"C:\Users\gwons\Desktop\netcc\gps.mp4")

   ret,frame=capture.read()
   cv2.imshow("VideoFrame",frame)
