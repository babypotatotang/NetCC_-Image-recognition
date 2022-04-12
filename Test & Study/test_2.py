import cv2
import numpy as np
import matplotlib.pyplot as plt

def findLocalMaxima(src):
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(11, 11))
    dilate = cv2.dilate(src, kernel)
    localMax = (src == dilate)

    erode = cv2.erode(src, kernel)
    localMax2 = src > erode
    localMax &= localMax2
    points = np.argwhere(localMax == True)
    points[:, [0, 1]] = points[:, [1, 0]]
    return points


src = cv2.imread('pic.JPG')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
res = cv2.preCornerDetect(gray, ksize=3)
ret, res2 = cv2.threshold(np.abs(res), 0.1, 0, cv2.THRESH_TOZERO)

corners = findLocalMaxima(res2)
print(corners)
print('corners.shape=', corners.shape)

dst = src.copy()
for x, y in corners:
    cv2.circle(dst, (x, y), 5, (0, 0, 255), 2)

print(corners[0])
cv2.circle(dst,(corners[0][0],corners[0][1]),5,(0,255,0),2) #335 75 right up
print(corners[1])
cv2.circle(dst,(corners[1][0],corners[1][1]),5,(0,255,0),2) #87 103 left up
print(corners[2])
cv2.circle(dst,(corners[2][0],corners[2][1]),5,(0,255,0),2) #449 332 right down
print(corners[3])
cv2.circle(dst,(corners[3][0],corners[3][1]),5,(0,255,0),2) #55 388 left up
print(corners[4])
cv2.circle(dst,(corners[4][0],corners[4][1]),5,(0,255,0),2) #335 75 right up

imgae2 = plt.subplot(1, 1,1)
imgae2.set_title('CornerTest')
plt.axis('off')
plt.imshow(dst, cmap="gray")

plt.show()
