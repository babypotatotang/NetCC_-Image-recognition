import matplotlib.pyplot as plt
import cv2
import numpy as np


def cornerDetect():
    fileName = 'pic.jpg'

    image = cv2.imread(fileName)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageCorner = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    imageGray = np.float32(imageGray)
    result = cv2.cornerHarris(imageGray, 4, 3, 0.07)
    result = cv2.dilate(result, None) #검출된 코너 부분을 확대해서 확인
    imageCorner[result > 0.01 * result.max()] = [255, 0, 0]

    plt.imshow(imageCorner)
    plt.show()


cornerDetect()
