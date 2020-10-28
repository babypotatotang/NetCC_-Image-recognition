import matplotlib.pyplot as plt
import cv2
import numpy as np

def corner():
    image = cv2.imread(fileName)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageCorner = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    imageGray = np.float32(imageGray)
    result = cv2.cornerHarris(imageGray, 4, 3, 0.07)
    result = cv2.dilate(result, None)

    imageCorner[result > 0.01 * result.max()] = [255, 0, 0]
    
    print('--imagecorner--')
    print(imageCorner)
    
    plt.imshow(imageCorner)
    plt.show()
    print('dl')

def warp():
    print('1')
    frame = cv2.imread(fileName)
    ret = cv2.imread(fileName)

    # Locate points of the documents or object which you want to transform
    pts1 = np.float32([[73, 300],[730, 63],[306, 1319],[1030, 1187]])
    pts2 = np.float32([[0, 0],[400, 0],  [0, 600], [400, 600]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (400, 600))

    plt.imshow(result)
    plt.show()


fileName = 'test2.jpg'

corner()
print('warp')
warp()
