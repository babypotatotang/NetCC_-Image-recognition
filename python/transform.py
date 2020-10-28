import matplotlib.pyplot as plt
import cv2
import numpy as np

def corner():
    while(True):
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

def warp():
    print('1')
    frame = cv2.imread(fileName)
    ret = cv2.imread(fileName)

    # Locate points of the documents or object which you want to transform
    pts1 = np.float32([[69, 297],[726, 59],[306, 1315],[1022, 1179]])
    pts2 = np.float32([[0, 0],  [0, 400],[400, 0], [400, 400]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (400, 400))

    plt.imshow(result)
    plt.show()


fileName = 'gps1..jpg'

corner()
print('warp')
warp()
