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

def warp():
    print('1')
    frame = cv2.imread(fileName)
    ret = cv2.imread(fileName)

    # Locate points of the documents or object which you want to transform
    pts1 = np.float32([[292, 330],[505, 341],[141, 651],[567, 671]])
    pts2 = np.float32([[0, 0],  [800, 0],[0,1000], [800, 1000]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (800, 1000))

    plt.imshow(result)
    plt.show()


fileName = 'gps1.jpg'

corner()
print('warp')
warp()
