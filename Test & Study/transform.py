<<<<<<< HEAD
import numpy as np
import cv2


point_list = []
count = 0

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

# 원본 이미지
img_original = cv2.imread('pic.jpg')


while(True):

    cv2.imshow("original", img_original)


    height, width = img_original.shape[:2]


    if cv2.waitKey(1)&0xFF == 32: # spacebar를 누르면 루프에서 빠져나옵니다.
        break


# 좌표 순서 - 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝
pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

print(pts1)
print(pts2)

M = cv2.getPerspectiveTransform(pts1,pts2)

img_result = cv2.warpPerspective(img_original, M, (width,height))


cv2.imshow("result1", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
=======
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
>>>>>>> c156b12588c3f9f16fec198c50419e2354b099e4
