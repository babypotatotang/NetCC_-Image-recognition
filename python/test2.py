import cv2

image = cv2.imread("C:\Users\gwons\Desktop\NET 챌린지\test2.jpg", cv2.IMREAD_ANYCOLOR)
cv2.imshow("Moon", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
