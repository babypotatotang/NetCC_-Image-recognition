import cv2
import numpy

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
detector = cv2.CascadeClassifier('haarcascade_fullbody.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    size = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    if not ret:
        break

    detected, _ = hog.detectMultiScale(frame)
    for (x, y, w, h) in detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    for (i, j, k, l) in faces:
        cv2.rectangle(frame, (i, j), (i + k, j + l), (255, 0, 0), 2)
        # Nose
        a = i + k / 2
        b = j + l / 2
        A = int(a)
        B = int(b)

    # 2D image points. If you change the image, you need to change vector
    image_points = numpy.array([
        (359, 391),  # Nose tip
        (399, 561),  # Chin
        (337, 297),  # Left Eye corner
        (513, 301),  # Right Eye corner
        (345, 465),  # Left Mouth Corner
        (453, 469)  # Right Mouth Corner
    ])
    image_points = image_points.astype('float32')

    # 3D model points.
    model_points = numpy.array([
        (0, 0, 1),  # Nose tip
        (0, -330, -65),  # Chin
        (-225, 170, -135),  # Left Eye corner
        (225, 170, -135),  # Right Eye corner
        (-150, -150, -125),  # Left Mouth Corner
        (150, -150, -125)  # Right Mouth Corner
    ])
    model_points = model_points.astype('float64')

    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = numpy.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double"
    )

    print("Camera Matrix :\n {0}".format(camera_matrix))
    dist_coeffs = numpy.zeros((8, 1))  # Assuming no lens distortion
    print("dist_coeffs :\n {0}".format(dist_coeffs))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    print("Rotation Vector:\n {0}".format(rotation_vector))
    print("Translation Vector:\n {0}".format(translation_vector))

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    # (nose_end_point2D, jacobian) = cv2.projectPoints(numpy.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # for p in image_points:
    #     cv2.circle(frame, (int(p[0][0]),int(p[0][1])), 3, (0, 0, 255), -1)

    # p1 = (int(image_points[0][0]), int(image_points[0][1]))
    # p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    # cv2.line(frame, p1, p2, (255, 0, 0), 2)

    # Object_픽셀좌표
    p = numpy.array([[x + w / 2, y + h]])
    rotation_transpose = rotation_vector.T
    #print("Pixel 좌표:\n {0}".format(p))

    # pc:정규좌표
    u = (x - center[0]) / focal_length
    v = (y - center[1]) / focal_length
    pc = numpy.array([[u, v, 1]])
    #print("정규 좌표:\n {0}".format(pc))

    # pw:월드좌표
    pw_a = pc - translation_vector
    pw = numpy.dot(rotation_transpose, pw_a)
    #print("월드 좌표:\n {0}".format(pw))

    # cc:카메라 좌표(카메라), cw:월드좌표(카메라)
    cc = numpy.array([[0, 0, 0]])
    cw = numpy.dot(rotation_transpose, (cc - translation_vector))
    print("월드 좌표(카메라):\n {0}".format(cw))
    print(cw[0][2])

    # 3차원 상 물체의 좌표 값
    P = numpy.zeros((1, 3))
    cw_z = cw[0][2]
    pw_z = pw[0][2]
    k = cw_z / (cw_z - pw_z)
    P = cw + numpy.dot(k, (pw - cw))
    print("3차원 상 물체의 좌표 값:\n {0}".format(P))

    cv2.imshow('Detect', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()