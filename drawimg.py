import numpy as np
import cv2

rvecs, tvecs, points_3D, points_2D = dict(np.load("savedOldSchool.npz")).values()


for j in range(1,95):
    points_2D = np.array(points_2D)[j-1]
    points_2D = np.reshape(points_2D, (-1,2))

    img = cv2.imread("C:/Users/Jakob/Documents/SDU/8.Sem/Project/bundle/cam_0_00{j}.png")
    for i, point in enumerate(points_2D):
        c = i / len(points_2D)
        cv2.circle(img, (point[0], point[1]), 5, (255*c,0,(1-c)*255), -1)

    cv2.imshow("test",img)

