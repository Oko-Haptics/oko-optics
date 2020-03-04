import cv2

# naming convention of cameras is driven by user perspective, right is on their right.
cam_left = cv2.VideoCapture(2)
cam_right = cv2.VideoCapture(1)
key = 0
while not key == 27:
    _, frame1 = cam_left.read()
    _, frame2 = cam_right.read()
    cv2.imshow('left', frame1)
    cv2.imshow('right', frame2)
    key = cv2.waitKey(20)

cv2.destroyAllWindows()
cam_left.release()
cam_right.release()

