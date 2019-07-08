import cv2

cam = cv2.VideoCapture(2)
key = 0
while not key == 27:
    _, frame = cam.read()
    cv2.imshow('oop', frame)
    key = cv2.waitKey(20)

cv2.destroyAllWindows()
cam.release()

