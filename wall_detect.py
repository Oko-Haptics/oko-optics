import cv2
import numpy as np


class FindWall:
    """
    Properties and functions to detect if the user is in front of a wall which presents a risk of collision
    """
    @classmethod
    def find(cls, _img):
        """
        search for threshold patters which are indicative of a wall
        :param _img: np.array, containing image
        :return: boolean, indicating whether a wall was detected
        """
        _blur = cv2.GaussianBlur(_img, (5, 5), 3)
        _hsv = cv2.cvtColor(_blur, cv2.COLOR_RGB2HSV)
        _gray = cv2.cvtColor(_hsv, cv2.COLOR_RGB2GRAY)
        _mean = np.mean(_gray)
        _std = np.std(_gray)
        thresh = cv2.inRange(_gray, _mean-_std, _mean+_std, cv2.THRESH_BINARY_INV)
        contours = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
        return True if len(contours) > 2000 else False


if __name__ == "__main__":
    left = cv2.VideoCapture(2)
    while True:
        _, cap = left.read()
        cv2.imshow('test', cap)
        if FindWall.find(cap):
            print("WALL!")
        if cv2.waitKey(20) == 27:
            break
    left.release()
    cv2.destroyAllWindows()
