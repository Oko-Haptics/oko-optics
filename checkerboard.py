import cv2
import numpy as np


class Corners:

    IterCriteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.01)

    def __init__(self, img, grid_size):
        # 11 rows, 16 columns in image 12 horizontal lines, 17 horizontal lines
        self._image = np.copy(img)
        self._rows = grid_size[0]
        self._columns = grid_size[1]
        self._gray = None
        self._ret = False
        self._corners = None
        self._to_gray()

    def _to_gray(self):
        _b = np.copy(self._image)
        self._gray = cv2.cvtColor(_b, cv2.COLOR_BGR2GRAY)
        return

    def draw_chessboard(self):
        img = cv2.drawChessboardCorners(self._image, (self._rows, self._columns), self._corners, self._ret)
        return img if self._ret else np.zeros_like(self._image)

    def find_corners(self):
        _b = np.copy(self._gray)
        self._ret, corners = cv2.findChessboardCorners(_b, (self._rows, self._columns))
        if self._ret:  # refine the location of the corners
            self._corners = \
                cv2.cornerSubPix(np.copy(self._gray), corners, (11, 11), (-1, -1), Corners.IterCriteria)
        return True if self._ret else False

    @property
    def corners(self):
        return self._corners.reshape(-1, 2) if self._ret else self._corners
