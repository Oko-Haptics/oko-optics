import cv2
import numpy as np

from stereopy_controls import Controls


class Floor:
    """
    Class to store color properties of the floor being traversed by the user
    """
    def __init__(self):
        """
        initialize the mean and standard deviation of the floor
        """
        self._mean = 0.0
        self._std = 0.0
        self._initialized = False

    def update(self, n_mean, n_std):
        """
        develop a weighted moving average of the floor being traversed by the user and update the values of mean and
        standard deviation of the floor space in use
        :param n_mean: double, current measurement of the mean of the floor
        :param n_std: double, current measurement of the standard deviation of the floor
        :return:
        """
        if not self._initialized:
            self._mean = n_mean
            self._std = n_std
            self._initialized = True
        else:
            self._mean = self._mean*0.99 + 0.01*n_mean
            self._std = self._std*0.99 + 0.01*n_std
        return

    @property
    def low(self):
        return self._mean - self._std

    @property
    def high(self):
        return self._mean + self._std


class FloorFinder:
    """
    Tools for the detection of allowable walking regions using image segmentation to detect floors.
    Assumptions/Process followed by algorithm
    - Establish a perspective roi in the image sent for processing
    - The image dimensions are nxmx3, where 3 are the values of the HSV color scale
    - The most popular color in the roi is the color of the floor being traversed by the user
    - Using a mean-std threshold, the associated region is segmented
    """

    floor = Floor()

    @classmethod
    def _get_hsv(cls, _img):
        """
        convert the image passed to the function from RGB (Red Gree Blue) to HSV (Hue Saturation Value)
        :param _img: np.array, containing RGB image
        :return: np.array, containing image converted to HSV color scale
        """
        return cv2.cvtColor(_img, cv2.COLOR_RGB2HSV)

    @classmethod
    def _gaussian_blur(cls, _img):
        """
        apply gaussian blur to the image which used to detect floors
        :param _img: np.array, containing the image data which needs to be blurred
        :return: np.array, blurred image
        """
        return cv2.GaussianBlur(_img, Controls.gaussian_kernel, Controls.gaussian_std_x)

    @classmethod
    def _get_3d_roi(cls, _row, _col):
        """
        initialize a 3D region of interest in which the color of the floor is to be detected
        :param _row: int, number of rows of pixels in the roi
        :param _col: int, number of columns in the roi
        :return: np.array, _rowx_colx3 dimension array containing the 3D roi mask
        """
        _corners = np.array(
            [[[int(3 * _col / 5), 0], [int(2 * _col / 5), 0], [0, int(_row)], [int(_col), int(_row)]]]
        ).astype('int32')
        _mask = np.zeros((_row, _col))
        cv2.fillPoly(_mask, _corners, (1, 1, 1))
        _mask3d = np.array([_mask] * 3).T.swapaxes(0, 1)  # stack roi to nxnx3 for the three HSV arrays
        return _mask3d

    @classmethod
    def _get_thresholds(cls, _row, _col, _img):
        """
        compute the lower and upper threshold limits for the hsv image region of interest
        :param _row: int, number of rows of pixels in the source image
        :param _col: int, number of columns of pixels in the source image
        :param _img: np.array, the source image
        :return: dict {low, high} containing the target thresholds for image segmentation designed for floor detection
        """
        _bottom = _img[int(_row//2):, :, :]
        _roi = cls._get_3d_roi(_row=_bottom.shape[0], _col=_col)
        _search_space = np.multiply(_bottom, _roi)  # convert 3D ROI to a region of interested with blocked black pixels
        _target = _search_space[~np.all(_search_space == 0, axis=2)] # ignore black pixels, change to a Nx3 array
        _mean = np.mean(np.mean(_target, axis=0))
        _std = np.std(np.std(_target, axis=0))
        cls.floor.update(n_mean=_mean, n_std=_std)
        return {
            'low': cls.floor.low,
            'high': cls.floor.high
        }

    @classmethod
    def _bincounter(cls, _img):
        """
        function used to determine the most popular segmented region in the post processed image.
        :param _img: np.array, containing segmented image
        :return: np.array 3x1, indicating the HSV scale of the largest region identified in the image
        """
        arr_2d = _img.reshape(-1, _img.shape[-1])
        col_max = (256, 256, 256)
        arr_1d = np.ravel_multi_index(arr_2d.T, col_max)
        return np.array(np.unravel_index(np.bincount(arr_1d).argmax(), col_max))

    @classmethod
    def _image_segmentation(cls, _low, _high, _img):
        """
        apply opencv2 thresholding to segment the image on the basis of thresholds identified in cls._get_thresholds
        :param _low: int, lower threshold
        :param _high: int, higher threshold
        :param _img: np.array, the image to which thresholds are applied
        :return: np.array, the image after applying thresholds
        """
        _, _result = cv2.threshold(_img, _low, _high, cv2.THRESH_BINARY)
        return _result

    @classmethod
    def find(cls, _src, _shape):
        """
        non-protected member, serves as the interface between client script and class FindFloor, accepts the souce image
        and returns the interpretted floor region in the image
        :param _src: np.array, source image
        :param _shape: np.array, dimensions of the source image
        :return: np.array, like _src, contains 1 where the floor is detected else 0
        """
        _hsv = cls._gaussian_blur(_img=cls._get_hsv(_img=_src))
        thresholds = cls._get_thresholds(_row=_shape[0], _col=_shape[1], _img=_hsv)
        segmented = cls._image_segmentation(
            _low=thresholds['low'], _high=thresholds['high'], _img=_hsv
        )
        target_hsv = cls._bincounter(_img=segmented)
        return cv2.inRange(segmented, target_hsv, target_hsv, cv2.THRESH_TOZERO)


if __name__ == '__main__':

    left = cv2.VideoCapture(0)

    while True:
        _, cap = left.read()
        cv2.imshow('raw', cap)

        floor_finder = FloorFinder()

        floor_zone = floor_finder.find(_src=cap, _shape=cap.shape[:-1])

        cv2.imshow('result', np.invert(floor_zone))
        if cv2.waitKey(20) == 27:
            break
    left.release()
    cv2.destroyAllWindows()
