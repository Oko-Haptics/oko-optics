import cv2
from functools import partial
import numpy as np
import os
import pickle
from tkinter import Label, Scale, Tk, HORIZONTAL

from checkerboard import Corners


class _RectifyCalibration:
    """
    Protected member with instances used to calibrate cameras by storing cv2.remap transforms from
    disparity calculations
    """
    calib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    flags = (cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_SAME_FOCAL_LENGTH)
    calibration_file = 'stereo_cal.pkl'

    def __init__(self, input_folder=None):
        """
        Initialize camera calibration.
        If an input file is provided, initialize the class instance with values from that file
        :param input_folder: str, path to a target input folder
        """
        #: Camera matrices (M)
        self.cam_mats = {"left": None, "right": None}
        #: Distortion coefficients (D)
        self.dist_coefs = {"left": None, "right": None}
        #: Rotation matrix (R)
        self.rot_mat = None
        #: Translation vector (T)
        self.trans_vec = None
        #: Essential matrix (E)
        self.e_mat = None
        #: Fundamental matrix (F)
        self.f_mat = None
        #: Rectification transforms (3x3 rectification matrix R1 / R2)
        self.rect_trans = {"left": None, "right": None}
        #: Projection matrices (3x4 projection matrix P1 / P2)
        self.proj_mats = {"left": None, "right": None}
        #: Disparity to depth mapping matrix (4x4 matrix, Q)
        self.disp_to_depth_mat = None
        #: Bounding boxes of valid pixels
        self.valid_boxes = {"left": None, "right": None}
        #: Undistortion maps for remapping
        self.undistortion_map = {"left": None, "right": None}
        #: Rectification maps for remapping
        self.rectification_map = {"left": None, "right": None}
        if input_folder:
            self.load(input_folder)

    def export(self, output_folder):
        """
        export the instance properties as a json file
        :param output_folder: str, path to the output file where disparity remap values are to be stored
        :return:
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(os.path.join(output_folder, self.calibration_file), 'wb') as cal_file:
            pickle.dump(self.__dict__, cal_file)
        return

    def load(self, input_folder):
        """
        load disparity remap calibration values from the specified json file
        :param input_folder: str, path to the target json file
        :return:
        """
        assert os.path.isfile(os.path.join(input_folder, self.calibration_file)), \
            "Calibration file does not exist!"
        with open(os.path.join(input_folder, self.calibration_file), 'rb') as cal_file:
            _cal = pickle.load(cal_file)
        for key, val in _cal.items():
            self.__dict__[key] = val
        return

    def rectify(self, frame, side):
        """
        use the openCv2 instance of the image remap applied to the disparity calculation values
        :param frame: np.array, storing the image being remapped
        :param side: str, indicating whether the image being remapped is `left` or `right
        :return: cv2.remap
        """
        return (cv2.remap(frame,
                          self.undistortion_map[side],
                          self.rectification_map[side],
                          cv2.INTER_NEAREST))


class RectifyCalibrator:
    """
    Instances used to applying image rectification for calibrating stereo vision pairs
    """
    def __init__(self, row, col, edge, left_imgs, right_imgs, show_results=False):
        """
        initialize properties and features to rectify stereo vision pairs using disparity calibration on chessboards
        :param row: int, number of rows in the chessboard used for calibration
        :param col: int, number of columns in the chessboard used for calibration
        :param edge: float, the edge width of the chessboard used for calibration
        :param left_imgs: list, containing paths to the images captured from the left camera
        :param right_imgs: list, containing paths to the images captured from the right camera
        :param show_results: boolean, indicate whether the results are to be shown
        """
        self._grid = (row-1, col-1)
        self._edge = edge
        self._left_imgs = left_imgs
        self._right_imgs = right_imgs
        self._image_pairs = []
        self._points = {'left': [], 'right': []}
        self._checkerboard_coordinates = []
        self._images_found = False
        self._image_size = (0, 0)
        self._display = show_results
        self._verify_images()
        row, col = np.indices(self._grid)
        self._coordinates = np.zeros((np.prod(self._grid), 3), dtype=np.float32)
        self._coordinates[:, :2] = np.array((row.T.reshape(-1), col.T.reshape(-1))).T * edge

    def _verify_images(self):
        _files = self._left_imgs + self._right_imgs
        _missing = [not(os.path.isfile(l)) for l in _files]
        if True in _missing:
            print('The following images are missing:')
            print('\n'.join(list(np.array(_files)[_missing])))
            return
        try:
            assert len(self._left_imgs) == len(self._right_imgs), \
                "Image pair lists are not equal length"
            self._images_found = True
            _prev_size = None
            for i, _ in enumerate(self._left_imgs):
                _right = cv2.imread(self._right_imgs[i])
                _left = cv2.imread(self._left_imgs[i])
                assert np.array_equal(_right.shape, _left.shape), \
                    f"Sizes of image pair {i+1} not equal"

                if _prev_size is None:
                    _prev_size = _left.shape
                else:
                    assert np.array_equal(_prev_size, _left.shape), \
                        f"Sizes of images are not equal between {i} and {i+1} image pairs"
                self._image_pairs.append({"left": _left, "right": _right})
            self._image_size = _prev_size[1], _prev_size[0]
        except AssertionError as e:
            print(f"Error in setting up images:\n{e}")
            raise RuntimeError(e)

        return

    def find_corners(self):
        """
        identify and extract positional information for the corners of a chessboard in a image
        :return:
        """
        for index, image_pair in enumerate(self._image_pairs):
            print(f"Processing image pair: {index+1}/{len(self._image_pairs)}")
            _found = {'left': False, 'right': False}
            _tmp = {'left': [], 'right': []}
            for key, _img in image_pair.items():
                corners = Corners(img=_img, grid_size=self._grid)
                _found[key] = corners.find_corners()
                _tmp[key] = corners.corners
                if self._display:
                    self._show_image(f"{key}{index+1}", corners.draw_chessboard())
            if False in _found.values():
                print(f"Did not find corners in both images\n")
            else:
                self._points['left'].append(_tmp['left'])
                self._points['right'].append(_tmp['right'])
                self._checkerboard_coordinates.append(self._coordinates)
        return

    def calibrate(self):
        """
        use instance properties to calculate disparity values and instantiate protected class member _RectifyCalibration
        to store resultant remap matrix values as a calibration object and json file
        :return: _RectifyCalibration, calibration object storing remap values to calibrate left and right images
        """
        calibration = _RectifyCalibration()
        (calibration.cam_mats["left"], calibration.dist_coefs["left"],
         calibration.cam_mats["right"], calibration.dist_coefs["right"],
         calibration.rot_mat, calibration.trans_vec, calibration.e_mat,
         calibration.f_mat) = cv2.stereoCalibrate(self._checkerboard_coordinates,
                                                  self._points["left"],
                                                  self._points["right"],
                                                  calibration.cam_mats["left"],
                                                  calibration.dist_coefs["left"],
                                                  calibration.cam_mats["right"],
                                                  calibration.dist_coefs["right"],
                                                  self._image_size,
                                                  calibration.rot_mat,
                                                  calibration.trans_vec,
                                                  calibration.e_mat,
                                                  calibration.f_mat,
                                                  criteria=calibration.calib_criteria,
                                                  flags=calibration.flags)[1:]

        (calibration.rect_trans["left"], calibration.rect_trans["right"],
         calibration.proj_mats["left"], calibration.proj_mats["right"],
         calibration.disp_to_depth_mat, calibration.valid_boxes["left"],
         calibration.valid_boxes["right"]) = cv2.stereoRectify(calibration.cam_mats["left"],
                                                               calibration.dist_coefs["left"],
                                                               calibration.cam_mats["right"],
                                                               calibration.dist_coefs["right"],
                                                               self._image_size,
                                                               calibration.rot_mat,
                                                               calibration.trans_vec,
                                                               flags=0)
        for side in ("left", "right"):
            (calibration.undistortion_map[side],
             calibration.rectification_map[side]) = cv2.initUndistortRectifyMap(
                                                        calibration.cam_mats[side],
                                                        calibration.dist_coefs[side],
                                                        calibration.rect_trans[side],
                                                        calibration.proj_mats[side],
                                                        self._image_size,
                                                        cv2.CV_32FC1)
        width, height = self._image_size
        focal_length = 0.8 * width
        calibration.disp_to_depth_mat = np.array([[1, 0, 0, -0.5 * width],
                                                  [0, -1, 0, 0.5 * height],
                                                  [0, 0, 0, -focal_length],
                                                  [0, 0, 1, 0]])
        return calibration

    @staticmethod
    def load(cal_folder='calibration'):
        return _RectifyCalibration(input_folder=cal_folder)

    @classmethod
    def _show_image(cls, _title, _image_array):
        cv2.imshow(_title, _image_array)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()

    @property
    def images_found(self):
        return self._images_found


class _DepthMapCorrection:
    """
    Protected member used to implement the calibration of depth map of a stereo vision pair
    """
    calibration_file = 'disparity_cal.pkl'
    LIMITS = {
        'SWS': {'low': -150, 'high': 150, 'inc': 1},
        'PFS': {'low': 5, 'high': 255, 'inc': 1}, # odd between 5 and 255
        'PFC': {'low': 1, 'high': 63, 'inc': 1},  # must be between 1 and 63
        'MDS': {'low': -150, 'high': 50, 'inc': 1},
        'NOD': {'low': 16, 'high': 256, 'inc': 16}, # must be divisible by 16
        'TTH': {'low': 0, 'high': 150, 'inc': 1}, # must be non-negative
        'UR': {'low': 0, 'high': 150, 'inc': 1},  # must be non-negative
        'SR': {'low': 0, 'high': 150, 'inc': 1},  # must be non-negative
        'SPWS': {'low': 0, 'high': 150, 'inc': 1},  # must be non-negative
    }

    def __init__(self, input_folder=''):
        """
        initialize properties used for depth map calibration of stereo vision pairs
        if a input folder is provided, load the values of the stored calibration as self.__dict__
        :param input_folder: str, path to an existing calibration file
        """
        self.SWS = 5
        self.PFS = 5
        self.PFC = 29
        self.MDS = -15
        self.NOD = 128
        self.TTH = 100
        self.UR = 10
        self.SR = 10
        self.SPWS = 100
        if not(input_folder.__eq__('')):
            self._load(input_folder=input_folder)

    def export(self, output_folder):
        """
        export the class instance as a pkl file by dumping all instance properties in self.__init__
        :param output_folder: str, target folder where the file is to be stored
        :return:
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(os.path.join(output_folder, self.calibration_file), 'wb') as cal_file:
            pickle.dump(self.__dict__, cal_file)
        return

    def _load(self, input_folder):
        """
        load the values of a a previous calibration stored in a target folder
        :param input_folder: str, path to a file already containing calibration values
        :return:
        """
        _cal_file = os.path.join(input_folder, self.calibration_file)
        assert os.path.isfile(_cal_file), \
            f"Calibration file: {_cal_file} does not exist!"
        with open(os.path.join(input_folder, self.calibration_file), 'rb') as cal_file:
            _cal = pickle.load(cal_file)
        for key, val in _cal.items():
            self.__dict__[key] = val
        return


class DisparityCalibrator:
    """
    Develop GUI interface for calibration of rectified depth map images using disparity values
    """
    def __init__(self, _path_left='', _path_right=''):
        """
        initialize properties for the disparity calibration of a stereo vision pair
        :param _path_left: str, path to the left image
        :param _path_right: str, path to the right image
        """
        self._DepthMap = _DepthMapCorrection()
        self._controller = {}
        self.cal = None

        left = cv2.imread(_path_left)
        right = cv2.imread(_path_right)
        self.left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        self.right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

    def calibrate(self):
        """
        create tkinter window for adaptive calibration of a depth map generated by a stereo vision pair
        :return: _DepthMapCorrection, as a calibration object
        """
        self.cal = RectifyCalibrator.load()
        root = Tk()
        root.title("Calibration Sequence II")
        for key, val in self._DepthMap.__dict__.items():
            print(key, val)
            Label(root, text=key).pack()
            self._controller[key] = Scale(root, from_=self._DepthMap.LIMITS[key]['low'],
                                          to=self._DepthMap.LIMITS[key]['high'], orient=HORIZONTAL,
                                          resolution=self._DepthMap.LIMITS[key]['inc'],
                                          command=partial(self._get_throttle, _name=key))
            self._controller[key].set(val)
            self._controller[key].pack()
        root.mainloop()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("\nCalibrated Result:")
        for key, val in self._DepthMap.__dict__.items():
            print(key, val)
        return self._DepthMap

    def _get_throttle(self, event, _name):
        """
        add a tkinter throttle to the window used for disparity calibration
        :param event: self, reference to the calling throttle, used to monitor changes in value
        :param _name: str, name of the associated value being changed
        :return:
        """
        _curr = self._controller[_name].get()
        if _name.__eq__('PFS') and _curr % 2 == 0:
            return
        self._DepthMap.__dict__[_name] = _curr
        self._show_depthmap()
        return

    def _show_depthmap(self):
        """
        show the depth map image with properties defined by the user tkinter window
        :return:
        """
        rectified_left = self.cal.rectify(frame=self.left, side='left')
        rectified_right = self.cal.rectify(frame=self.right, side='right')
        sbm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        sbm.setPreFilterType(1)
        sbm.setPreFilterSize(self._DepthMap.PFS)
        sbm.setPreFilterCap(self._DepthMap.PFC)
        sbm.setMinDisparity(self._DepthMap.MDS)
        sbm.setNumDisparities(self._DepthMap.NOD)
        sbm.setTextureThreshold(self._DepthMap.TTH)
        sbm.setUniquenessRatio(self._DepthMap.UR)
        sbm.setSpeckleRange(self._DepthMap.SR)
        sbm.setSpeckleWindowSize(self._DepthMap.SPWS)
        cv2.imshow('rectified_left', rectified_left)
        cv2.imshow('rectified_right', rectified_right)
        disparity = sbm.compute(rectified_left, rectified_right)

        # normalize disparity with local max and min
        local_max = disparity.max()
        local_min = disparity.min()
        depth_map = (disparity - local_min) * (1.0 / (local_max - local_min))

        local_max = depth_map.max()
        local_min = depth_map.min()
        disparity_grayscale = (depth_map - local_min) * (65535.0 / (local_max - local_min))
        disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0 / 65535.0))
        disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
        cv2.imshow("depth map", cv2.convertScaleAbs(disparity_color))
        return

    @staticmethod
    def load(cal_folder='calibration'):
        return _DepthMapCorrection(input_folder=cal_folder)
