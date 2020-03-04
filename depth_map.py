import atexit
import cv2
import dask
import numpy as np

from stereopy_calibrate import DisparityCalibrator, RectifyCalibrator
from stereopy_controls import Controls
from wall_detect import FindWall
from floor_finder import FloorFinder


class Camera:
    left = 2
    right = 0

    @classmethod
    def initialize(cls, _port):
        _cam = cv2.VideoCapture(_port)
        atexit.register(cls.destroy, _cam)
        return _cam

    @classmethod
    def capture(cls, _cam):
        _, frame = _cam.read()
        return frame

    @classmethod
    def destroy(cls, _cam):
        _cam.release()
        return


class PreProcess:
    @classmethod
    def read_image(cls, _path):
        return cv2.imread(_path)

    @classmethod
    def _grayscale(cls, _img):
        return cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY)

    @classmethod
    def _gaussian_blur(cls, _frame):
        return cv2.GaussianBlur(_frame, Controls.gaussian_kernel, Controls.gaussian_std_x)

    @classmethod
    def _bilateral_filter(cls, _frame):
        return cv2.bilateralFilter(_frame, Controls.bl_filter_radii, Controls.bl_sigma_color, Controls.bl_sigma_space)

    @classmethod
    def frame(cls, _img, _filter='g'):
        _img = cls._gaussian_blur(
            _frame=cls._grayscale(
                _img=_img)
            )
        return _img

    @classmethod
    def load_calibration(cls):
        _disparity = DisparityCalibrator.load()
        _rectifier = RectifyCalibrator.load()
        return _disparity, _rectifier


class ApplyCalibrations:

    @classmethod
    def rectify(cls, _rectifier, _image, _side):
        return _rectifier.rectify(_image, _side)

    @classmethod
    def get_sbm(cls, _disparity):
        _sbm = cv2.StereoBM_create(numDisparities=Controls.sv_num_disparities,
                                   blockSize=Controls.sv_block_size)
        _sbm.setPreFilterType(1)
        _sbm.setPreFilterSize(_disparity.PFS)
        _sbm.setPreFilterCap(_disparity.PFC)
        _sbm.setMinDisparity(_disparity.MDS)
        _sbm.setNumDisparities(_disparity.NOD)
        _sbm.setTextureThreshold(_disparity.TTH)
        _sbm.setUniquenessRatio(_disparity.UR)
        _sbm.setSpeckleRange(_disparity.SR)
        _sbm.setSpeckleWindowSize(_disparity.SPWS)
        return _sbm

    @classmethod
    def disparity(cls, _sbm, _left, _right):
        _disparity = _sbm.compute(_left, _right)

        # normalize disparity with local max and min
        local_max = _disparity.max()
        local_min = _disparity.min()
        depth_map = (_disparity - local_min) * (1.0 / (local_max - local_min))

        local_max = depth_map.max()
        local_min = depth_map.min()
        disparity_grayscale = (depth_map - local_min) * (65535.0 / (local_max - local_min))
        return disparity_grayscale


class PostProcess:

    @staticmethod
    def depth_map(_disparity):
        disparity_fixtype = cv2.convertScaleAbs(_disparity, alpha=(255.0 / 65535.0))
        disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
        return disparity_color

    def __init__(self):
        self._rows = 2
        self._cols = 7

        self.disparity, self.rectifier = PreProcess.load_calibration()
        self.sbm = ApplyCalibrations.get_sbm(_disparity=self.disparity)

    def image_rectifier(self, _img, _side):
        return ApplyCalibrations.rectify(_rectifier=self.rectifier, _image=PreProcess.frame(_img=_img), _side=_side)

    def get_disparity(self, _left_rectified, _right_rectified):
        return ApplyCalibrations.disparity(_sbm=self.sbm, _left=_left_rectified, _right=_right_rectified)

    def depth_intensity(self, _disparity, _floor):
        _h_step = _disparity.shape[0]//self._rows
        _l_step = int(np.floor(_disparity.shape[1]/self._cols))
        _norm = np.array([[0.0]*self._cols]*(self._rows+1))
        for i in range(self._rows):
            for j in range(self._cols):
                if i == 1:
                    _norm[self._rows][j] = np.mean(_floor[i*_h_step:(i+1)*_h_step, j*_l_step:(j+1)*_l_step])
                _norm[i][j] = np.mean(_disparity[i*_h_step:(i+1)*_h_step, j*_l_step:(j+1)*_l_step])
        return _norm


if __name__ == '__main__':
    processor = PostProcess()
    left = Camera.initialize(Camera.left)
    right = Camera.initialize(Camera.right)
    _ = Camera.capture(left)
    img = Camera.capture(right)
    _hgt, _len, _ = img.shape
    while True:
        left_ = Camera.capture(left)
        right_ = Camera.capture(right)
        left_rectified = dask.delayed(processor.image_rectifier)(left_, 'left')
        right_rectified = dask.delayed(processor.image_rectifier)(right_, 'right')
        floor_zone = dask.delayed(FloorFinder.find)(right_, (_hgt, _len))
        wall_warning = dask.delayed(FindWall.find)(left_)
        disparity_ = dask.delayed(processor.get_disparity)(_left_rectified=left_rectified, _right_rectified=right_rectified)
        depth_map_ = dask.delayed(processor.depth_map)(disparity_)
        vibrations = processor.depth_intensity(depth_map_.compute(), floor_zone.compute())
        haptic_output = dask.compute(*vibrations)
        cv2.imshow('depth', depth_map_.compute())
        cv2.imshow('floor', floor_zone.compute())
        cv2.imshow('raw', left_)
        if cv2.waitKey(20) == 27:
            break

    cv2.destroyAllWindows()
