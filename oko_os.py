import cv2
import atexit


class Controls:
    gaussian_kernel = (5, 5)
    gaussian_std_x = 0
    bl_filter_radii = 9
    bl_sigma_color = 75
    bl_sigma_space = 75
    sv_num_disparities = 16  # this needs to be a factor of 8
    sv_block_size = 15  # this needs to be odd
    canny_limits = (50, 120)


class Safety:
    @classmethod
    def system_stop(cls, cam_id1, cam_id2):
        cv2.destroyAllWindows()
        cam_id1.shutdown()
        cam_id2.shutdown()


class Camera:
    def __init__(self, port=None):
        self._port = port
        self._device = None

    def initialize(self):
        self._device = cv2.VideoCapture(self._port)

    def operational(self):
        return self._device.isOpened()

    def capture(self):
        _, frame = self._device.read()
        return frame

    def shutdown(self):
        self._device.release()


class Processing:
    @classmethod
    def _gray_scale(cls, _frame):
        return cv2.cvtColor(_frame, cv2.COLOR_RGB2GRAY)

    @classmethod
    def _gaussian_blur(cls, _frame):
        return cv2.GaussianBlur(_frame, Controls.gaussian_kernel, Controls.gaussian_std_x)

    @classmethod
    def _bilateral_filter(cls, _frame):
        return cv2.bilateralFilter(_frame, Controls.bl_filter_radii, Controls.bl_sigma_color, Controls.bl_sigma_space)

    @classmethod
    def filter_frame(cls, _frame):
        _g = Processing._gray_scale(_frame)
        _gb = Processing._gaussian_blur(_g)
        _bl = Processing._bilateral_filter(_gb)
        return _bl


class StereoVision:
    @classmethod
    def disparity_map(cls, _left, _right):
        stereo = cv2.StereoBM_create(numDisparities=Controls.sv_num_disparities, blockSize=Controls.sv_block_size)
        disparity = stereo.compute(_left, _right)
        return cv2.convertScaleAbs(disparity)


if __name__ == '__main__':
    left = Camera(0)
    right = Camera(2)
    try:
        left.initialize()
        right.initialize()
        if left.operational() and right.operational():
            atexit.register(Safety.system_stop, left, right)
        else:
            raise AttributeError("Cameras are not operational")
    except AttributeError as e:
        print(e)
        exit()

    key = 0
    while not key == 27:
        left_image = left.capture()
        right_image = right.capture()
        left_processed = Processing.filter_frame(left_image)
        right_processed = Processing.filter_frame(right_image)
        depth_map = StereoVision.disparity_map(left_processed, right_processed)
        test = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
        cv2.imshow("depth map", test)
        cv2.imshow("left", left_image)
        cv2.imshow("right", right_image)
        key = cv2.waitKey(20)
