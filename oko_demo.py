import cv2
import numpy as np
import serial
from arduino_talker import ArduinoTalker
from depth_map import Camera, PostProcess


port = 'COM4'
ard = serial.Serial(port, 9600, timeout=1)


def demo_depth(_disparity, _rows, _cols):
    _h_step = _disparity.shape[0] // _rows
    _l_step = int(np.floor(_disparity.shape[1] / _cols))
    _norm = np.array([[0.0] * _cols] * _rows)
    for i in range(_rows):
        for j in range(_cols):
            _norm[i][j] = np.mean(_disparity[i * _h_step:(i + 1) * _h_step, j * _l_step:(j + 1) * _l_step])
    return _norm


if __name__ == '__main__':
    c = 0
    arduino_service = ArduinoTalker()
    processor = PostProcess()
    left = Camera.initialize(Camera.left)
    right = Camera.initialize(Camera.right)
    right_ = Camera.capture(right)
    right_rectified = processor.image_rectifier(right_, 'right')
    left_ = Camera.capture(left)
    left_rectified = processor.image_rectifier(left_, 'left')
    _hgt, _len, _ = right_.shape

    while True:
        try:
            left_ = Camera.capture(left)
            left_rectified = processor.image_rectifier(left_, 'left')
            right_ = Camera.capture(right)
            right_rectified = processor.image_rectifier(right_, 'right')
            disparity = processor.get_disparity(_left_rectified=left_rectified, _right_rectified=right_rectified)
            depth_map = processor.depth_map(disparity)
            vibrations = demo_depth(depth_map, 9, 15)
            haptic_drive = arduino_service.encode(vibrations)
            ard.write(haptic_drive)
            cv2.waitKey(50)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            break
    cv2.destroyAllWindows()
    ard.close()
