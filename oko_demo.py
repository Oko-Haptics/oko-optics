import cv2
import numpy as np
import serial
import time
from arduino_talker import ArduinoTalker
from depth_map import Camera, PostProcess


port = 'COM6'
ard = serial.Serial(port, 9600)
_reset = b'0000000000000009'
time.sleep(2)


def demo_depth(_disparity, _rows, _cols):
    _h_step = _disparity.shape[0] // _rows
    _l_step = int(np.floor(_disparity.shape[1] / _cols))
    _norm = np.array([[0.0] * _cols] * _rows)
    for i in range(_rows):
        for j in range(_cols):
            _norm[i][j] = np.mean(_disparity[i * _h_step:(i + 1) * _h_step, j * _l_step:(j + 1) * _l_step])
    return _norm.T  # the transpose is added due to the orientation of the demo rig


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
            cv2.imshow('Oko View', processor.depth_map2color(depth_map))
            cv2.imshow('left', left_)
            vibrations = demo_depth(depth_map, 3, 5)
            haptic_drive = arduino_service.encode(vibrations)
            ard.write(haptic_drive)
            if cv2.waitKey(20) == 27:
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            break
    cv2.destroyAllWindows()
    ard.write(_reset)
    ard.close()
