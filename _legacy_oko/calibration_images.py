import atexit
import time
import cv2
import os
import numpy as np
import oko_os as ok

cal_dir = os.path.join(os.getcwd(), 'calibration_images')
left = ok.Camera(0)
right = ok.Camera(2)
font = cv2.FONT_HERSHEY_SIMPLEX

try:
    left.initialize()
    right.initialize()
    if left.operational() and right.operational():
        atexit.register(ok.Safety.system_stop, left, right)
    else:
        raise AttributeError("Cameras are not operational")
except AttributeError as e:
    print(e)
    exit()

target_images = 30
_count = 0
_base = time.time()
while _count < target_images:
    _lim = 10 if _count == 0 else 5
    left_image = left.capture()
    right_image = right.capture()
    dummy = np.copy(left_image)
    _trigger_time = time.time()
    cv2.putText(dummy, str(int(_lim - (_trigger_time-_base))), (50, 50), font, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.imshow('left', dummy)
    cv2.imshow('right', right_image)
    if cv2.waitKey(20) == 27:
        break
    if (_trigger_time - _base) > _lim:
        _count += 1
        cv2.imwrite(os.path.join(cal_dir, f"left_{str(_count)}.png"), left_image)
        cv2.imwrite(os.path.join(cal_dir, f"right_{str(_count)}.png"), right_image)
        _base = time.time()
        print(f"took image pair {_count}")

