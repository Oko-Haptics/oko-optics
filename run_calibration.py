import cv2
import os
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError


class ChessBoard:
    rows = 6
    columns = 9
    square = 2.34


def get_cal_dir():
    return os.path.join(os.getcwd(), 'calibration_images')


def num_images():
    return len(os.listdir(get_cal_dir()))/2


test_image = os.path.join(os.getcwd(), 'test.png')
width, height, _ = cv2.imread(test_image).shape
calibrator = StereoCalibrator(ChessBoard.rows, ChessBoard.columns, ChessBoard.square, (width, height))
_count = 0
total_photos = num_images()
cal_dir = get_cal_dir()
while _count != total_photos:
    _count += 1
    print(f"attempting to analyze calibration images: {_count}")
    _left = os.path.join(cal_dir, f"left_{_count}.png")
    _right = os.path.join(cal_dir, f"right_{_count}.png")
    try:
        if not(os.path.isfile(_left) and os.path.isfile(_right)):
            raise RuntimeError("Missing Image Pair:", _count)
        left_img = cv2.imread(_left)
        right_img = cv2.imread(_right)
        calibrator._get_corners(left_img)
        calibrator._get_corners(right_img)
        calibrator.add_corners((left_img, right_img), show_results=True)
    except ChessboardNotFoundError as e:
        print(f"error in finding chessboard in pair {_count}: {e}")
        print("skipping this pair of images")
    except RuntimeError as e:
        print(e)
        break
    print(f"finished analysis of {_count}")

print("Calibration instance defined!")

print('Starting calibration... It can take several minutes!')
calibration = calibrator.calibrate_cameras()
calibration.export('calib_result')
print('Calibration complete!')


# Lets rectify and show last pair after  calibration
calibration = StereoCalibration(input_folder='calib_result')
rectified_pair = calibration.rectify((left_img, right_img))

cv2.imshow('Left CALIBRATED', rectified_pair[0])
cv2.imshow('Right CALIBRATED', rectified_pair[1])
cv2.imwrite("rectifyed_left.jpg",rectified_pair[0])
cv2.imwrite("rectifyed_right.jpg",rectified_pair[1])
cv2.waitKey(0)
