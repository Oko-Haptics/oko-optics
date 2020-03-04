import cv2
import os

from stereopy_calibrate import RectifyCalibrator
from stereopy_calibrate import DisparityCalibrator


def show_image(cb, image_name='chessboard'):
    cv2.imshow(image_name, cb.draw_chessboard())
    if cv2.waitKey(0):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Calibrating Image Rectification")
    _target = os.path.join(os.getcwd(), 'calibration_images')
    _rows = 7
    _cols = 10
    _edge = 2.54  # cm
    _left = []
    _right = []
    [_left.append(os.path.join(_target, t)) if 'left' in t else _right.append(os.path.join(_target, t))
     for t in os.listdir(_target)]
    print(len(_left))
    print(len(_right))
    c = RectifyCalibrator(row=_rows, col=_cols, edge=_edge, left_imgs=_left, right_imgs=_right,
                          show_results=False)
    c.find_corners()
    print("Calibrating this may take several minutes")
    cal = c.calibrate()
    print("Exporting")
    cal.export(output_folder=os.path.join(os.getcwd(), 'calibration'))
    print("Finished Exporting Rectification Calibration")

    print("Calibrating Depth Maps")
    cd = DisparityCalibrator(_path_left=os.path.join(os.getcwd(), 'calibration_images/left_2.png'),
                             _path_right=os.path.join(os.getcwd(), 'calibration_images/right_2.png'))

    dm = cd.calibrate()
    dm.export(output_folder=os.path.join(os.getcwd(), 'calibration'))
    print("Finished Exporting Depth Map Calibration")
