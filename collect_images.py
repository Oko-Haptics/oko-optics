import cv2
import numpy as np
import os
import time


class Setup:
    num_images = 30
    count_down = 5


class Camera:
    left = 2
    right = 1


class Display:
    position = (640, 240)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 10
    font_color = (255, 0, 0)  # blue
    font_width = 5


if __name__ == '__main__':
    target_dir = os.path.join(os.getcwd(), 'calibration_images')
    left = cv2.VideoCapture(Camera.left)
    right = cv2.VideoCapture(Camera.right)
    _, _ = left.read()
    _, _ = right.read()
    time.sleep(5)
    img_num = 0
    while img_num <= Setup.num_images:
        print(f'Capturing image pair: {img_num}')
        stop_ = Setup.count_down
        count = 0
        start_ = time.time()
        img_left, img_right = None, None
        while count < stop_:
            ret, img_left = left.read()
            ret_, img_right = right.read()
            combo = np.hstack((img_right, img_left))
            trigger_ = time.time()
            cv2.putText(combo, str(stop_-count), Display.position, Display.font,
                        Display.font_size, Display.font_color, Display.font_width)
            cv2.imshow('capture', combo)
            count = int(trigger_ - start_)

            if cv2.waitKey(20) == 27:
                left.release()
                right.release()
                cv2.destroyAllWindows()
                raise RuntimeError("Execution aborted")
        if img_num == 0:
            pass
        else:
            cv2.imwrite(os.path.join(target_dir, f'left_{img_num}.png'), img_left)
            cv2.imwrite(os.path.join(target_dir, f'right_{img_num}.png'), img_right)
        img_num += 1

    left.release()
    right.release()
    cv2.destroyAllWindows()
