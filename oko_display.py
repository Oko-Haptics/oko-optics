import oko_os
import atexit
import cv2
import oko_disparity as oko_d


def run():
    oko_dm = oko_d.DepthMap()
    left = oko_os.Camera(0)
    right = oko_os.Camera(2)
    try:
        left.initialize()
        right.initialize()
        if left.operational() and right.operational():
            atexit.register(oko_os.Safety.system_stop, left, right)
        else:
            raise AttributeError("Cameras are not operational")
    except AttributeError as e:
        print(e)
        exit()

    key = 0
    while not key == 27:
        left_image = left.capture()
        right_image = right.capture()
        left_processed = oko_os.Processing.filter_frame(left_image)
        right_processed = oko_os.Processing.filter_frame(right_image)
        depth_map = oko_dm.get_disparity((left_processed, right_processed))
        local_max = depth_map.max()
        local_min = depth_map.min()
        disparity_grayscale = (depth_map - local_min) * (65535.0 / (local_max - local_min))
        disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0 / 65535.0))
        disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
        cv2.imshow("depth map", cv2.convertScaleAbs(disparity_color))
        cv2.imshow("left", left_image)
        cv2.imshow("right", right_image)
        key = cv2.waitKey(20)


if __name__ == '__main__':
    run()
