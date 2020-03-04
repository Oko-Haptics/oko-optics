import oko_os
import atexit
import cv2
import time
import queue
from threading import Thread
import oko_disparity as oko_d
import oko_coco_implementation as oko_coco


class RunOko:
    def __init__(self):
        self.right_image = None
        self.left_image = None
        self.key = 0
        print("Initializing oko neural network")
        self.oko_dnn = oko_coco.CNN()
        self.oko_dnn.initialize()
        time.sleep(0.5)
        print("neural network initialized")
        self.oko_dm = oko_d.DepthMap()
        self.left_camera = oko_os.Camera(0)
        self.right_camera = oko_os.Camera(2)
        try:
            self.left_camera.initialize()
            self.right_camera.initialize()
            if self.left_camera.operational() and self.right_camera.operational():
                atexit.register(oko_os.Safety.system_stop, self.left_camera, self.right_camera)
            else:
                raise AttributeError("Cameras are not operational")
        except AttributeError as e:
            print(e)
            exit()
        self._run_imaging()

    def _run_imaging(self):
        _coco_queue = queue.Queue()
        _recognized = self.left_camera.capture()
        Thread(target=self._object_recognition, args=(_recognized, _coco_queue,)).start()
        while not(self.key == 27):
            self.right_image = self.right_camera.capture()
            self.left_image = self.left_camera.capture()
            _left_processed = oko_os.Processing.filter_frame(self.left_image)
            _right_processed = oko_os.Processing.filter_frame(self.right_image)
            depth_map = self.oko_dm.get_disparity((_left_processed, _right_processed))
            local_max = depth_map.max()
            local_min = depth_map.min()
            disparity_grayscale = (depth_map - local_min) * (65535.0 / (local_max - local_min))
            disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0 / 65535.0))
            disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
            cv2.imshow("depth map", cv2.convertScaleAbs(disparity_color))

            if _coco_queue.empty() is False:
                _recognized = _coco_queue.get()
                Thread(target=self._object_recognition, args=(self.left_image, _coco_queue,)).start()
            cv2.imshow("recognized objects", _recognized)
            self.key = cv2.waitKey(50)

    def _object_recognition(self, _image, _image_que):
        left_post_processed = self.oko_dnn.get_objects(_image)
        _image_que.put(left_post_processed)
        return


if __name__ == '__main__':
    RunOko()
