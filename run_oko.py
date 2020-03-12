import cv2
from arduino_talker import ArduinoTalker
from depth_map import Camera, PostProcess
from floor_finder import FloorFinder
from wall_detect import FindWall


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
    floor_zone = []
    while True:
        if c == 0:
            left_ = Camera.capture(left)
            left_rectified = processor.image_rectifier(left_, 'left')
            floor_zone = FloorFinder.find(left_, (_hgt, _len))
        else:
            right_ = Camera.capture(right)
            right_rectified = processor.image_rectifier(right_, 'right')
            wall_warning = FindWall.find(right_)
        disparity = processor.get_disparity(_left_rectified=left_rectified, _right_rectified=right_rectified)
        depth_map = processor.depth_map(disparity)
        vibrations = processor.depth_intensity(depth_map, floor_zone)
        haptic_drive = arduino_service.encode(vibrations)
        cv2.imshow("left", left_)
        cv2.imshow("right", right_)
        cv2.imshow('depth', processor.depth_map2color(depth_map))
        if cv2.waitKey(20) == 27:
            break
        c = (c+1) % 2
    cv2.destroyAllWindows()
