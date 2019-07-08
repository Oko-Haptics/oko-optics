import cv2
import os
import matplotlib.pyplot as plt
from stereovision.calibration import StereoCalibration

class DepthMap:
    SWS = 5
    PFS = 5
    PFC = 29
    MDS = -15 #-25
    NOD = 128
    TTH = 100
    UR = 10
    SR = 25 #15
    SPWS = 100

    def __init__(self):
        self.calibration = StereoCalibration(input_folder='calib_result')

    def get_disparity(self, _disparity_pair):
        rectified_pair = self.calibration.rectify(_disparity_pair)
        sbm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        sbm.setPreFilterType(1)
        sbm.setPreFilterSize(DepthMap.PFS)
        sbm.setPreFilterCap(DepthMap.PFC)
        sbm.setMinDisparity(DepthMap.MDS)
        sbm.setNumDisparities(DepthMap.NOD)
        sbm.setTextureThreshold(DepthMap.TTH)
        sbm.setUniquenessRatio(DepthMap.UR)
        sbm.setSpeckleRange(DepthMap.SR)
        sbm.setSpeckleWindowSize(DepthMap.SPWS)

        # develop the disparity map
        rectified_left = rectified_pair[0]
        rectified_right = rectified_pair[1]
        disparity = sbm.compute(rectified_left, rectified_right)

        # normalize disparity with local max and min
        local_max = disparity.max()
        local_min = disparity.min()
        disparity_visual = (disparity-local_min)*(1.0/(local_max-local_min))
        return disparity_visual


if __name__ == '__main__':
    cwd = os.getcwd()
    left_image_path = os.path.join(cwd, 'test_left.png')
    right_image_path = os.path.join(cwd, 'test_right.png')
    _left = cv2.imread(left_image_path, 0)
    _right = cv2.imread(right_image_path, 0)
    DM = DepthMap()
    depth_map = DM.get_disparity((_left, _right))
    plt.imshow(depth_map, aspect='equal', cmap='jet')
    plt.show()
