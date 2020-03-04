import cv2
import os
import numpy as np


class Setup:
    """
    Define setup and limit for the coco cnn object detection functionality
    """
    CONF_THRESH = 0.25  # lower acceptable confidence for object recognition
    NMS_THRESH = 0.4  # keep the best identified box for an object
    INP_WIDTH = 416  # width of the dnn blob
    INP_HEIGHT = 416  # height of the dnn blob


class CNN:
    """
    define the coco cnn for detecting objects
    """
    def _mark_objects(self, _img, _objects):
        """
        mark rectangles on the frame to mark locations of objects detected by coco
        :param _img: frame in which objects are to be detected
        :param _objects: string list of the neuron names which were itentified by coco and yolo
        :return: processed frame
        """
        _frame = np.copy(_img)
        with open(self.class_file, 'rt') as f:
            coco_classess = f.read().rstrip('\n').split('\n')

        def draw_prediction(class_id, pred_confidence, _left, _top, _right, _bottom):
            """
            draw a rectangle around the object which was identified at its target location
            :param class_id: name of the object which was identified
            :param pred_confidence: the confidence that it was identifed
            :param _left: left side of the rectangle
            :param _top: top side of the rectangle
            :param _right: right side of the rectangle
            :param _bottom: bottom side of the rectangle
            :return:
            """
            cv2.rectangle(_frame, (_left, _top), (_right, _bottom), (255, 178, 50), 3)
            assert class_id < len(coco_classess)  # ensure that the class identified is within the coco library
            label = f"{coco_classess[class_id]} {round(pred_confidence, 2)}"
            cv2.putText(_frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            print(f"{label}  at")
            print(_left, _top, _right, _bottom)
            return

        # get size of the frame from the object
        frame_height = _frame.shape[0]
        frame_width = _frame.shape[1]

        ident_classes = []  # objects from the coco model which were identified in the image
        ident_confidence = []  # corresponding confidences for the coco objects which were identified
        boxes = []  # location of the boxes around the coco images developed by the yolo algorithm
        for obj in _objects:
            for detect in obj:
                scores = detect[5:]
                _class = np.argmax(scores)
                _conf = scores[_class]

                if _conf > Setup.CONF_THRESH:  # if the maximum confidence in the detection is above threshold
                    center_x = int(detect[0] * frame_width)
                    center_y = int(detect[1] * frame_height)
                    width = int(detect[2] * frame_width)
                    height = int(detect[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)

                    ident_classes.append(_class)
                    ident_confidence.append(float(_conf))
                    boxes.append([left, top, width, height])

        # formats the identified boxes and clears out the duplicate boxes around the same object using nms suppression
        indices = cv2.dnn.NMSBoxes(boxes, ident_confidence, Setup.CONF_THRESH, Setup.NMS_THRESH)
        for i in indices:
            # for the objects which were confirmed to be identified, draw boxes around them
            target_index = i[0]
            box = boxes[target_index]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            draw_prediction(ident_classes[target_index], ident_confidence[target_index],
                            _left=left, _top=top, _right=left + width, _bottom=top + height)
        return _frame

    @staticmethod
    def _get_dnn_neurons(_coco):
        """
        get names of output layers (neurons) which have been triggered
        :param _coco: deep neural network object
        :return: list of names of the objects which were identified by the deep neural network
        """
        named_layers = _coco.getLayerNames()
        return [named_layers[i[0] - 1] for i in _coco.getUnconnectedOutLayers()]

    @staticmethod
    def _identify_objects(_img, _coco):
        """
        parse image with yolov3 neural net to identify the objects which are identified
        :param _img: cv2 array defining the image being processed
        :param _net: cv2 deep neural network instance defined by the yolov3 weights and configurations
        :return: outs_list listing the objects which have been identified and their location in the image
        """
        # create a 4D blob from the image - subtracts the mean RGB color from the image with respect to a scale
        blob = cv2.dnn.blobFromImage(_img, 1 / 255, (Setup.INP_WIDTH, Setup.INP_HEIGHT), [0, 0, 0], 1, crop=False)
        # set the blob as the input for the neural network
        _coco.setInput(blob)
        outs = _coco.forward(CNN._get_dnn_neurons(_coco))
        return outs

    def __init__(self):
        self._coco = None
        _cwd = os.getcwd()
        self.class_file = os.path.join(_cwd, "coco.names")
        self._model_cfg = os.path.join(_cwd, "yolov3.cfg")
        self._model_weights = os.path.join(_cwd, "yolov3.weights")

    def _initialize(self):
        """
        initialize the coco cnn for object recognition
        :return:
        """
        self._coco = cv2.dnn.readNetFromDarknet(self._model_cfg, self._model_weights)
        self._coco.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._coco.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return

    def initialize(self):
        """
        client side request for initializing the cnn
        :return: boolean indicating success of operation
        """
        try:
            self._initialize()
        except cv2.error as e:
            print(e)
            return False
        return True

    def get_objects(self, _img):
        """
        client side request for identifying objects
        :param _img: image array in which objects need to be identified
        :param _display_image: image array on which objects are to be highlighted
        :return: boolean indicating success of operation
        """
        _targets = CNN._identify_objects(_img=_img, _coco=self._coco)
        _marked_frame = self._mark_objects(_img=_img, _objects=_targets)
        return _marked_frame


def self_test():
    poop = cv2.imread(os.path.join(os.getcwd(), 'test_right.png'))
    coco_obj = CNN()
    coco_obj.initialize()
    marked = coco_obj.get_objects(poop)
    cv2.imshow("read result", poop)
    cv2.imshow("coco result", marked)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
    print("detection process complete")
