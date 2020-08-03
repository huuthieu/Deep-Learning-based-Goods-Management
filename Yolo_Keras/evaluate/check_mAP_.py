import colorsys
import os
import cv2
import time
import read_plate
import h5py

# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.Session(config = config)

from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess
from Tesseract_OCR import Image_to_string
import read_plate



class YOLO(object):
    _defaults = {
        "model_path": 'model_data\yolov3-tiny-obj_50000.h5',

        "anchors_path": 'model_data/anchors.txt',
        "classes_path": 'model_data/class_list.txt',
        "score": 0.35,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "text_size": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        # self.anchors = np.array([[87, 25], [70, 235], [135, 186], [91, 321], [154, 315], [264, 360]])
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=0.25)
        return boxes, scores, classes

    def detect_image(self, image):
        # if self.model_image_size != (None, None):
        #     assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        #     assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        #     boxed_image = image_preporcess(np.copy(image), tuple(reversed(self.model_image_size)))
        #     image_data = boxed_image

        # image_data = image/255
        # image_data = np.expand_dims(image_data,0)
        frame = cv2.resize(image,(416,416))
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = image_preporcess(np.copy(frame), tuple(reversed(self.model_image_size)))
            image_data = boxed_image

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [frame.shape[0], frame.shape[1]],  # [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        Rx = image.shape[1] / 416
        Ry = image.shape[0] / 416

        left_text = top_text = width_text = height_text = _conf_text = 0
        _class_text = None
        for i, c in reversed(list(enumerate(out_classes))):

            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            # label = '{}'.format(predicted_class)
            scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            top = int(top * Ry)
            left = int(left * Rx)
            bottom = int(bottom * Ry)
            right = int(right * Rx)

            if c == 14:
            # put object rectangle
                left_text = left
                top_text = top
                right_text = right
                bottom_text = bottom
                _class_text = c
                _conf_text = score
            # image = frame[top:top + height, left:left + width]  # crop image to recognize

            # ve bounding boxes

            # drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)

            # if len(indices) and having_text != 1:
            #     detect = 1
            #     having_text = 0
            # else:
            #     detect = 0
        print(_class_text)
        # print(len(indices))
        if _class_text == None:
            return [_class_text]
        else:
            return [_class_text, left_text, top_text, right_text, bottom_text, _conf_text]

    def close_session(self):
        self.sess.close()

    def detect_img(self, image):
        # image = cv2.imread(image, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        detect = self.detect_image(original_image)
        # r_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return detect








def calculate_IoU(pos1,pos2):
    x_min1,y_min1,x_max1,y_max1 = pos1
    x_min2, y_min2, x_max2, y_max2 = pos2
    x_min = max(x_min1,x_min2)
    x_max = min(x_max1,x_max2)
    y_min = max(y_min1,y_min2)
    y_max = min(y_max1,y_max2)
    union = max(0,(x_max-x_min))*max(0,(y_max-y_min))
    area1 = (x_max1-x_min1)*(y_max1-y_min1)
    area2 = (x_max2-x_min2) * (y_max2 - y_min2)
    area = area1 + area2 - union
    IoU = union/area
    return IoU

def compute_ap(recall,precision):
    mrec = np.concatenate(([0.],recall,[1.]))
    mpre = np.concatenate(([0.],precision,[0.]))

    for i in range(mpre.size - 1,0,-1):
        mpre[i-1] = np.maximum(mpre[i-1],mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i+1]-mrec[i])*mpre[i+1])
    return ap

# Set up the net



# Process inputs


# dir = 'test_image/'
# paths = [os.path.join(dir,path) for path in os.listdir(dir)]
#
# list_image = [os.path.join(dir,path) for path in os.listdir(dir)]
# num = 0
# j = 0
#
# classes = [8,0,3,9,1,12,5,6,2,7]
# for path in paths:
#     img_path = [os.path.join(path,img) for img in os.listdir(path)]
#     # print(img_path)
#     size = len(img_path)
#
#     for img in img_path:
#
#         frame = cv.imread(img)
#         blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
#
#     # Set the input the the net
#         net.setInput(blob)
#         outs = net.forward(getOutputsNames(net))
#
#     # postprocess(frame,outs)
#         _class = postprocess(frame, outs)  # draw and set flag detect when no object
#         print(_class)
#         if _class == classes[j]:
#             num +=1
#     print(num/size)
#     j +=1
#     num = 0

    # show the image


### Test mAP


img_dir = 'test_image/Romano'
img_paths = [os.path.join(img_dir,path) for path in os.listdir(img_dir)]

dir = 'bbox_txt/Romano.txt'
with open(dir) as f:
    txt = f.read().splitlines()
all_annotations = txt

# calculate IoU
# IoU = []
# for i,img in enumerate(img_paths):
#     frame = cv.imread(img)
#     blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
#
#         # Set the input the the net
#     net.setInput(blob)
#     outs = net.forward(getOutputsNames(net))
#
#         # postprocess(frame,outs)
#     pred = postprocess(frame, outs)
#     if pred[0] == None:
#         mAP = 0
#     else:
#         pos = [float(position) for position in txt[i].split(',')]
#         IoU.append(calculate_IoU(pred[1:],pos))
#
# print(np.sum(IoU)/len(IoU))

# calculate mAP
yolo = YOLO()
all_detections = []
for img in img_paths:
    frame = cv2.imread(img)

    # frame = cv2.resize(frame, (416, 416))
    pred = yolo.detect_img(frame)
    all_detections.append(pred)

false_positives = np.zeros((0,))
true_positives  = np.zeros((0,))
scores          = np.zeros((0,))
num_annotations = 0.0

for i in range(len(img_paths)):
    detections = all_detections[i]
    annotations = all_annotations[i]
    num_annotations += 1


    if detections[0] == None:
        # false_positives = np.append(false_positives, 1)
        # true_positives = np.append(true_positives, 0)

        continue

    detected_annotations = []
    # scores = np.append(scores,detections[5])
    pos = [float(position) for position in annotations.split(',')]
    IoU = calculate_IoU(detections[1:5], pos)
    if IoU >= 0.7 and detections[0] == 14:
        false_positives = np.append(false_positives, 0)
        true_positives = np.append(true_positives, 1)
        scores = np.append(scores, detections[5])
    elif IoU < 0.7 and detections[0] == 14:
        false_positives = np.append(false_positives, 1)
        true_positives = np.append(true_positives, 0)
        scores = np.append(scores, detections[5])
        print(i)
    else:
        print(detections[5],detections[0])

indices         = np.argsort(-scores)
print(len(scores))
print(len(false_positives))
print(num_annotations)
false_positives = false_positives[indices]
true_positives  = true_positives[indices]

false_positives = np.cumsum(false_positives)
true_positives = np.cumsum(true_positives)


recall = true_positives/num_annotations
precision = true_positives/(np.maximum(true_positives + false_positives, np.finfo(np.float64).eps))
# print('recall',recall)
print('precision',precision)
average_precision  = compute_ap(recall, precision)

print(average_precision)


















