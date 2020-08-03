import PySimpleGUI as sg
import colorsys
import os
import cv2
import time

import h5py

from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval,tiny_yolo_body
from yolo3.utils import image_preporcess

import urllib.request as urllib
import scipy.ndimage
from IP_camera import get_frame_from_IP_camera
from Tesseract_OCR import Image_to_string
import detect_text,detect_barcode
import OCR_CNN
from export_excel import save
# import DEMO_text
# import detect_product,barcode
from barcode_detect import barcode_detect

def convert2int(list):
    int_list = [int(i) for i in list]
    return int_list

def make_no_obj_text(no_obj):
    no_obj = ','.join(no_obj)
    return no_obj

def unmake(string):
    if string == '...':
        return None
    else:
        list_obj_string = string.split(',')
        list_obj = [int(i) for i in list_obj_string]
        return list_obj

def unmake_string(string):
    list_obj_string = string.split(',')
    return list_obj_string

def check(pair):
    label, value = pair
    if (label, value) in dict.items() or (value, label) in dict.items():
        return True
    else:
        return False


def convert(barcode):
    dict = {'8938501434012': 'coca','89345885903024':'pepsi','8934822901332':'bia Larue','8934803043891':'Pillow',
                '8934804025766':'Milo','8938501434012':'aquafina','8934588593024':'Oishi',
            '8934588593024':'sting','8934822101336':'tiger','error': '...'}
    barcode_convert = []
    for bar_number in barcode:
        barcode_convert.append(dict[str(bar_number)])
    return barcode_convert


def check_most_freq(list):
    return max(set(list),key = list.count)
dict = {'coca_cola': '22000', 'mirinda': '12000', 'sting': '11000', 'fanta': '8000',
        'red_bull': '10000', 'saigon_special': '16000', 'tiger': '18000', 'aquafina': '28000',
        'highland_coffee': '15000', 'haohao': '4000', 'oishi_pillow': '10000', 'pepsi': '22000',
        'TH': '9000', 'Romano': '99000'}

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo_57000.h5',

        "anchors_path": 'model_data/anchor_16.txt',
        "classes_path": 'model_data/class_list.txt',
        "score": 0.2,
        "iou": 0.4,
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
        self.num_obj = 0
        self.start_tracking = 0
        self.count_update_notmatch = 0

        self.barcode_list = []
        self.match_text_list = []
        self.center_axes = 0
        self.wait_moving = 0



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

        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        # hsv_tuples = [(x / len(self.class_names), 1., 1.)
        #               for x in range(len(self.class_names))]
        # self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # self.colors = list(
        #     map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        #         self.colors))
        #
        # np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        frame = cv2.resize(image, (416, 416))

        # image_data = image/255
        # image_data = np.expand_dims(image_data,0)
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

        thickness = (image.shape[0] + image.shape[1]) // 600

        # print(out_classes)

        # left_axes = np.array([max(0, np.floor(Rx*out_boxes[i][1] + 0.5).astype('int32')) for i,c in enumerate(out_classes) if c == 4])
        center_axes = np.array([max(0, np.floor((Rx*out_boxes[i][1] + Rx*(out_boxes[i][3]))/2+0.5).astype('int32')) for i,c in enumerate(out_classes) if c == 4])


        # left_axes.sort()
        center_axes.sort()
        # print(center_axes)

        num_obj_true = 3 * len(center_axes)
        num_barcode = len([c for c in out_classes if c == 15])
        hieu = num_barcode - len(center_axes)

        if len(out_classes) - hieu < num_obj_true:
            detect = 0
        else:
            detect = 1

        no_obj = [str(i + self.num_obj) for i in range(len(center_axes))]

        list_text = []
        list_obj = []
        pair = []
        # left_list = []
        # left_list_barcode = []
        center_list = []
        center_list_barcode = []
        remove_list = []
        barcode_list = []
        # left_object_list = []
        center_object_list = []
        pair_sort = []

        for i, c in (enumerate(out_classes)):

            predicted_class = self.class_names[c]

            box = out_boxes[i]
            # score = out_scores[i]

            # label = '{} {:.2f}'.format(predicted_class, score)
            label = '{}'.format(predicted_class)
            # scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            top = int(top * Ry)
            left = int(left * Rx)
            bottom = int(bottom * Ry)
            right = int(right * Rx)

            # mid_h = (bottom - top) / 2 + top
            mid_v = (right - left) / 2 + left
            try:
                if center_axes[0] - self.center_axes > 8:
                    detect = 1
                    self.count_update_notmatch = 0
                    self.barcode_list = []
                    self.match_text_list = []
                    self.wait_moving = 0
            except:
                pass


            if mid_v < 55 and c == 4 and (time.time() - self.start_tracking) > 5 :
                self.num_obj += 1
                self.start_tracking = time.time()
                # detect = 1
                # self.count_update_notmatch = 0
                # self.barcode_list = []
                # self.match_text_list = []
                # self.wait_moving = 0
                print(mid_v)


            if (c != 15) and (c != 4):
                # cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)
                # (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                #                                                       thickness / self.text_size, 1)
                #
                # # put text rectangle
                # cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c],
                #               thickness=cv2.FILLED)
                #
                # # put text above rectangle

                cv2.putText(image, label, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX, thickness / self.text_size,
                            (0, 0, 0),1)

                # print(distance)
                distance = np.absolute(mid_v - center_axes)
                # distance = np.absolute(left - left_axes)
                # print('left',left)
                # print('distance',distance)
                distance_index = np.where(distance < 105)

                distance_index = list(distance_index[0])
                list_text_sort = list(np.array(list_text)[sorted(range(len(center_list)),key=center_list.__getitem__)])
                # list_text_sort = list(np.array(list_text)[sorted(range(len(left_axes)),key=left_axes.__getitem__,reverse = True)])
                # list_text_sort = list(np.array(list_text)[sorted(range(len(left_list)),key=left_list.__getitem__)])


                # print(list_text_sort)

                try:

                    remove_list.append(distance_index[0])

                    center_object_list.append(mid_v)
                    # left_object_list.append(left)
                    pair.append([distance_index[0], label, list_text_sort[distance_index[0]]])

                except:
                    # print('Something Wrong')
                    pass
            elif c == 4:
                # having_text = 1
                # image = scipy.ndimage.zoom(image,(2,2,1),order = 1)## upsampling image

                try:
                    img = image[top:bottom + 15, left - 20:right + 20]
                    text = read_plate.detect(img)

                    n = len(text) - 3
                    text = text[:n] + '000'
                    list_text.append(text[:5])
                    center_list.append(mid_v)
                    # left_list.append(left)
                    cv2.putText(image, text[:5], (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                except:
                    # print('wrong in text')
                    pass

            elif c == 15:
                try:
                    img = image[top:bottom + 15, left - 20:right + 20]
                    text = barcode_detect(img).decode("utf-8")
                    # cv2.putText(image, text[:6], (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    # print(text)
                    barcode_list.append(text)
                    center_list_barcode.append(mid_v)
                    # left_list_barcode.append(left)

                except:
                    center_list_barcode.append(mid_v)
                    # left_list_barcode.append(left)
                    # print('sth wrong')
                    barcode_list.append('error')


                #### hien thi text gia tren object

                # print(text[:6])

        not_match_list = []

        no_obj = [i for j, i in enumerate(no_obj) if j not in remove_list]

        obj_text = make_no_obj_text(no_obj)

        # print(left_object_list)
        barcode_list_sort = list(np.array(barcode_list)[sorted(range(len(center_list_barcode)), key=center_list_barcode.__getitem__)])
        # barcode_list_sort = list(np.array(barcode_list)[sorted(range(len(left_list_barcode)), key=left_list_barcode.__getitem__)])
        if len(pair) == 0:
            pass
        else:
            try:
                pair_sort = list(np.array(pair)[sorted(range(len(center_object_list)), key=center_object_list.__getitem__)])
                # pair_sort = list(np.array(pair)[sorted(range(len(left_object_list)), key=left_object_list.__getitem__)])

            except:
                pair_sort = []
        barcode_string = make_no_obj_text(barcode_list_sort)

        if len(no_obj) == 0:
            obj_text = '...'

        for idx, object, text in pair_sort:
            if check([object, text]):
                continue
            else:
                not_match_list.append(str(int(idx) + self.num_obj))
        if len(not_match_list) == 0:
            match_text = '...'
        else:
            match_text = make_no_obj_text(not_match_list)

        try:
            self.center_axes = center_axes[0]
        except:
            pass
            # add everything to list
            # ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])

        return image, detect, obj_text, match_text, no_obj, not_match_list, barcode_string

    def close_session(self):
        self.sess.close()

    def detect_img(self, image):
        # image = cv2.imread(image, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image, detect, obj_text, match_text, no_obj, not_match_list, barcode_string = self.detect_image(original_image)
        # r_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, detect, obj_text, match_text, no_obj, not_match_list, barcode_string


sg.ChangeLookAndFeel('LightGreen')
menu_def = [
    ['File', ['Open', ['Hello']]],
    ['Help']
]
sg.SetOptions(text_color="#e4e4e4")

layout = [
    [sg.Menu(menu_def)],
    [sg.Text("DEEP LEARNING BASED GOODS MANAGEMENT", size=(30, 2), justification='center',
             font=("Any", 20), background_color="#343434")],
    [sg.Text("_" * 70, background_color="#343434")],
    [sg.Text(' ' * 36, background_color="#343434"),
     sg.Button("DETECT_PRODUCT", size=(20, 3), button_color=('#343434', 'white'), font=("Helvetica", 12))],
    [sg.Text('', background_color="#343434")],
    [sg.Text(' ' * 42, background_color="#343434"),
     sg.Button("DEMO",size=(15, 3), button_color=('#343434', 'white'), font=("Helvetica", 12))],
    [sg.Text('', background_color="#343434")],
    [sg.Text(' ' * 42, background_color="#343434"),
     sg.Button("RESET",size=(15, 3), button_color=('#343434', 'white'), font=("Helvetica", 12))],
    [sg.Text('', background_color="#343434")],
    [sg.Text(' ' * 42, background_color="#343434"),
     sg.Exit(size=(15, 3), button_color=('#343434', 'white'), font=("Helvetica", 12))]
]
win1 = sg.Window('GOODS MANAGEMENT', background_color="#343434").Layout(layout)
win2_active = False
start_time = time.time()
FPS = 0
count_detect = 0
count_save = 0

# match_text_list = []
match_text = '...'
no_object_list = []
not_match_list = []
# barcode_list = []
start = 0
count_save = 0
text = '...'
no_object_text = ''

_barcode = ''
_no_object_barcode_save = []
_not_match_barcode_save = []

while True:
    ev1, vals1 = win1.Read()

    if vals1 is None or ev1 == 'Exit':
        break
    if ev1 == 'DEMO':
        # DEMO_text.demo()

        yolo = YOLO()
        # winName = 'Detection'
        # cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(winName, 1000, 1000)


        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("We cannot open webcam")
        win_started = False
        while True:
            yolo.wait_moving +=1
            if yolo.wait_moving > 100000:
                yolo.wait_moving = 20
            ret, frame = cap.read()

            # resize our captured frame if we need

            # detect object on our frame
            r_image, detect, obj_text, check_match, no_obj_list, no_match_list, barcode_string = yolo.detect_img(frame)

            # show us frame with detection
            if yolo.wait_moving > 5:
                yolo.match_text_list.append(check_match)
                yolo.barcode_list.append(barcode_string)


            if detect == 0  and yolo.wait_moving > 5:
                count_detect = count_detect + 1
                if count_detect > 20 :
                    # cv.putText(frame, 'No Object', (0, 50), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
                    text = 'No Object'
                    # print(no_object_list)
                    no_object_text = obj_text
                    no_object_list.extend(no_obj_list)

                    no_object_list = list(set(no_object_list))
                    no_object_list_int = unmake(no_object_text)
                    count_detect = 0


            else:
                count_detect = 0
                text = "Having Object"
                no_object_text = '...'
                no_object_list_int = []

            if yolo.count_update_notmatch > 20:
                print('===============')
                match_text = check_most_freq(yolo.match_text_list)
                yolo.match_text_list = []
                barcode_string_infor = check_most_freq(yolo.barcode_list)
                yolo.barcode_list = []
                # print(barcode_string_infor)
                try:
                    barcode_elem = unmake_string(barcode_string_infor)
                    # print(barcode_elem)
                except:
                    # print('wrong')
                    pass

                try:

                    _barcode = convert(barcode_elem)
                    # print(_barcode)

                except:
                    # print('wrong at convert barcode')
                    _barcode = ['...'] *(len(no_match_list)+len(no_obj_list))


                not_match_elem = unmake(match_text)


                if not_match_elem == None:
                    pass
                else:
                    not_match_list.extend(not_match_elem)
                    not_match_list = list(set(not_match_list))

                try:

                    _barcode_not_match = list(np.array(_barcode)[np.array(not_match_elem)-yolo.num_obj])
                    # print(_barcode_not_match)
                    _not_match_barcode_save = _not_match_barcode_save+_barcode_not_match

                    # print(_not_match_barcode_save)
                    _not_match_barcode_save = sorted(set(_not_match_barcode_save), key=_not_match_barcode_save.index)

                except:
                    # print('not enough')
                    _barcode_not_match = []
                    _not_match_barcode_save.extend(_barcode_not_match)

                try:
                    _barcode_no_object = list(np.array(_barcode)[np.array(no_object_list_int) - yolo.num_obj])
                    # print(_barcode_no_object)
                    _no_object_barcode_save = _no_object_barcode_save+_barcode_no_object
                    # print(_no_object_barcode_save)
                    _no_object_barcode_save = sorted(set(_no_object_barcode_save), key=_no_object_barcode_save.index)
                except:
                    # print('no_obj_ntengou')
                    _barcode_no_object = []
                    _no_object_barcode_save = _no_object_barcode_save + _barcode_no_object

                yolo.count_update_notmatch = 0

            elif yolo.wait_moving > 5:
                yolo.count_update_notmatch += 1

                # show the image

            r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)

            # r_image = cv2.line(r_image, (20,0), (20,416), (255,0,255), 3)

            # r_image = cv2.resize(r_image,(416,416))
            imgbytes = cv2.imencode('.png', r_image)[1].tobytes()

            if not win_started:
                win_started = True
                layout = [
                    [sg.Image(data=imgbytes, key='_IMAGE_')],
                    # [sg.Text(text=text, size=(30, 1), font=('Any', 18), text_color='#1c4a34', key='_TEXT_'),
                     # sg.Text(text=no_object_text, size=(30, 1), font=('Any', 18), text_color='#1c4a34',
                     #         key='_OBJTEXT_')],
                     [sg.Text(text=text, key='_TEXT_',font=('Any', 18),size = (20,1),text_color='#1c4a34'),
                      sg.Text(text=no_object_text, font=('Any', 18),size = (20,1),text_color='#1c4a34',
                              key='_OBJTEXT_')],
                    # [sg.Text(text='Not Match', size=(30, 1), font=('Any', 18), text_color='#1c4a34'),
                    #  sg.Text(text=match_text, size=(30, 1), font=('Any', 18), text_color='#1c4a34',
                    #          key='_MATCHTEXT_')],
                    [sg.Text(text='Not Match' ,font=('Any', 18),size = (20,1),text_color='#1c4a34'),
                     sg.Text(text=match_text, font=('Any', 18),text_color='#1c4a34',size = (20,1),
                             key='_MATCHTEXT_')],
                    [sg.Exit(), sg.Button("Save")]
                ]
                win = sg.Window('Detection',
                                text_justification='left',
                                default_element_size=(14, 1),
                                auto_size_text=False).Layout(layout).Finalize()
                # win.maximize()

                image_elem = win.FindElement('_IMAGE_')
                text_elem = win.FindElement('_TEXT_')
                text_elem1 = win.FindElement('_OBJTEXT_')
                text_elem2 = win.FindElement('_MATCHTEXT_')
            else:
                image_elem.Update(data=imgbytes)
                text_elem.Update(value=text)
                text_elem1.Update(value=no_object_text)
                text_elem2.Update(value=match_text)
            FPS += 1
            TIME = time.time() - start
            if TIME > 2:
                print('FPS', FPS / TIME)
                FPS = 0
                start = time.time()
            event, values = win.Read(timeout=0)
            if event == 'Save':
                try:
                    no_object_list.sort()
                    not_match_list.sort()
                    count_save += 1

                    save(no_object_list, not_match_list,_no_object_barcode_save,_not_match_barcode_save,count_save)
                except:
                    print('Save wrong')
            elif event is None or event == 'Exit':
                break

        win.Close()
        cap.release()


    elif ev1 == 'DETECT_PRODUCT':
        layout = [
            [sg.Text(' ' * 5, background_color="#343434"),
             sg.Button("WEBCAM", size=(15, 3), button_color=('#343434', 'white'), font=("Helvetica", 12)),
             sg.Text(' ' * 5, background_color="#343434")],
            [sg.Text('', background_color="#343434")],
            [sg.Text(' ' * 5, background_color="#343434"),
             sg.Button("IP CAMERA", size=(15, 3), button_color=('#343434', 'white'), font=("Helvetica", 12)),
             sg.Text(' ' * 5, background_color="#343434")],
            [sg.Text('', background_color="#343434")],
            [sg.Text(' ' * 5, background_color="#343434"),
             sg.Exit(size=(15, 3), button_color=('#343434', 'white'), font=("Helvetica", 12)),
            sg.Text(' ' * 5, background_color="#343434")]
        ]
        win2 = sg.Window('Detection',
                        background_color="#343434",
                        text_justification='left',
                        default_element_size=(14, 1),
                        auto_size_text=False).Layout(layout)

        while True:
            ev2, vals2 = win2.Read()
            if vals2 is 'WEBCAM' or ev2 == 'WEBCAM':
                layout = [
                    [sg.Text(' ' * 5, background_color="#343434"),
                     sg.Button("TEXT", size=(15, 3), button_color=('#343434', 'white'), font=("Helvetica", 12)),
                     sg.Text(' ' * 5, background_color="#343434")],
                    [sg.Text('', background_color="#343434")],
                    [sg.Text(' ' * 5, background_color="#343434"),
                     sg.Button("BARCODE", size=(15, 3), button_color=('#343434', 'white'), font=("Helvetica", 12)),
                     sg.Text(' ' * 5, background_color="#343434")],
                    [sg.Text('', background_color="#343434")],
                    [sg.Text(' ' * 5, background_color="#343434"),
                     sg.Exit(size=(15, 3), button_color=('#343434', 'white'), font=("Helvetica", 12)),
                     sg.Text(' ' * 5, background_color="#343434")]
                ]

                win3 = sg.Window('Detection',
                                background_color="#343434",
                                text_justification='left',
                                default_element_size=(14, 1),
                                auto_size_text=False).Layout(layout)
                while True:
                    ev3, vals3 = win3.Read()
                    if vals3 is None or ev3 == 'TEXT':
                        detect_text.detect()
                    elif vals3 is None or ev3 == 'BARCODE':
                        detect_barcode.detect()
                    elif vals3 is None or ev3 == 'Exit':
                        break
                win3.Close()

            if vals2 is None or ev2 == 'Exit':
                break

        win2.Close()
    elif ev1 == 'RESET':
        win2_active = False
        start_time = time.time()
        FPS = 0
        count_detect = 0
        count_save = 0

        # match_text_list = []
        match_text = '...'
        no_object_list = []
        not_match_list = []
        # barcode_list = []
        start = 0
        count_save = 0
        text = '...'
        no_object_text = ''

        _barcode = ''
        _no_object_barcode_save = []
        _not_match_barcode_save = []
        _no_object_barcode_save = []
        not_match_barcode_save = []
        count_save = 0

