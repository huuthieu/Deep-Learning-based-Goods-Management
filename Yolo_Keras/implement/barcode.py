import cv2 as cv
import numpy as np
import time

import urllib.request as urllib
import scipy.ndimage
# from IP_camera import get_frame_from_IP_camera
from Tesseract_OCR import Image_to_string
from barcode_detect import barcode_detect

# url = 'http://192.168.1.5:8080/shot.jpg'

# Write down conf, nms thresholds,inp width/height



def detect():
    confThreshold = 0.25
    nmsThreshold = 0.40
    inpWidth = 416
    inpHeight = 416

    # Load names of classes and turn that into a list
    classesFile = "obj.names"
    classes = None

    # count_detect to delay


    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

        # Model configuration
    modelConf = 'yolov3-tiny-obj.cfg'
    modelWeights = 'yolov3-tiny-obj_50000.weights'

    def wait():
        time.sleep(5)

    def postprocess(frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIDs = []
        confidences = []
        boxes = []
        having_text = 0

        for out in outs:
            for detection in out:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > confThreshold:
                    centerX = int(detection[0] * frameWidth)
                    centerY = int(detection[1] * frameHeight)

                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)

                    left = int(centerX - width / 2)
                    top = int(centerY - height / 2)

                    classIDs.append(classID)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

        for i in indices:

            i = i[0]  # lay so thu tu cho vao box chu khong phai lay 1 list
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            image = frame[top:top + height + 15, left - 20:left + width + 20]  # crop image to recognize

            # ve bounding boxes
            if (classIDs[i] != 15) and (classIDs[i] != 4):
                drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)
            elif (classIDs[i] == 15):
                try:
                    val = barcode_detect(image)
                except:
                    print('erro in get image')
                # if val != None:
                # cv.imwrite('barc.jpg', image)
                drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)
                print(val)

            # if classIDs[i] != 4 and classIDs[i] != 15:
            #     drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)
            # elif classIDs[i] == 4:
            #     having_text = 1
            #     # image = scipy.ndimage.zoom(image,(2,2,1),order = 1)## upsampling image
            #     text = Image_to_string(image)
            #     #### hien thi text gia tren object
            #     cv.putText(frame, text[:6], (left, top), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            #     print(text[:6])

        # if len(indices) and having_text != 1:
        #     detect = 1
        #
        # else:
        #     detect = 0
        # return detect

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        # label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if classes:
            assert (classId < len(classes))
            # label = '%s:%s' % (classes[classId], label)
            label = '%s' % (classes[classId])

        # A fancier display of the label from learnopencv.com
        # Display the label at the top of the bounding box
        # labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
        # (255, 255, 255), cv.FILLED)
        # cv.rectangle(frame, (left,top),(right,bottom), (255,255,255), 1 )
        # cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()

        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Set up the net

    net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # Process inputs
    winName = 'Detection'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    cv.resizeWindow(winName, 1000, 1000)

    cap = cv.VideoCapture(0)
    FPS = 0
    display_time = 2
    start = time.time()
    while cv.waitKey(1) < 0:

    ###get frame from video

        hasFrame, frame = cap.read()

    ### get frame from IP_webcam

    # imgResp = urllib.urlopen(url)
    # imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    # img = cv.imdecode(imgNp, -1)
    # frame = img

    ##### call from IP_camera.py function

    # frame = get_frame_from_IP_camera()

    # Create a 4D blob from a frame

        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Set the input the the net
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))

    # postprocess(frame,outs)
        postprocess(frame, outs)  # draw and set flag detect when no object
        # if detect == 0:
        #     count_detect = count_detect + 1
        #     if count_detect > 50:
        #         cv.putText(frame, 'No Object', (0, 50), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        # else:
        #     count_detect = 0

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    # show the image
        cv.imshow(winName, frame)
        # break
    ## calculate FPS

        FPS += 1
        TIME = time.time() - start
        if TIME > display_time:
            print('FPS', FPS / TIME)
            FPS = 0
            start = time.time()

    cap.release()
    cv.destroyAllWindows()














