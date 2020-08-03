import numpy as np
import cv2
import pyzbar.pyzbar as pyzbar
from PIL import Image
from IP_camera import get_frame_from_IP_camera
def barcode_detect(img):
# img = cv2.imread('zbar2.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                          cv2.THRESH_BINARY,11,2)

    decodeObjects = pyzbar.decode(gray)

    # cv2.imshow('zbar',img)
    try:
        obj = decodeObjects[0]
        value = obj.data
    except:
        print('sth wrong')
        return None
    return value
    # cv2.waitKey(0)

