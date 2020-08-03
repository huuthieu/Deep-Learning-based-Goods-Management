import pytesseract
import cv2
import numpy as np
# from lib_detection import load_model, detect_lp, im2single
from keras.models import load_model
import cv2 as cv
import numpy as np
import time
# model --> (30,60)
# model 2,3,4 --> (30,45)

model = load_model('model2.h5')
model_svm = cv2.ml.SVM_load('svm.xml')

def detect(img):
# Ham sap xep contour tu trai sang phai
    def sort_contours(cnts):

        reverse = False
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
        return cnts


    char_list =  '0123456789'



    def fine_tune(lp):
        newString = ""
        for i in range(len(lp)):
            if lp[i] in char_list:
                newString += lp[i]
        return newString


# Cau hinh tham so cho model 
    digit_w = 30 # Kich thuoc ki tu
    digit_h = 45 # Kich thuoc ki tu

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
	    angle = -(90 + angle)

    else:
	    angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)




    img = cv2.resize(rotated, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # img = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = img.copy()

    # Ap dung threshold de phan tach so va nen
    binary = cv2.threshold(gray, 127, 255,
                         cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]



    # Segment kí tự
    #     kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #     thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    cont, _  = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    plate_info = ""
    # print(len(cont))

    for i,c in enumerate(sort_contours(cont)):
        if i == len(cont)-7:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if ratio<=3: # Chon cac contour dam bao ve ratio w/h
            if h/img.shape[0]>=0.3: # Chuan la 0.25

                # Ve khung chu nhat quanh so
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Tach so va predict
                curr_num = binary[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY_INV)


                #
                # curr_num = np.expand_dims(curr_num, 0)
                # curr_num = np.expand_dims(curr_num, -1)
                curr_num = curr_num.reshape((1, 45, 30 , 1))
                curr_num = curr_num / 255
                pred = model.predict(curr_num)
                result = np.argmax(pred)
                result = str(result)
                # curr_num = np.array(curr_num,dtype=np.float32)
                # curr_num = curr_num.reshape(-1, digit_w * digit_h)
                #
                # # Dua vao model SVM
                # result = model_svm.predict(curr_num)[1]
                # result = int(result[0, 0])
                #
                # if result<9: # Neu la so thi hien thi luon
                #     result = str(result)
                # else: #Neu la chu thi chuyen bang ASCII
                #     result = chr(result)

                plate_info +=result
                if len(plate_info) == 5:
                    break

    # cv2.imshow("Cac contour tim duoc", roi)
    # cv2.waitKey()

    
    # cv2.putText(img,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)


    return plate_info



# FPS = 0
# display_time = 2
# start = time.time()
#
#
#
#     # show the image
#
#     ## calculate FPS















# cv2.destroyAllWindows()
