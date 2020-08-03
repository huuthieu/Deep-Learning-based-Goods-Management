import os
import cv2
import numpy as np

dir = 'bbox_txt/'

img_dir = 'images/'

imgList = [os.path.join(dir,path) for path in os.listdir(dir)]
imgImage = [os.path.join(img_dir,path) for path in os.listdir(img_dir)]

k = 0
with open ('data_keras/train.txt','w') as f1:
    for path in imgList:
        with open(path) as f:
            txts = f.read().splitlines()
            text = 'data/'+ imgImage[k]
            for txt in txts:
                size = txt.split()[1:]
            
                img = cv2.imread(imgImage[k])
                h,w = img.shape[:2]
                center_x = (w*(float(size[0])))
    
                center_y = (h*(float(size[1])))
                width = (w*(float(size[2])))
                height = (h*(float(size[3])))
                x_min = int(center_x -width/2)
                x_max = int(center_x + width/2)
                y_min = int(center_y -height/2)
                y_max = int(center_y + height/2)
                text = text + ' ' + str(x_min)+','+str(y_min)+','+ str(x_max)+','+str(y_max)+','+ txt.split()[0]
            k = k+1
        text = text + '\n'
        f1.write(text)
##

####path = img_dir +'/'+ str(4).zfill(6) +'.jpg'
##print(imgImage[k])
##img = cv2.imread(imgImage[k])
##cv2.imshow('',img)
##print(img.shape[:2])
