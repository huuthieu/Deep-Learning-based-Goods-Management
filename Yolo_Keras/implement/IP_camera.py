import urllib.request as urllib
import cv2
# import vlc
import numpy as np
import time

url = 'http://192.168.1.5:8080//shot.jpg'
# winName = 'Detection'
# # cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
# # cv2.resizeWindow(winName, 1000,1000)
# # while True:
def get_frame_from_IP_camera():
    imgResp = urllib.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()),dtype = np.uint8)
    img = cv2.imdecode(imgNp,-1)
    return img
# if ord('q') == cv2.waitKey(1):
# exit(0)

# import os
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"
#
# cap = cv2.VideoCapture("rtsp://admin:thieu12345@192.168.1.8:554/onvif1",apiPreference=cv2.CAP_FFMPEG)
# print(cap.isOpened())
# while(True):
#     ret, frame = cap.read()
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# print(cv2.getBuildInformation())
#
# player=vlc.MediaPlayer('rtsp://admin:thieu12345@192.168.1.8:554/onvif1')
# player.play()
#
# while 1:
#     time.sleep(1)
#     player.video_take_snapshot(0, '.snapshot.tmp.png', 0, 0)
# #     cap = cv2.imread('.snapshot.tmp.png')
#     cv2.imshow('frame',cap)
#     if cv2.waitKey() == ord('q'):
#         break
