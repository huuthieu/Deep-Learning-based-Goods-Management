webcam.py: run file in idle (raw code)
GUI.py: a GUI for whole code (detect product, price tag and barcode), including some modules:
a. Tesseract_OCR.py: OCR with Tesseract API
b. IP_camera (if necessary): Get frame from IP Camera
c. detect_product.py: run raw predict before process (output is the bounding box of objects).
d. detect_barcode.py: detect product and barcode.
e. detect_text.py: detect product and price tag.
f. OCR_CNN: OCR with CNN model buit in SVHN dataset (with 2 weights model.h5 and model2.h5).
g. export_excel.py: save some necessary information to excel file.

model_data: include some config file of model.
yolo3: include all class (process, loss, backbone) of yolov3 model.