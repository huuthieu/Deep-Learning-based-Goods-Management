from PIL import Image
import pytesseract
import cv2
import os

tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
os.environ['TESSDATA_PREFIX'] = 'C:/Program Files (x86)/Tesseract-OCR'
os.environ['OMP_THREAD_LIMIT']= '1'
def Image_to_string(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    # gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # gray = cv2.erode(gray, kernel, iterations=1)

    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    # text = pytesseract.image_to_string(Image.open(filename), config = tessdata_dir_config)
    text = pytesseract.image_to_string(Image.open(filename),config = '--psm7 -c tessedit_char_whitelist=0123456789 outputbase digits tessedit_do_invert=0')
    os.remove(filename)
    return text
