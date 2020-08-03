import os
dir = 'test_image/text_barcode'
num = [img for img in os.listdir(dir)]
print(sum(num))
