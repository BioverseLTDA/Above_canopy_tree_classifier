# Script for running yolo

# set working directory
#import os
#os.chdir('/Users/zach/Dropbox (ZachTeam)/Projects/darkflow-master_1')

# import libraries
from darkflow.net.build import TFNet
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# % config InlineBackend.figure_format = 'svg'

# set model options
options = {
        'model': 'cfg/tiny-yolo-voc-1class_toh_above_canopy.cfg',
        'load': 2625,
        'threshold': 0.3,
        'gpu': 1.0
}
# create tfnet object
tfnet = TFNet(options)

colors = (0,255,0)

# create image object
global img
img = None

img = cv2.imread('sample_img/000054.jpg', cv2.IMREAD_COLOR)

img
img.shape

frame = img
results = tfnet.return_predict(frame)
    

for color, result in zip(colors, results):
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label']
        confidence = result['confidence']
        text = '{}: {:.0f}%'.format(label, confidence*100)
        frame = cv2.rectangle(frame, tl, br, (50,0,255), 8)
        #frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 2, (50,0,255), 3) #class label + confidence
        frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 2, (50,0,255), 3) #class label only
# display image with bounding box
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.show()

# save image
mpimg.imsave("new_img_out.png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))