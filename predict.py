# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 01:18:59 2022

@author: mislam8
https://github.com/msminhas93/DeepLabv3FineTuning/blob/master/Analysis.ipynb
"""
from timeit import default_timer as timer

import copy
import csv
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import cv2

im='v9'

img = cv2.imread('./man/256x256/' +im + '.JPG').transpose(2,0,1).reshape(1,3,256,256) # 320,480
#img=np.float16(img)
model = torch.load('./exp_dir/mnv256cuda.pt')
start=timer()
with torch.no_grad():
    a = model(torch.from_numpy(img)/255)
    #a = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)



vtime = timer() - start
print(vtime)


plt.subplot(311)
plt.imshow(cv2.imread('./val/val_images/' +im + '.JPG'))
plt.title('Groundtruth:')
plt.axis('off')

plt.subplot(312)
plt.imshow(a['out'].cpu().detach().numpy()[0][0]>0.2);
plt.axis('off')
plt.savefig('./val/p2.png', bbox_inches='tight',pad_inches = 0)

### Saving without border
#plt.savefig('p4.png', bbox_inches='tight',pad_inches = 0)
plt.title('Prediction')# Yellow=weed Purple=BG

### plt.savefig('./CFExp/SegmentationOutput.png',bbox_inches='tight')

plt.subplot(313)
plt.imshow(cv2.imread('./val/val_masks/' +im + '.png'))
plt.title('Groundtruth mask:')
plt.axis('off')


""" check model parameters
for param in model.parameters():
    print(param)
    
"""

