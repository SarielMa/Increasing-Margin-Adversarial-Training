# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 16:11:54 2021

@author: linhai
"""

import sys
import inspect
import os
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p, subfiles, subdirs, isfile

#print (sys.path)
curDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parDir = os.path.dirname(curDir)
sys.path.insert(0, parDir)
#sys.path.insert(0, )
p1 = 'C:\\Research\\IMA_on_segmentation\\nnUnet\\nnUNet\\rawData\\nnUNet_raw_data\\Task05_Prostate\\imagesTr'
p2 = 'C:/Research/IMA_on_segmentation/nnUnet/nnUNet/rawData/nnUNet_raw_data\\Task05_Prostate'
p3 = 'C:\\Research\\IMA_on_segmentation\\aaa'
p4 = 'C:/Research/IMA_on_segmentation/333/aaab'
print (os.path.join(p1, "aaa")+"\\")
print (isdir(join(p1, "aaa")+"\\"))
print(p1)
print (isdir(p2))
#os.mkdir(p4)
maybe_mkdir_p(p4)
#os.makedirs(p4, exist_ok=True)