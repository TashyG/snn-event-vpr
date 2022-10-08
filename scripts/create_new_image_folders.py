from tkinter import image_names
import cv2
import numpy as np
import os
import tqdm
import sys
from os import path

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from src.constants import path_to_image_files

def processImageDataset(inpath, outpath):
    print("Extracting images for image path: {} ...".format(inpath))
    imgList = np.sort(os.listdir(inpath))
    imgList = [os.path.join(inpath,f) for f in imgList]
    for i, imPath in enumerate(imgList):
        if i % 4 == 0:
            image_name = imPath[-27:]
            im = cv2.imread(imPath)
            out_name = outpath + image_name
            cv2.imwrite(out_name, im)


inpath = path_to_image_files + 'bags_2022-03-28-12-01-42/frames'
outpath = path_to_image_files + 'bags_2022-03-28-12-01-42_less_images/'

processImageDataset(inpath, outpath)