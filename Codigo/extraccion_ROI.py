#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 21:37:05 2021

@author: pablo
"""

import cv2
import numpy as np
#from matplotlib import pyplot, cm
#import imutils 
#import os
import pydicom as dicom 
from skimage.feature import greycomatrix, greycoprops
#from skimage.measure import shannon_entropy
#import pandas as pd

#debe estar listo para aplicar el for a todo el path 

 #image_path
#input_path_mask = "/home/pablo/Escritorio/calcification"
#files_names = os.listdir(input_path_mask)
#  for file_name in files_names:
#      print(filename)

BI=2
img_path='C:/Users/DSPLab/Documents/Pablo/CAL_BIRADS_2'
imageDicom = dicom.dcmread('C:/Users/DSPLab/Documents/Pablo/CAL_BIRADS_2/BIRADS_2_ROI_DICOM/20587148_fd746d25eb40b3dc_MG_R_CC_ANON.dcm')
imgD = cv2.imread('C:/Users/DSPLab/Documents/Pablo/CAL_BIRADS_2/BIRADS_2_ROI_CALC/20587148_mask.png',0)
ds = imageDicom.pixel_array

#resize images
#image = imutils.resize(imgD, width=720)
#imagedc = imutils.resize(ds, width=720)
lstFilesDCM = [1,2,3,4]
# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(imageDicom.Rows), int(imageDicom.Columns), len(lstFilesDCM))

# Load spacing values (in mm)
#ConstPixelSpacing = (float(imageDicom.PixelSpacing[0]), float(imageDicom.PixelSpacing[1]), float(imageDicom.SliceThickness))
ConstPixelSpacing = [1.0, 1.0, 1.0]
x1 = np.arange(0.0, (ConstPixelDims[0])*ConstPixelSpacing[0], ConstPixelSpacing[0])
y1 = np.arange(0.0, (ConstPixelDims[1])*ConstPixelSpacing[1], ConstPixelSpacing[1])
z1 = np.arange(0.0, (ConstPixelDims[2])*ConstPixelSpacing[2], ConstPixelSpacing[2])

ds = np.zeros(ConstPixelDims, dtype=imageDicom.pixel_array.dtype)


#ds[:, :, lstFilesDCM.index(filenameDCM)] = imageDicom.pixel_array
ds[:, :, 1] = imageDicom.pixel_array

#

contours,hierarchy = cv2.findContours(imgD, 1, 2)

cnt = contours[0]
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
delt = 10
#Crop selected roi from raw image
#roi_cropped = imagedc[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
roi_cropped = ds[center[1]-delt:center[1]+delt , center[0]-delt:center[0]+delt, 1]


#show cropped image

numpy_data = np.asarray(roi_cropped)
numpy_data2 = numpy_data.astype(int)

"""
def ENERGY(inicio, final):
    control = 0
    ar = numpy_data2
    for x in range(inicio, final + 1):
        control  += (ar)**2
    return control

def CONTRASTE(n1, f1):
    ctr = 0
    art = numpy_data2
    ar_i = numpy_data2[0]
    ar_j = numpy_data2[1]
    for x in range(n1, f1 + 1):
        ctr += ((ar_i - ar_j)**2)*(art)
        return ctr

def ENTROPIA(n2, f2):
    ctr = 0
    art = numpy_data2
    for x in range(n2, f2 + 1):
        ctr += (art) * log2*(art)
        return ctr

def HOMOGENEIDAD(n3, f3):
    ctr = 0
    art = numpy_data2
    ar_i = numpy_data2[0]
    ar_j = numpy_data2[1]
    for x in range(n3, f3 + 1):
        ctr += art / (1 + (ar_i - ar_j)**2 )   
        return ctr    

"""
    

print(numpy_data2)

biggest = np.amax(numpy_data2)

#g=greycomatrix(numpy_data2, [1, 5], [0, np.pi/2], levels=3140,
 #                 normed=True, symmetric=True)


g=greycomatrix(numpy_data2, [1, 5], [0, np.pi/4, np.pi/2], levels=biggest+2,
                   symmetric=True, normed=True)


contrast = greycoprops(g, 'contrast')
homo = greycoprops(g, 'homogeneity')
energy = greycoprops(g, 'energy')
correlation = greycoprops(g, 'correlation')
ASM = greycoprops(g, 'ASM')
#entropy = shannon_entropy(g, base=2)

cont = np.zeros((1,6))
cont = contrast
hom = np.zeros((1,6))
hom = homo
ener = np.zeros((1,6))
ener=energy
corr = np.zeros((1,6))
corr=correlation
ASm = np.zeros((1,6))
ASm=ASM
#vector=np.concatenate([cont.reshape(-1), hom.reshape(-1), ener.reshape(-1) , corr.reshape(-1), ASm.reshape(-1), [entropy]])
vector=np.concatenate([cont.reshape(-1), hom.reshape(-1), ener.reshape(-1) , corr.reshape(-1), ASm.reshape(-1),])

try:
    pass
except Exception:
    pass
#BI=2
archivo = open("caract.txt", "w")
for i in range(0,len(vector),1):
    archivo.write('%f\t'%vector[i])
archivo.write('%d'%BI)    
archivo.write('\n')
archivo.close()
#path = 'C:/Users/DSPLab/Documents/Pablo/CAL_BIRADS_2/MASK_BIRADS_2'
#cv2.imwrite(os.path.join(path, '50997515_ROI_BIRADS_2.png')  , roi_cropped)
#cv2.imshow("ROI", roi_cropped)
#hold window
#cv2.waitKey(0)
#cv2.imshow('imagen', roi_cropped)
#cv2.destroyAllWindows()