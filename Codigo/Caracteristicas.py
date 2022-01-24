#Caracteristicas importantes de las imagenes

#Librerias
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import cv2 as cv
from skimage import exposure


#Lectura de la imagen
im = pydicom.dcmread('P_00005\Imagen1.dcm')
imO = im.pixel_array



#Mostrar imagen original MATPLOTLIB
# plt.imshow(imO, cmap = 'gray') 
# plt.axis('off')
# plt.show()

#Mostrar imagen original OPENCV
# cv.imshow('Imagen Original',imO)
# cv.waitKey()


# #Cálculo del área y perímetro

# #Binarización de la imagen
_,th = cv.threshold(imO,100,255,cv.THRESH_BINARY_INV,)
cv.imshow('Imagen Original',imO)
cv.waitKey()
# np.array(th)
# plt.imshow(th, cmap = 'gray')
# plt.axis('off')
# plt.show()

# cv.convertScaleAbs(th)

# #Perímetro
# imagen, contornos, jerarquia = cv.findContours(th,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(imO,contornos,-1,(255,0,0),3)



