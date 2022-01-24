"""
Lectura de 

"""

#Librerias Utilizadas
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2
import os

#Entropy
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk


#Lectura de archivo
w_d  = 'CBIS-DDSM/Calc-Test' #ruta
full = w_d + 'P_00005.dcm'
roi  = w_d +  'P_00005-ROI.dcm'
#w_i = w_d + 'ejem.dcm' 

#Lectura de las Imagenes
ImDCMFull = pydicom.dcmread(full)
ImDCMROI = pydicom.dcmread(roi)

#Conversión a arreglos de Numpy
ImFull = np.asarray(ImDCMFull.pixel_array)
ImROI = np.asarray(ImDCMROI.pixel_array)

#Muestra de Imagen Original
plt.figure('Mamografía Original')
plt.imshow(ImFull, cmap = 'gray')
plt.axis('off')

#Muestra de la máscara de la región de Interes
plt.figure('Máscara del Área de Interés')
plt.imshow(ImROI, cmap = 'gray')
plt.axis('off')

#Creación de Imagen que únicamente contiene el área de Interés con la unidad
u, v= np.shape(ImROI)
IdentidadROI = np.zeros((u,v)) #Mascara de zeros del tamaño de ROI

#Cambiando 255 por 1
for i in range(u):
    for j in range(v):
        if(ImROI[i,j]) > 0:
            IdentidadROI[i,j] = 1 #Cambia los valores de la mascara en zeros con valores en 1

#Multiplicando elemento con elemento
ImagenInteres = np.multiply(ImFull,IdentidadROI)




#Muestra de la región de interés de la mamografía
plt.figure('Region de Interes de la Mamografía')
plt.imshow(ImagenInteres, cmap = 'gray')
plt.axis('off')
plt.show()


#Obtención de Características Relevantes
_, ImaBin = cv2.threshold(ImagenInteres,0,255,cv2.THRESH_BINARY) #Binarización de la Imagen
ImaBin = ImaBin.astype(np.uint8) #Conversión a Tipo de Dato UINT8 NECESARIA PARA TRABAJAR CON OPENCV
#contornos, _ = cv2.findContours(ImaBin, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #Encontrar los contornos y su jerarquia
#contornos, _ = cv2.findContours(ImaBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornos, _ = cv2.findContours(ImaBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(ImFull, contornos,-1,(0,255,0),5) #Dibujar los contornos en la imagen original

#Rectangle 
cnt = contornos[0]
M = cv2.moments(cnt)
print('Moments: ', M)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(ImagenInteres,(x,y), (x+w, y+h), (0,255,0), 2)
ME = 100
imgOut = ImagenInteres[2500:3500, 1500:2500 ]
imgOut = ImagenInteres[y-ME:y+h+ME, x-ME:x+h+ME]


#Muestra de la región de interés de la mamografía
plt.figure('Region de Interes de la Mamografía BOX')
plt.imshow(imgOut, cmap = 'gray')
plt.axis()
plt.show()

#Extraccion de area
u, v= np.shape(ImROI)
