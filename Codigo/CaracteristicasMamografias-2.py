#Obtencion de las caracteristicas de las ROI de las mamografias

#Librerias Utilizadas
from PIL.Image import Image
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2
#Entropy
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

from os import environ
def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

if __name__ == "__main__":
    suppress_qt_warnings()


#Lectura de archivo
w_d  = 'C:/Users/leoes/Desktop/9no Semestre/Investigacion/Imagenes_prueba/' #ruta
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

#Obtención de Características Relevantes
_, ImaBin = cv2.threshold(ImagenInteres,0,255,cv2.THRESH_BINARY) #Binarización de la Imagen
ImaBin = ImaBin.astype(np.uint8) #Conversión a Tipo de Dato UINT8 NECESARIA PARA TRABAJAR CON OPENCV
#contornos, _ = cv2.findContours(ImaBin, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #Encontrar los contornos y su jerarquia
contornos, _ = cv2.findContours(ImaBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(ImFull, contornos,-1,(0,255,0),5) #Dibujar los contornos en la imagen original

#Muestra de la imagen original marcando su región de interés
plt.figure('Marcado de la Región de Interes')
plt.imshow(ImFull)
plt.axis('off')
#plt.show()

# Obtención del Area
'''Pixeles contenidos en la región de Interés'''
# M = cv2.moments(contornos[0])
# print(M)
area = cv2.contourArea(contornos[0]) 
print(f'Area de la ROI: {area}')

# Obtención del Perímetro
'''Pixeles que constituyen el borde de la región de interés'''
perimetro =  cv2.arcLength(contornos[0], True)
print(f'Perimetro de la ROI: {perimetro}')

#Densidad
'''Relación entre el perímetro y el area'''
densidad = perimetro ** 2 / area
print(f'Densidad de la ROI: {densidad}')

#Compacidad
'''Relación normalizada entre el cuadrado del perímetro y el area de la región de interés'''
compacidad = perimetro ** 2 / (4 * np.pi * area)
print(f'Compacidad de la ROI: {compacidad}')

#Entropia del gradiente



#Contraste
contraste  = 0
for i in range(u):
    for j in range(v):
        contraste += ((i - j) ** 2) * ImagenInteres[i,j]

print(f'Contraste de la ROI: {contraste}')

#Correlación



'''REVISAR'''
#Uniformidad
ImagenUniformidad = ImagenInteres
'''
for i in range(u):
    for j in range(v):
        if (ImagenUniformidad[i,j]) <= 0:
            ImagenUniformidad[i,j] = 1
'''

uniformidad  = 0
for i in range(u):
    for j in range(v):
        uniformidad += ImagenUniformidad[i,j] ** 2

print(f'Uniformidad de la ROI: {uniformidad}')

'''END-REVISAR'''

plt.show()

