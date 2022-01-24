#Obtencion de las caracteristicas de las ROI de las mamografias

#Librerias Utilizadas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
import cv2

# #Lectura de archivo CSV de datos y direcciones
databaseCALC = pd.read_csv("calc_case_description_train_set.csv") #Calcificaciones
#databaseMASS = pd.read_csv("mass_case_description_train_set.csv") #Masas

# #Filtro por vistas Craneocaudales (CC)
databaseCALCCC = databaseCALC[databaseCALC['image view'].str.contains("CC")] #Calcificaciones
# # databaseMASSCC = databaseMASS[databaseMASS['image view'].str.contains("CC")] #Masas

# #Seleccion de Caso
NumeroDeCaso = 'P_00008'
DatosDelCaso = databaseCALCCC[databaseCALCCC['patient_id'].str.contains(NumeroDeCaso)]

# #Lectura de direccion de mamografía completa - DERECHA
IMGRIGHT = DatosDelCaso[DatosDelCaso['left or right breast'].str.contains('RIGHT')]
PATHIMGRIGHT  = list(IMGRIGHT["image file path"])
# PATHIMGRIGHT = PATHIMGRIGHT [0] 

# #Obtención de las direcciones de las ROI - DERECHA
# ROIIMGRIGHT = list(IMGRIGHT["ROI mask file path"])

# #Lectura de direccion de mamografía completa - IZQUIERDA
# IMGLEFT = DatosDelCaso[DatosDelCaso['left or right breast'].str.contains('LEFT')]
# PATHIMGLEFT = list(IMGLEFT["image file path"])
# PATHIMGLEFT = PATHIMGLEFT [0] 

# #Obtención de las direcciones de las ROI - IZQUIERDA
# ROIIMGLEFT = list(IMGLEFT["ROI mask file path"])

# #Lectura de las Imagenes
# #Derecha
ImDCMFull = pydicom.dcmread('Imagenes\P_00005.dcm')
ImDCMROI = pydicom.dcmread('Imagenes\P_00005-ROI.dcm')


#Izquierda
# ImDCMLeftFull = pydicom.dcmread(PATHIMGLEFT)
# ImLeftFull = np.asarray(ImDCMFull.pixel_array)

# ImleftFull = 0
# aux1 = 0
# aux2 = 0
# for i in ROIIMGLEFT:
#     aux1 = pydicom.dcmread(ROIIMGLEFT[i])
#     aux2 = np.asarray(aux1.pixel_array)
#     ImleftFull = aux2 + ImleftFull   
    

#Conversión a arreglos de Numpy
ImFull = np.asarray(ImDCMFull.pixel_array)
ImROI = np.asarray(ImDCMROI.pixel_array)


#Muestra de Imagen Original
# plt.figure('Mamografía Original')
# plt.imshow(ImFull, cmap = 'gray')
# plt.axis('off')

#Muestra de la máscara de la región de Interes
# plt.figure('Máscara del Área de Interés')
# plt.imshow(ImROI, cmap = 'gray')
# plt.axis('off')


#Creación de Imagen que únicamente contiene el área de Interés con 1 en regiones de interes
u, v= np.shape(ImROI)
IdentidadROI = np.zeros((u,v))

#Cambiando 255 por 1
for i in range(u):
    for j in range(v):
        if(ImROI[i,j]) > 0:
            IdentidadROI[i,j] = 1

#Multiplicando elemento con elemento
ImagenInteres = np.multiply(ImFull,IdentidadROI)

#Muestra de la región de interés de la mamografía
# plt.figure('Region de Interes de la Mamografía')
# plt.imshow(ImagenInteres, cmap = 'gray')
# plt.axis('off')



#Obtención de Características Relevantes
_, ImaBin = cv2.threshold(ImagenInteres,0,255,cv2.THRESH_BINARY) #Binarización de la Imagen

ImaBinUI8 = ImaBin.astype(np.uint8) #Conversión a Tipo de Dato UINT8 NECESARIA PARA TRABAJAR CON OPENCV

contornos, _ = cv2.findContours(ImaBinUI8, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #Encontrar los contornos y su jerarquia

cv2.drawContours(ImFull, contornos,-1,(0,255,0),5) #Dibujar los contornos en la imagen original

# print(contornos[0])

#Muestra de la imagen original marcando su región de interés
# plt.figure('Marcado de la Región de Interes')
# plt.imshow(ImFull, cmap = 'gray')
# plt.axis('off')
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
        contraste = contraste +  ((i - j) ** 2) * ImagenInteres[i,j]

print(f'Contraste de la ROI: {contraste}')

#Correlación

#Uniformidad
ImagenUniformidad = ImagenInteres

uniformidad = 1
for i in range(u):
    for j in range(v):
        if(ImagenUniformidad[i,j] > 0):
            uniformidad = uniformidad * (ImagenUniformidad[i,j] ** 2)


print(f'Uniformidad de la ROI: {uniformidad}')





