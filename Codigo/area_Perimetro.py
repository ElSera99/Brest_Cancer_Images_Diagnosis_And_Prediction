import cv2
import numpy as np
#import matplotlib.pyplot as plt
from numpy.core.fromnumeric import resize

def resise(imagen, scale_percent ):
    width = int(imagen.shape[1] * scale_percent / 100)
    height = int(imagen.shape[0] * scale_percent / 100)
    return (width, height)
    

w_i = 'P_00005/imagen1.dcm'
 


imagen = cv2.imread(w_i, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
print('Original NORMAL Dimensions : ',imagen.shape)
print('Original GRAY Dimensions : ',gray.shape)


# resize image
dim = resise(imagen, 20)
imagen = cv2.resize(imagen, dim)
#dim2 = resise(gray, 20)
gray = cv2.resize(gray, dim)

#imagen = cv2.resize(imagen, (0, 0), fx=0.9, fy=0.9)
#gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

print('Original NORMAL Dimensions : ',imagen.shape)
print('Original GRAY Dimensions : ',gray.shape)

ret, th = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY)
contornos, jerarquia = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = contornos[1]
M = cv2.moments(cnt)
print(M)
cX = int(M['m10']/M['m00'])
cY = int(M['m01']/M['m00'])

print(f'cX: {cX}')
print(f'cY: {cY}')

area = cv2.contourArea(cnt)
print(f'Area: {area} px^2')
perimetro = cv2.arcLength(cnt, True)
print(f'Perimetro: {perimetro} px')


# Buscamos los contornos de las bolas y los dibujamos en verde
contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imagen, contours, -1, (0, 255, 0), 4)


cv2.circle(imagen, (cX,cY), 5, (0,255,0), -1)
cv2.putText(imagen, "x: "+str(cX)+", y: "+str(cY), (cX,cY), 1,1, (0,0,0),1)

#plt.imshow(imagen)
#plt.show()


cv2.imshow('Image', imagen)
#cv2.resizeWindow('resized', (300,300))
#cv2.imshow('Gray', gray)
cv2.imshow('th', th)
cv2.waitKey(0)
cv2.destroyAllWindows()