import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2

from os import environ
def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

if __name__ == "__main__":
    suppress_qt_warnings()

def filter_patient(cc_filter, data):
    #Filtro CC derecha en dataset
    patient = cc_filter[cc_filter["left or right breast"].str.startswith(data)] #Dataframe completo
    #Lista de Pacientes CC derecha
    patient_PI = set(list(patient["patient_id"])) #
    if data == 'R':
        patient_PI = [i + '_RIGHT_CC' for i in patient_PI]
    else:
        patient_PI = [i + '_LEFT_CC' for i in patient_PI]
    patient_PI.sort()
    return patient_PI
   
def directory_read(metadata,patient_list,data ): #dataframe/ R o L / 
    #Lectura de Directorios de Imagenes Derecha
    patient_right_coincidences = patient_list[data]
    paths_cc_right = metadata[metadata["File Location"].str.contains(patient_right_coincidences)]
    files_location_cc_right = list(paths_cc_right["File Location"])
    files_location_cc_right.sort()

    full = files_location_cc_right[0] #Mamografia Orig
    roi = files_location_cc_right[1:] #ROIs
    return full,roi
 
# Lectura Archivo metadata   
metadata = pd.read_csv('metadata.csv',index_col=False)
metadata = metadata[metadata["File Location"].str.startswith('.\CBIS-DDSM\Calc-Training')]
# Lectura Archivo calc_case_description_train_set
calc_trainig_data_file = pd.read_csv('calc_case_description_train_set.csv', index_col=False)
#Filtro CC en dataset
cc_filter = calc_trainig_data_file[calc_trainig_data_file["image view"].str.startswith('CC')]


# filtro pacientes
patient_right_cc_PI = filter_patient(cc_filter,'R') # filtro derecha
patient_left_cc_PI  = filter_patient(cc_filter,'L') # filtro izquierda

# Lectura de directorios 
path_R_full, path_R_ROI = directory_read(metadata,patient_right_cc_PI, 14)
path_L_full, path_L_ROI = directory_read(metadata,patient_left_cc_PI, 0)


#Lectura de imagenes DCM
dcm_1 = '/1-1.dcm'
dcm_2 = '/1-2.dcm'

#Lectura de imagen mamografia completa
ImDCMFull = pydicom.dcmread(path_R_full + dcm_1)
ImFull = np.asarray(ImDCMFull.pixel_array)
u, v= np.shape(ImFull)
zeros_size = np.zeros((u,v))

#Lectura de ROI's

for path in path_R_ROI:

    ImROI_1 = pydicom.dcmread(path + dcm_1)
    ImROI_1 = np.asarray(ImROI_1.pixel_array)
    u_1, v_1 = np.shape(ImROI_1)
    
    ImROI_2 = pydicom.dcmread(path + dcm_2)
    ImROI_2 = np.asarray(ImROI_2.pixel_array)
    u_2, v_2 = np.shape(ImROI_2)
    
    if (u_1 == u) and (v_1 == v):
        print('ROI_1')
    elif (u_2 == u) and (v_2 == v):
        print('ROI_2')
    else:
        print('Camara puto')
    

    
    


# plt.figure('Mamografía Original')
# plt.suptitle('Full Mammografia')
# plt.imshow(ImFull, cmap = 'gray')
# plt.axis('off')



# plt.figure('ROI 2')
# plt.suptitle('ROI 2')
# plt.imshow(ImROI_2, cmap = 'gray')
# plt.axis('off')



    






# patient_right_cc_PI = patient_right_cc[]






# training_filter = origin_file[origin_file["File Location"].str.startswith('.\CBIS-DDSM\Calc-Training')]


# cc_filter = training_filter[training_filter["File Location"].str.contains('CC')]

# full_image_filter = cc_filter[training_filter["Series Description"].str.contains('full mammogram images')]

# full_image_path = full_image_filter["File Location"]
# print(full_image_path.head(5))

