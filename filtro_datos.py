import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2


metadata = pd.read_csv('metadata.csv',index_col=False)
calc_trainig_data_file = pd.read_csv('calc_case_description_train_set.csv', index_col=False)

#Filtro CC en dataset
cc_filter = calc_trainig_data_file[calc_trainig_data_file["image view"].str.startswith('CC')]

#Filtro CC derecha en dataset
patient_right_cc = cc_filter[cc_filter["left or right breast"].str.startswith('R')] #Dataframe completo
#Filtro CC derecha en dataset
patient_left_cc = cc_filter[cc_filter["left or right breast"].str.startswith('L')]

#Lista de Pacientes CC derecha
patient_right_cc_PI = set(list(patient_right_cc["patient_id"])) #Lista
patient_right_cc_PI = [i + '_RIGHT_CC' for i in patient_right_cc_PI]
patient_right_cc_PI.sort()

#Lista de Pacientes CC izquierda
patient_left_cc_PI = set(list(patient_left_cc["patient_id"]))
patient_left_cc_PI = [i + '_LEFT_CC' for i in patient_left_cc_PI]
patient_left_cc_PI.sort()

#Seleccion de Paciente Derecha
patient_number_right = 1

#Lectura de Directorios de Imagenes Derecha
patient_right_coincidences = patient_right_cc_PI[patient_number_right]
paths_cc_right = metadata[metadata["File Location"].str.contains(patient_right_coincidences)]
files_location_cc_right = list(paths_cc_right["File Location"])
files_location_cc_right.sort()

files_location_cc_right_full = files_location_cc_right[0] #Mamografia Orig
files_location_cc_right_ROI = files_location_cc_right[1:] #ROIs


#Lectura de imagenes DCM
dcm_1 = '/1-1.dcm'
dcm_2 = '/1-2.dcm'

ImDCMFull = pydicom.dcmread(files_location_cc_right_full + dcm_1)
ImFull = np.asarray(ImDCMFull.pixel_array)








    






# patient_right_cc_PI = patient_right_cc[]






# training_filter = origin_file[origin_file["File Location"].str.startswith('.\CBIS-DDSM\Calc-Training')]


# cc_filter = training_filter[training_filter["File Location"].str.contains('CC')]

# full_image_filter = cc_filter[training_filter["Series Description"].str.contains('full mammogram images')]

# full_image_path = full_image_filter["File Location"]
# print(full_image_path.head(5))

