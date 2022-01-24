# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 23:58:27 2021

@author: leoes
"""
import pandas as pd
import matplotlib.pyplot as plt
#import math as mt

'''Filter'''
#filter per gender
def filter_data(df,filtro,atribute_1): #values/filter/atribute origin/atribute to filter
    #gender = gender.lower()
    gen_filter = df[atribute_1] == filtro  #Filter by gender
    #gen = df[gen_filter][atribute_2].tolist[]
    gen = df[gen_filter]
    
    return gen


'''Main'''
#Read file
#w_d = 'C:/Users/leoes/Desktop/9no Semestre/Investigacion/archivos/'
w_d = 'C:\\Users\\leoes\\Desktop\\9no Semestre\\Investigacion\\archivos\\'
f_i = w_d + 'calc_case_description_train_set.csv'
df = pd.read_csv(f_i) #Create a Dataframe from a CSV 

fil = filter_data(df, 'CC', 'image view')
# Guarda datos en CSV:
fil.to_csv('filtrado_CC.csv', header=True, index=False)
df_cc_1 = filter_data(fil, 3, 'assessment')
