# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:29:13 2020

@author: uasmt
"""

import DTClassification as DTC
import pandas as pd
import numpy as np
import warnings
# filter warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('appointment.csv')
# hazır veritabanı olduğu için bize yardı11mcı olmayacak nitelikleri çıkartalım.
data = data.drop(["PatientId"],axis=1)
data = data.drop(["AppointmentID"],axis=1)
data = data.drop(["AppointmentDay"],axis=1)
data = data.drop(["Neighbourhood"],axis=1)
data = data.drop(["ScheduledDay"],axis=1)

# yaş kısmı için medyanını bulalım.
ages = data["Age"]
sortedValues = ages.sort_values()
uniqueValues = sortedValues.unique()
median = np.median(uniqueValues)
#medyan degerinden kucuk olanlara smaller, buyuk olanlara bigger diyelim. 
data['Age'] = ['bigger' if x >= median else 'smaller' for x in data['Age']]

# nitelikleri hedeften ayıralım.
features = data.drop(["No-show"],axis=1)
# niteliklerin isimlerini label diyelim
labels = list(features.columns.values)
# datamızı list olarak cevirelim
dataList = data.values.tolist()

#Olusturdugumuz Kara agacı sınıflandırmamıza labelleri ve verisetimii yerleştiriyoruz.
tree = DTC.DecisionTreeClassificaiton(dataList,labels)
print('İlk Dictonary olarak Ağaç Modelimiz:\n',tree.model)
#Olusan modelimizde verdiğimiz tahminlere göre test ediyoruz.
tree.predict(tree.model,['Gender','Age','Scholarship','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received'],['F','bigger',0,0,0,0,0,0])


#bilinen hava durumu için classımızı burdan test edebiliriz.
#iki faklı veri seti testi
weather = pd.read_csv('weather.csv',delimiter=';')

features2 = weather.drop(["Play"],axis=1)
labels2 = list(features2.columns.values)
dataList2 = weather.values.tolist()

tree2 = DTC.DecisionTreeClassificaiton(dataList2,labels2)
print('\n İkinci Dictonary olarak Ağaç Modelimiz:\n',tree2.model)
tree2.predict(tree2.model,['Outlook','Temperature','Humidity','Wind'],['Sunny','Hot','Normal','Weak'])


