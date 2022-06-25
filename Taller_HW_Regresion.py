# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 15:34:55 2022

@author: Elsy
"""

#Elsy Yuliana Silgado Rivera
#ID: 502194
#elsy.silgado@upb.edu.co

# Importamos librerias
import pandas as pd
import numpy as np
from sklearn import linear_model


# Se importa desde CSV y se lee archivo
cdf = pd.read_csv('cars.csv')

modelos = [(cdf["Car"] == "Audi"), (cdf["Car"] == "BMW"), (cdf["Car"] == "Fiat"),
    (cdf["Car"] == "Ford"), (cdf["Car"] == "Honda"), (cdf["Car"] == "Hundai"),
    (cdf["Car"] == "Mazda"),  (cdf["Car"] == "Mercedes"), (cdf["Car"] == "Mini"),
    (cdf["Car"] == "Mitsubishi"), (cdf["Car"] == "Opel"),  (cdf["Car"] == "Skoda"),
    (cdf["Car"] == "Suzuki"), (cdf["Car"] == "Toyoty"),   (cdf["Car"] == "VW"),
    (cdf["Car"] == "Volvo"),  (cdf["Car"] == "Hyundai")]
     
num = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0]

cdf["Number_Car"] =  np.select(modelos, num, default='Not Specified')
pd.to_numeric(cdf["Number_Car"])

# ********** Variables independiente ************ #
x = cdf[["Volume","Weight","CO2"]]

# Variable Dependiente
y = cdf["Number_Car"]

# *********** Regresion ***************** #
reg_mod = linear_model.LinearRegression()
reg_mod.fit(x,y)

# *********** Prediccion **************** #
predict_co2 = reg_mod.predict([[900,750,80]])
print(predict_co2)
print(reg_mod.coef_)
