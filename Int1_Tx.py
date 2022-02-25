#Uses python3
import os
import csv
import numpy as np
import pandas as pd
import Funciones as fu
pd.options.plotting.backend = "plotly"
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pylab import *
#import Matrix as mat
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from matplotlib.widgets import Cursor, Button


#cm = 1/2.54  # centimeters in inches

##main
#initial xRange
xRange = [1520,1570]
whichDir = os.getcwd()  # current directory
#load ASE
os.chdir('/home/estudiante/PythonDoctorado/Signals/Interferometro_01/Erbio')
#ASE corresponding to laser diode current = 140 mA
[xASE,yASE] = fu.LoadFile('W0255.CSV',29,xRange)
x = fu.DownSample(xASE,5)
yASE_Down = fu.DownSample(yASE,5)
# MZI folder
os.chdir('/home/estudiante/PythonDoctorado/Signals/Interferometro_01/Temperature/Inc')
whichDir = os.getcwd()
#temperatura
dfParam = pd.read_csv('Inc.csv', skiprows=1,header=None, names=["fileName", "param"])
param = dfParam["param"].tolist()
df = pd.DataFrame(list(zip(x,yASE_Down)), columns = ['Wavelength','ASE'])#lista de floats
fileInit = dfParam["fileName"][0]
#Read CSV files (.CSV upeprcase)
df = fu.ReadFolderTx(df, fileInit, param, xRange)
paramTitle = 'Temperature (Celsius deg)'
"""
fig0 = fu.PlotInteractiveTx(df, param, paramTitle)
fig0.show()

#Choose temperature reference
x1 = df["Wavelength"].tolist()
y1 = df[str(param[0])].tolist()
fu.TxRef(x1,y1,xRange)
"""
#Tx parametric
#Specify xRange
xRange = [1546, 1560]
#Specify varControl
varControl = 'Temp'
#Specify parameter values (range de 25 a 120 porque sólo allí emitio el laser)
indexSel = list(range(11))
paramSel = []
for i in range(len(indexSel)):
    k = indexSel[i]
    paramSel.append(param[k])
dfSel = fu.SelectDataFrame(df,xRange, param, indexSel)
fu.TxParametric(dfSel, varControl)
"""
#Para graficar todos los valores de param
fu.TxParametric(df,xRange, list(range(len(param))), varControl)
"""
#Linearity
val= 'min'
dfLin = fu.PointsLinearity(dfSel, val)
#df1 = fu.PointsLinearity(df,xRange, param, val)
fig1 = fu.PlotInteractiveLin(dfLin, paramSel, val)
fig1.update_layout(title="MZI Giselle vs Temperature Inc.")
fig1.show()
