import os
import numpy as np
import pandas as pd
import Funciones as fu
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pd.options.plotting.backend = "plotly"
os.getcwd()  # current directory
os.chdir(r'C:\Users\Erika\OneDrive - Universidad de Guanajuato\Fattening\Laser de Fibra-6cm\CurvaturaAscendente')
paramTitle = 'Curvature (1/mm)'   #curvature
dfParam = pd.read_csv('Inc.csv', skiprows=1,header=None, names=["fileName", "param"])
fileInit = dfParam["fileName"][0]
param = dfParam["param"].tolist()
xRange = [1530,1590]
[x, y,L] = fu.ReadFolderPout(fileInit, xRange, param) #x,y son listas de listas con diferente longitud
NOF = len(param)
df = pd.DataFrame()
df = fu.List2df(x,y,L,param)
val= 'max'
height=-60; thresh = -60; prom = 40;
df = fu.LinearityLaser(df, param, height, thresh, prom)
aux = []
"""
for i in range(len(param)):
    FWHM = df[~pd.isnull(df['FWHM'+str(i)])]['FWHM'+str(i)].tolist()
    aux.append(FWHM[0])
print(aux)
"""
#fig1 = fu.PlotInteractive(df, param, paramTitle, val)
#fig1.update_layout(title="Laser Fattening vs. Curvature Inc")
#fig1.show()
#Tunable in 3 regions
xRange = [1560,1562]
sel = [0,7,8,9,18,19]
paramSel = [param[i] for i in sel]
xRange = [1530,1568]
sel = [0,7,8,9,18,19]# señales para mostrar switcheo
paramSel = [param[i] for i in sel]
fu.LaserStability(df, xRange, paramSel)
print("END")
