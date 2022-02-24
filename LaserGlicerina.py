import os
import numpy as np
import pandas as pd
import Funciones as fu
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pd.options.plotting.backend = "plotly"
os.getcwd()  # current directory
os.chdir(r'C:\Users\Erika\OneDrive - Universidad de Guanajuato\Giselle\INTERFEROMETRO_20_08_21\Láser\caracterización glicerina')
paramTitle = 'Temperature (Celsius deg)'
dfParam = pd.read_csv('Inc.csv', skiprows=1,header=None, names=["fileName", "param"])
fileInit = dfParam["fileName"][0]
param = dfParam["param"].tolist()
xRange = [1520,1580]

[x, y,L] = fu.ReadFolderPout(fileInit, xRange, param) #x,y son listas de listas con diferente longitud
NOF = len(param)
df = pd.DataFrame()
df = fu.List2df(x,y,L,param)
val= 'max'
height=-60; thresh = -60; prom = 40;
df = fu.LinearityLaser(df, param, height, thresh, prom)
fig1 = fu.PlotInteractive(df, param, paramTitle, val)
fig1.update_layout(title="Laser Giselle Glyc under Temp")
fig1.show()
