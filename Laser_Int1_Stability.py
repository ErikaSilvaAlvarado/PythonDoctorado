import os
import numpy as np
from scipy import signal
import pandas as pd
import Funciones as fu
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
pd.options.plotting.backend = "plotly"

#main
os.getcwd()  # current directory
os.chdir('/home/estudiante/PythonDoctorado/Signals/Interferometro_01/Laser/Stability')
paramTitle = 'Time (s)'
#Parametros para cargar relacion archivo vs tiempo de muestra
dfTime = pd.read_csv('Stability.csv', skiprows=1,header=None, names=["fileName", "time"])
fileInit = dfTime["fileName"][0]
time = dfTime["time"].tolist()
timeSel = fu.DownSample(time,4)
NOF = len(time)
xRange = [1542, 1560]
yRange = [-80, -10]
#x,y son listas de listas. L es una lista con la longitud de x[i}, y[i]
[x,y,L] = fu.ReadFolderStability(fileInit, xRange, yRange, time)
#fig = fu.PlotLists(x, y, L)
#Select the signal having more Ppeak if has narrow FWHM
kymax, ymax, FWHM = fu.SelectLaserSignal(x, y, L)
xSel = np.array(x[kymax])
ySel = np.array(y[kymax])
Lsel = L[kymax]
#fig = fu.SignalPlot(xSel,ySel)
#Detecting peaks parameters
height = yRange[0]
prom = 2
dist = 1000
#Generate png with SMSR and FWHM
fu.PlotLaserFeatures(xSel,ySel, xRange, yRange, height, prom, dist)
#Generate interative wwavetfall
fu.LaserStability3DInteractive(x,y,timeSel)
##Generate Waterfall 3D png
fu.LaserStability3D(x, y, timeSel, xRange)
print("End")
