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

numberFig = 0
cm = 1/2.54  # centimeters in inches
resolDPI = 300

font = {'size': 12,
        'stretch':'ultra-condensed',
        'weight': 'ultralight'
        }


##main
os.getcwd()  # current directory
os.chdir(r'C:\Users\Erika\OneDrive - Universidad de Guanajuato\Fattening\Interferometro 6cm\6cmCurvaturaAscendente')
xRange = [1510,1580]
#temperatura
dfParam = pd.read_csv('Inc.csv', skiprows=1,header=None, names=["fileName", "param"])
param = dfParam["param"].tolist()
[xASE,yASE] = fu.LoadFile('EDFA400.csv',29,xRange)              #descomentar
x = fu.DownSample(xASE,5)
yASE_Down = fu.DownSample(yASE,5)
df = pd.DataFrame(list(zip(x,yASE_Down)), columns = ['Wavelength','ASE'])#lista de floats
fileInit = dfParam["fileName"][0]
df = fu.ReadFolderTx(df, fileInit, param, xRange)
#[x,y,L] = fu.ReadFolderPout(fileInit, xRange, param)
#df = fu.List2df(x,y,L,param)
paramTitle = 'Curvature(1/m)'
#fu.Transmission(df,[1520,1580],paramTitle).
val= 'min'
df1 = fu.PointsLinearity(df,xRange, param, val)
fig1 = fu.PlotInteractive(df1, param, paramTitle, val)
fig1.update_layout(title="MZI Fattening vs Curvature Inc.")
fig1.show()

val= 'max'
df1 = fu.PointsLinearity(df,xRange, param, val)
fig2 = fu.PlotInteractive(df1, param, paramTitle, val)
fig2.update_layout(title="MZI Fattening vs Curvature Inc.")
fig2.show()