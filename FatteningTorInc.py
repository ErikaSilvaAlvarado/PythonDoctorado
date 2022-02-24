#Uses python3
import os
import csv
import numpy as np
import pandas as pd
import math
import pywt
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
os.chdir(r'C:\Users\Erika\OneDrive - Universidad de Guanajuato\Fattening\Interferometro 6cm\6cmTorsionAscendente')
xRange = [1510,1580]

dfParam = pd.read_csv('Inc.csv', skiprows=1,header=None, names=["fileName", "param"])
param = dfParam["param"].tolist()
[xASE,yASE] = fu.LoadFile('EDFA400.csv',29,xRange)              #descomentar
x = fu.DownSample(xASE,5)
yASE_Down = fu.DownSample(yASE,5)
df = pd.DataFrame(list(zip(x,yASE_Down)), columns = ['Wavelength','ASE'])#lista de floats
fileInit = dfParam["fileName"][0]
df = fu.ReadFolderTx(df, fileInit, param, xRange)
paramTitle = 'torsio'
val= 'min'
df1 = fu.PointsLinearity(df,xRange, param, val)
fig1 = fu.PlotInteractive(df1, param, paramTitle, val)
fig1.update_layout(title="MZI Fattening vs Torsion Inc.")
fig1.show()

x = df['Wavelength'].tolist()
x = fu.DownSample(x,2)
y = df[str(param[0])].tolist()
y = fu.DownSample(y,2)
fu.SignalPlot(x,y)
#h = np.hanning(len(y)-1)
#h = h.tolist()
#ywin = [a*b for a,b in zip(y,h)]
#y = ywin
[sF, mY] = fu.FastFourier(x, y)
yprom = sum(y)/len(y)
y = y-yprom
fu.SignalPlot(x,y)
fig0 = fu.SignalSpectrogram(x,y)
fig0.show()
#[x,y,L] = fu.ReadFolderPout(fileInit, xRange, param)
#x0 = x[0]; y0 = y[0]
#fu.FastFourierPlot(x, y)
#fu.SignalPlot(x,y)
N = len(x)
MW = 'db14'
dx = round((x[1]-x[0]),4)
Fs = 1/dx
DLmax = pywt.dwt_max_level(N, MW)
Bmin =0.085
DL = int(floor(log2(Fs/Bmin)))

#Bmin = Fs/(2**(DL+1))

fig1, fig2 = fu.WaveletDecomposition(x, y, MW,DL+1)
fig1.show()
fig2.show()
"""
#df = fu.List2df(x,y,L,param)
paramTitle = 'Torsion (Deg)'
#fu.Transmission(df,[1520,1580],paramTitle).
val= 'min'
df1 = fu.PointsLinearity(df,xRange, param, val)
fig1 = fu.PlotInteractive(df1, param, paramTitle, val)
fig1.update_layout(title="MZI Fattening vs Torsion Inc.")
fig1.show()
"""