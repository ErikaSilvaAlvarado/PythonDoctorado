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
os.chdir(r'C:\Users\Erika\OneDrive - Universidad de Guanajuato\Giselle\INTERFEROMETRO_20_08_21\Inc')
xRange = [1520,1570]
#temperatura
dfParam = pd.read_csv('Inc.csv', skiprows=1,header=None, names=["fileName", "param"])
param = dfParam["param"].tolist()
[xASE,yASE] = fu.LoadFile('EDFA140.csv',29,xRange)              #descomentar
x = fu.DownSample(xASE,5)
yASE_Down = fu.DownSample(yASE,5)
df = pd.DataFrame(list(zip(x,yASE_Down)), columns = ['Wavelength','ASE'])#lista de floats
fileInit = dfParam["fileName"][0]
df = fu.ReadFolderTx(df, fileInit, param, xRange)
paramTitle = 'Temperature (Celsius deg)'
#fu.Transmission(df,[1520,1580],paramTitle).
val= 'min'
df1 = fu.PointsLinearity(df,xRange, param, val)
fig1 = fu.PlotInteractive(df1, param, paramTitle, val)
fig1.update_layout(title="MZI Giselle vs Temperature Inc.")
fig1.show()

#fu.FastFourierPlot(df,xRange)
#xRange = [1547,1560]; val= 'min'
#xRange = [1520,1580]; val= 'min'
#xRange = [1535,1550]; val= 'min'
#xRange = [1550,1580]; val= 'min'
#fu.Linear(df,xRange,val,paramTitle)