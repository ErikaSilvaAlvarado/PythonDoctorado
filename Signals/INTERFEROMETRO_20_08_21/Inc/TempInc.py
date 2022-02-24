#Uses python3
import os
import csv
import numpy as np
import pandas as pd
import plotly.express as px
import Funciones as fu

pd.options.plotting.backend = "plotly"

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.callbacks import Points, InputDeviceState
from ipywidgets import widgets
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
colorLegend =['aqua', ' black', ' blue', ' blueviolet', ' brown', ' cadetblue', ' chocolate', ' coral',
                    ' cornflowerblue', ' crimson', ' darkblue', ' darkcyan', ' darkmagenta', ' darkorange', ' darkred',
                    ' darkseagreen', ' darkslategray', ' darkviolet', ' deeppink', ' deepskyblue', ' dodgerblue',
                    ' firebrick', ' forestgreen', ' fuchsia', ' gold', ' goldenrod', ' green', ' hotpink', ' indianred',
                    ' indigo', ' orangered', ' purple', ' rebeccapurple', ' red', ' saddlebrown', ' salmon',
                    ' seagreen', ' sienna', ' slateblue', ' steelblue', ' violet', ' yellowgreen', 'aqua', 'aquamarine',
                    'darkgoldenrod', 'darkorchid', 'darkslateblue', 'darkturquoise', 'greenyellow', 'navy',
                    'palevioletred', 'royalblue', 'sandybrown', 'turquoise']

##main
os.getcwd()
xRange = [1500,1600]

#temperatura
dfParam = pd.read_csv('Inc.csv', skiprows=1,header=None, names=["fileName", "param"])
param = dfParam["param"].tolist()
[xASE,yASE] = fu.LoadFile('EDFA140.csv',29)              #descomentar
x = fu.DownSample(xASE,5)
yASE_Down = fu.DownSample(yASE,5)
df = pd.DataFrame(list(zip(x,yASE_Down)), columns = ['Wavelength','ASE'])#lista de floats
xRange = [1500,1600]
df = fu.ReadFolderTx(df, dfParam["fileName"].tolist(), dfParam["param"].tolist(), xRange)
A = df["Wavelength"].tolist()

fig1 = make_subplots()
for i in range(len(param)):
    #A = df["Wavelength"+str(i)].tolist()
    B = df[str(param[i])]
    fig1.add_trace(go.Scatter(
    x=A,
    y=B,
    mode="lines",
    line_color=colorLegend[i],
    name=str(param[i])
    ))
fig1.show()

paramTitle = r'$\mathrm{Temp.} (^{\circ}C)$'
#fu.Transmission(df,[1520,1580],paramTitle)
#xRange = [1520,1570]; val= 'max'
xRange = [1500,1570]; val= 'min'
col_names = df.columns.values[2:]
paramStr = col_names.tolist()
df1 = fu.PointsLinearity(df,xRange, paramStr, val)
fu.PlotInteractive(df1, paramStr, paramTitle, val)
#fu.FastFourierPlot(df,xRange)
#xRange = [1547,1560]; val= 'min'
#xRange = [1520,1580]; val= 'min'
#xRange = [1535,1550]; val= 'min'

#xRange = [1550,1580]; val= 'min'
#fu.Linear(df,xRange,val,paramTitle)