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
#temperatura
dfParam = pd.read_csv('Dec.csv', skiprows=1,header=None, names=["fileName", "param"])
param = dfParam["param"].tolist()
[xASE,yASE] = fu.LoadFile('EDFA140.csv',29)              #descomentar
x = fu.DownSample(xASE,5)
yASE_Down = fu.DownSample(yASE,5)                  #lista de floats
df = pd.DataFrame(list(zip(x,yASE_Down)), columns = ['Wavelength','ASE'])
xRange = [1500,1600]
df = fu.ReadFolderTx(df, dfParam["fileName"].tolist(), dfParam["param"].tolist(), xRange)
A = df["Wavelength"].tolist()
#df = fu.ReadFolderPout(dfParam["fileName"], param)

"""
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
"""
paramTitle = r'$\mathrm{Temp.} (^{\circ}C)$'
#fu.Transmission(df,[1520,1580],paramTitle)
xRange = [1520,1570]; val= 'max'
#xRange = [1520,1570]; val= 'min'
col_names = df.columns.values[2:]
paramStr = col_names.tolist()
df1 = fu.PointsLinearity(df,xRange, paramStr, val)
"""
for i in range(len(paramStr)):
   AA = df1[~pd.isnull(df1['min' + paramStr[i]])]['Wavelength'].tolist()
   AL=len(AA)
"""
fu.PlotInteractive(df1, paramStr, paramTitle, val)
#fu.LinearSel(df,xRange, val, paramStr,paramTitle)
#xRange = [1547,1560]; val= 'min'
#xRange = [1520,1580]; val= 'min'
#xRange = [1535,1550]; val= 'min'
#
#xRange = [1550,1580]; val= 'min'




###NEXT



"""
for i in range(NS):
    plt.plot(xAll[i], yAll[i])
#legend()
xlim(1500,1600)
xlabel('Wavelength (nm)', fontsize=14)
ylabel('Output power (dBm)', fontsize=14)
ylim(-80,-20)
plt.tick_params( labelsize='medium', width=1)
auxWidth = 8.9*cm
auxHeight = 8*cm
figure = plt.gcf()
figure.set_size_inches(auxWidth, auxHeight)
plt.savefig("Parametric.png", dpi=300, bbox_inches="tight",pad_inches=0.1,transparent=True)
plt.show()
"""