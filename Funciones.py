import os
import csv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.collections import PolyCollection
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import signal
from scipy.signal import argrelextrema
from scipy.fft import fft, ifft, fftfreq
import pywt



def ReadFolderPout(fileInit, xRange, param):
    #Read files (only xRange interval)
    x = []; y = []; L = [];
    NOF =len(param) # número de columnas
    for i in range(NOF):
        if fileInit + i  < 10:
             file = 'W000' + str(fileInit + i) + '.csv'
        else:
             if fileInit + i  < 100:
                file = 'W00' + str(fileInit + i) + '.csv'
             else:
                file = 'W0' + str(fileInit + i) + '.csv'
        [xi, yi] = LoadFile(file, 29, xRange)
        x.append(xi)
        y.append(yi)
        L.append(len(xi))
    return [x,y,L]

def LoadFile(file,jump,xRange):
    with open(file, newline='') as file:
        reader = csv.reader(file, delimiter =',')
        for k in range(jump):
            next(reader)
        xi = []; yi = []
        for row in reader:
            auxX = float(row[0])
            auxY = float(row[1])
            if (auxX >= xRange[0] and auxX <= xRange[1]):
                xi.append(auxX)
                yi.append(auxY)
    return [xi,yi]

def List2df(x,y,L,param):
#unifico la longitud de las listas para volverlas dataframe
    NOF = len(param)
    Lmax = max(L)
    for i in range(NOF):
        Li = L[i]
        if Li < Lmax:
            xMissed = (Lmax - Li)
            noisyPAd = np.random.normal(-0.1, 0.2, xMissed)
            nP= noisyPAd.tolist()
            yP = [y[i][Li-1]] * xMissed
            yPad = [sum(n) for n in zip(nP,yP)]
            auxList = y[i] + yPad
            y[i] = auxList
            if i == 0:
                xStep = round(x[i][1] - x[i][0], 4)
                x0 = x[i][Li-1]
                xPad = [x0 + x * xStep for x in range(0, xMissed)]
                x[i] = x[i] + xPad
                df = pd.DataFrame(list(zip(x[i], y[i])), columns=['Wavelength', str(param[i])])
            else:
                df[str(param[i])] = y[i]
        else:
            if i == 0:
                df = pd.DataFrame(list(zip(x[i], y[i])), columns=['Wavelength', str(param[i])])
            else:
                df[str(param[i])] = y[i]
    return df

def PointsLinearity(df, xRange, param, val):
    df1 = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])]
    NOF = len(param)
    paramStr = []
    if val == 'max':
        for i in range(NOF):
            paramStr.append(str(param[i]))
            df1['max' + str(i)] = df1.iloc[argrelextrema(df1[paramStr[i]].values, np.greater_equal, order=15)[0]][paramStr[i]]

    elif val == 'min':
        for i in range(NOF):
            paramStr.append(str(param[i]))
            df1['min' + str(i)] = df1.iloc[argrelextrema(df1[paramStr[i]].values, np.less_equal, order=15)[0]][
                    paramStr[i]]

    else:
        valY1 = df1[(df1[paramStr] >= val)][paramStr]
        kval = df1[(df1[paramStr] >= val)][paramStr].idxmin()
        valX1 = df1["Wavelength"].loc[kval].tolist()
    return df1

def LinearityLaser(df, param, height, thresh, prom):
    NOF = len(param)
    paramStr = []; FWHM = []
    for i in range(NOF):
        paramStr.append(str(param[i]))
        peaksIndex,properties = signal.find_peaks(df[paramStr[i]], height=height, threshold=thresh, prominence=prom)
        Pmax = properties["peak_heights"]
        df['max' + str(i)] = df.loc[peaksIndex][paramStr[i]]
        for j in range(len(Pmax)):
            init = properties["left_bases"][j]
            ending = properties["right_bases"][j]
            k=[index for index, value in enumerate(df[properties["left_bases"][j]:properties["right_bases"][j]][paramStr[i]]) if value > Pmax[j] - 3]
            k1 = init + k[1]
            k2 = init + k[-1]
            FWHM = df.iloc[k2,0]- df.iloc[k1,0]
            if j==0:
                df['FWHM' + str(i)] = df.loc[peaksIndex][paramStr[i]] # onlycreating the column
            df['FWHM' + str(i)][peaksIndex[j]] = FWHM
    return df

def PlotInteractive(df1, param, paramTitle, val):
    NOF = len(param)
    colorLegend =[ ' black', ' blue', ' blueviolet', ' brown', ' cadetblue', ' chocolate', ' coral',
                    ' cornflowerblue', ' crimson', ' darkblue', ' darkcyan', ' darkmagenta', ' darkorange', ' darkred',
                    ' darkseagreen', ' darkslategray', ' darkviolet', ' deeppink', ' deepskyblue', ' dodgerblue',
                    ' firebrick', ' forestgreen', ' fuchsia', ' gold', ' goldenrod', ' green', ' hotpink', ' indianred',
                    ' indigo', ' orangered', ' purple', ' rebeccapurple', ' red', ' saddlebrown', ' salmon',
                    ' seagreen', ' sienna', ' slateblue', ' steelblue', ' violet', ' yellowgreen', 'aqua', 'aquamarine',
                    'darkgoldenrod', 'darkorchid', 'darkslateblue', 'darkturquoise', 'greenyellow', 'navy',
                    'palevioletred', 'royalblue', 'sandybrown']

    A = df1["Wavelength"].tolist()
    fig1 = make_subplots(1,2)
    paramStr = []
    for i in range(NOF):
        paramStr.append(str(param[i]))
        B = df1[str(param[i])]
        fig1.add_trace(go.Scatter(
            x=A,
            y=B,
            legendgroup = 'lgd'+str(i),
            name=paramStr[i],
            mode="lines",
            line_color=colorLegend[i],
            ),row=1, col=1)
    fig1.update_layout(legend_title_text=paramTitle)
    for i in range(len(paramStr)):
        A1 = df1[~pd.isnull(df1[val + str(i)])]['Wavelength'].tolist()
        B1 = df1[~pd.isnull(df1[val + str(i)])][paramStr[i]].tolist()
        fig1.add_trace(go.Scatter(
            x=A1,
            y=B1,
            legendgroup = 'lgd'+ str(i),
            name =paramStr[i],
            mode ="markers",
            marker_color = colorLegend[i],
            showlegend=False
            ),row =1, col =1)
    for i in range(len(paramStr)):
        BB = df1[~pd.isnull(df1[val + str(i)])]['Wavelength'].tolist()
        AA = [param[i]]*len(BB)
        fig1.add_trace(go.Scatter(
            x= AA,
            y=BB,
            legendgroup ='lgd' + str(i),
            name =paramStr[i],
            mode ="markers",
            marker_color = colorLegend[i],
            showlegend=False,
            ),row=1, col=2)
    return fig1

def DownSample(x,m):
    xDown = []
    i = 0
    while i <= len(x):
        if (i % m )==0:
             xDown.append(x[i])
        i = i+1
    return(xDown)

def ReadFolderTx(df, fileInit, param, xRange):
    xi = []; yi = []
    NOF =len(param) # número de columnas
    for i in range(NOF):
        if fileInit + i  < 10:
             file = 'W000' + str(fileInit + i) + '.csv'
        else:
             if fileInit + i  < 100:
                file = 'W00' + str(fileInit + i) + '.csv'
             else:
                file = 'W0' + str(fileInit + i) + '.csv'
        [xi,yi] = LoadFile(file, 29, xRange)
        df[str(param[i])] = yi - df['ASE']
    return df

def ReadFolderLaserSame(df, fileInit, param, xRange):
    x = []; y = []
    NOF =len(param) # número de columnas
    for i in range(NOF):
        if fileInit + i  < 10:
             file = 'W000' + str(fileInit + i) + '.csv'
        else:
             if fileInit + i  < 100:
                file = 'W00' + str(fileInit + i) + '.csv'
             else:
                file = 'W0' + str(fileInit + i) + '.csv'
        dfi = pd.read_csv(file, skiprows=29,header=None, names=["Wavelength", str(param[i])])
        dfi = dfi[(dfi['Wavelength'] >= xRange[0]) & (dfi['Wavelength'] <= xRange[1])]
        df[str(param[i])] = dfi[str(param[i])] - df['ASE']
    return df

def FastFourier(x ,y):
    N = len(x)
    dx = round(x[1] - x[0],4)
    Fs = 1/dx
    Y = fft(y)
    sF = fftfreq(N, dx)[:N // 2]
    mY = 2.0 / N * np.abs(Y[0:N // 2])
    k1 = math.floor(N/Fs)
    return [sF[:k1], mY[:k1]]

"""
fig = make_subplots()
    fig.add_trace(go.Scatter(
        x=sF,
        y= mY,
        mode="lines",
        line_color='black',
        showlegend=True,
    ))
    fig.show()"""
"""
    plt.plot(sf, 2.0 / N * np.abs(Y[0:N // 2]), 'k-')
    xlim(0, 1)
    ylim(0, 10)
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 11, 2))
    xlabel('Spatial frequency ($nm^{-1}$)', fontdict=font)
    ylabel('Magnitude (A.U.)', fontdict=font)
    plt.tick_params(labelsize=10, width=1)
    auxWidth = 8.9 * cm
    auxHeight = 8 * cm
    figure = plt.gcf()
    figure.set_size_inches(auxWidth, auxHeight)
    plt.savefig("FFT.png", dpi=300, bbox_inches="tight", pad_inches=0.1, transparent=True)
    plt.show()
    """
def WaveletDecomposition(x, y, MW, DL):
    colorLegend = [' black', ' blue', ' blueviolet', ' brown', ' cadetblue', ' chocolate', ' coral',
                   ' cornflowerblue', ' crimson', ' darkblue', ' darkcyan', ' darkmagenta', ' darkorange', ' darkred',
                   ' darkseagreen', ' darkslategray', ' darkviolet', ' deeppink', ' deepskyblue', ' dodgerblue',
                   ' firebrick', ' forestgreen', ' fuchsia', ' gold', ' goldenrod', ' green', ' hotpink', ' indianred',
                   ' indigo', ' orangered', ' purple', ' rebeccapurple', ' red', ' saddlebrown', ' salmon',
                   ' seagreen', ' sienna', ' slateblue', ' steelblue', ' violet', ' yellowgreen', 'aqua', 'aquamarine',
                   'darkgoldenrod', 'darkorchid', 'darkslateblue', 'darkturquoise', 'greenyellow', 'navy',
                   'palevioletred', 'royalblue', 'sandybrown']
    N = len(y)
    L = []

    coeffs = pywt.wavedec(y, MW, mode='symmetric', level=DL, axis=-1)
    cAux = [];
    for i in range(DL + 1):
        L.append(len(coeffs[i]))
        cAux.append(np.zeros(L[i]))
        yr = []
    fig1 = make_subplots()
    for i in range(DL - 1):
        cAux[i] = coeffs[i]
        yr.append(pywt.waverec(cAux, MW))
        cAux[i] = np.zeros(L[i])
        if i == 0:
            fig1.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line_color=colorLegend[-1],
                name = 'signal'
                ))
        if i==0:
            nameLeg = 'a' + str(DL)
        else:
            nameLeg = 'd' + str(DL - i)

        fig1.add_trace(go.Scatter(
            x=x,
            y=yr[i],
            mode="lines",
            line_color=colorLegend[i],
            name=nameLeg
            ))

    [sF, mY] = FastFourier(x, y)
    fig2 = make_subplots()
    for i in range(DL - 1):
        [sFi, mYi] = FastFourierPlot(x, yr[i])
        if i == 0:
            fig2.add_trace(go.Scatter(
                x= sF,
                y= mY,
                mode="lines",
                line_color=colorLegend[-1],
                name='FFT signal'
                ))
        if i==0:
            nameLeg = 'a' + str(DL)
        else:
            nameLeg = 'd' + str(DL - i)

        fig2.add_trace(go.Scatter(
            x=sFi,
            y=mYi,
            line_color=colorLegend[i],
            name=nameLeg
            ))
    return fig1, fig2

    """
    fig1 = make_subplots(DL-1, 1, shared_xaxes=True)
    for i in range(DL-1):
        cAux[i] = coeffs[i]
        yr = pywt.waverec(cAux, MW)
        cAux[i] = np.zeros(L[i])
        if i==0:
            fig1.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines"), row=i + 1, col=1)

        fig1.add_trace(go.Scatter(
            x=x,
            y=yr,
            mode="lines"), row=i + 1, col=1)
    """

def SignalPlot(x,y):
    fig = make_subplots(1)
    fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line_color='black',
            ))
    fig.show()


def SignalSpectrogram(x,y):
    y = np.array(y)
    fig, ax = plt.subplots()
    dx = round(x[1] - x[0],4)
    Lx = len(x)
    Fs = int(1/dx)
    SF, wavelength, Sxx = signal.spectrogram(y, Fs)
    ax.pcolormesh(wavelength, SF, Sxx, shading='gouraud')
    ax.set_ylabel('Spatial frequency (1/nm)')
    ax.set_xlabel('Wavelength (nm)')
    """
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(y,Lx,Fs)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Spatial frequency (1/nm)')
    plt.show()
    """
    return fig

def LaserStability(df, xRange, paramSel):
    color = ['k','b','r','g','c','m','y']
    pl.figure()
    ax = pl.subplot(projection='3d')
    zi = []
    cValue = []
    verts = []
    auxXi =df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])]['Wavelength']
    xi =auxXi.tolist()
    Lx = len(xi)
    NS = len(paramSel)
    for i in range(NS-1,-1,-1):
        ci = [i] * len(xi)
        Lc = len(ci)
        cValue.append(str(paramSel[NS-1-i]))
        zi = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])][str(paramSel[i])].tolist()
        Lz = len(zi)
        ax.plot(xi, ci, zi, color=color[i],linewidth=1)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_yticks(list(range(NS)))
    ax.set_yticklabels(cValue)
    ax.set_zlabel('Output power (dBm)')
    ax.set_xlim(xRange[0], xRange[1])
    ax.set_zlim(-70,-20)
    return




