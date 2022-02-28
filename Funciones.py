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
#import pywt

cm = 1/2.54  # centimeters in inches

def ReadFolderPout(fileInit, xRange, param):
    #Read files (only xRange interval)
    x = []; y = []; L = [];
    NOF =len(param) # número de columnas
    for i in range(NOF):
        if fileInit + i  < 10:
             file = 'W00' + str(fileInit + i) + '.CSV'
             #file = 'W000' + str(fileInit + i) + '.csv'
        else:
             if fileInit + i  < 100:
                #file = 'W00' + str(fileInit + i) + '.csv'
                file = 'W00' + str(fileInit + i) + '.CSV'
             else:
                #file = 'W0' + str(fileInit + i) + '.csv'
                file = 'W0' + str(fileInit + i) + '.CSV'

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

def SelectDataFrame(df,xRange, param, indexSel):
    NOF = len(indexSel)
    paramStr = []
    x = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])]['Wavelength'].tolist()
    df1 = pd.DataFrame()
    df1['Wavelength'] = x
    for i in range(NOF):
        k = indexSel[i]
        paramStr.append(str(param[k]))
        yi = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])][paramStr[i]].tolist()
        df1[paramStr[i]] = yi
    return df1

def PointsLinearity(df1, val):
    col_names = df1.columns.values[1:]
    paramStr = col_names.tolist()
    NOF = len(paramStr)
    if val == 'max':
        for i in range(NOF):
            df1['max' + str(i)] = df1.iloc[argrelextrema(df1[paramStr[i]].values, np.greater_equal, order=15)[0]][paramStr[i]]
    elif val == 'min':
        for i in range(NOF):
            df1['min' + str(i)] = df1.iloc[argrelextrema(df1[paramStr[i]].values, np.less_equal, order=15)[0]][paramStr[i]]
    else:
        #falta verificar
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

def PlotInteractiveTx(df1, paramTitle):
    col_names = df1.columns.values[1:]
    paramStr = col_names.tolist()
    NOF = len(paramStr)
    colorLegend =[ ' black', ' blue', ' blueviolet', ' brown', ' cadetblue', ' chocolate', ' coral',
                    ' cornflowerblue', ' crimson', ' darkblue', ' darkcyan', ' darkmagenta', ' darkorange', ' darkred',
                    ' darkseagreen', ' darkslategray', ' darkviolet', ' deeppink', ' deepskyblue', ' dodgerblue',
                    ' firebrick', ' forestgreen', ' fuchsia', ' gold', ' goldenrod', ' green', ' hotpink', ' indianred',
                    ' indigo', ' orangered', ' purple', ' rebeccapurple', ' red', ' saddlebrown', ' salmon',
                    ' seagreen', ' sienna', ' slateblue', ' steelblue', ' violet', ' yellowgreen', 'aqua', 'aquamarine',
                    'darkgoldenrod', 'darkorchid', 'darkslateblue', 'darkturquoise', 'greenyellow', 'navy',
                    'palevioletred', 'royalblue', 'sandybrown']

    A = df1["Wavelength"].tolist()
    fig1 = make_subplots()
    for i in range(NOF):
        B = df1[paramStr[i]]
        fig1.add_trace(go.Scatter(
            x=A,
            y=B,
            legendgroup = 'lgd'+str(i),
            name=paramStr[i],
            mode="lines",
            line_color=colorLegend[i],
            ))
    fig1.update_layout(legend_title_text=paramTitle)
    return fig1

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
    # add val points
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

def ReadFolderStability(fileInit, xRange, yRange, param):
    #Read files (only xRange interval)
    x = []; y = []; L = [];
    NOF =len(param) # número de columnas
    for i in range(0, NOF, 4):
        if fileInit + i  < 10:
             file = 'W00' + str(fileInit + i) + '.CSV'
        else:
             if fileInit + i  < 100:
                file = 'W00' + str(fileInit + i) + '.CSV'
             else:
                file = 'W0' + str(fileInit + i) + '.CSV'
        [xi, yi] = LoadFile(file, 29, xRange, yRange)
        x.append(xi)
        y.append(yi)
        L.append(len(xi))
    return [x,y,L]

def LoadFile(file,jump, xRange, yRange):
    #jump especifica cuantas filas se salta
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
                if auxY < yRange[0]:
                    auxY = yRange[0]
                if auxY > yRange[1]:
                    auxY = yRange[1]
                yi.append(auxY)
    return [xi,yi]

def SelectLaserSignal(x,y,L):
    LL = len(L)
    x1 = np.empty(LL)
    x2 = np.empty(LL)
    ymax = np.empty(LL)
    FWHM = np.empty(LL)
    #Hallar todos y elegir el mayoor pico de potencia
    for i in range(LL):
        xi = np.array(x[i])
        yi = np.array(y[i])
        x1[i], x2[i], ymax[i], FWHM[i] = Calculate_yMax_FWHM(xi, yi)
    kymax = np.argmax(ymax)
    return kymax, ymax[kymax], FWHM[kymax]

def Calculate_yMax_FWHM(x, y):
    kmax = np.argmax(y)
    ymax = y[kmax]
    y3dB = ymax - 3
    d = np.asarray(np.where((y - y3dB) > 0))
    k1 = d[0, 0]
    k2 = d[0, -1]
    FWHM = x[k2] - x[k1]
    return x[k1], x[k2], ymax, FWHM


def PlotLaserFeatures(x,y, xRange, yRange, height, prom, dist):
    fig, ax = plt.subplots()
    ax.set_xlim(xRange)
    ax.set_ylim(yRange)
    # ax.set_xlabel('Longitud de onda (nm)', fontsize=16)
    ax.set_xlabel('Wavelength (nm)', fontsize=16)
    # ax.set_ylabel('Transmisión (dB)', fontsize=16)
    ax.set_ylabel('Output power (dBm)', fontsize=16)
    plt.plot(x, y, color='k', linewidth=0.8)
    x1, x2, ymax, FWHM = Calculate_yMax_FWHM(x,y)
    # FWHM
    #left arrow
    xy1 = (x1,ymax-3)
    xytext1 =(x1-1,ymax-3)
    ax.annotate('', xy=xy1, xycoords='data',
                xytext=xytext1, textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                ec="k",
                                shrinkA=0, shrinkB=0))
    #right arrow
    xy2 = (x2, ymax - 3)
    xytext2 = (x2+1, ymax - 3)
    ax.annotate('', xy=xy2, xycoords='data',
                xytext=xytext2, textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                ec="k",
                                shrinkA=0, shrinkB=0))
    xFWHM = x1+1
    yFWHM = ymax-2
    plt.text(xFWHM, yFWHM, ' FWHM\n' + str(round(FWHM,4)) + 'nm')
    # SMSR
    SMSR, peaksDec, xPeaksDec = CalculateSMSR(x, y, height, prom, dist)
    xprom = (xPeaksDec[0] + xPeaksDec[1]) / 2
    yprom = (peaksDec[0]+peaksDec[1])/2
    xy = ((xPeaksDec[1]+3*xPeaksDec[0])/4, peaksDec[1])
    xytext = ((xPeaksDec[1]+3*xPeaksDec[0])/4, peaksDec[0])
    ax.annotate('', xy=xy, xycoords='data',
                xytext=xytext, textcoords='data',
                arrowprops=dict(arrowstyle="<->",
                                ec="k",
                                shrinkA=0, shrinkB=0))
    plt.text(xprom,yprom,' SMSR\n'+str(SMSR)+'dB')
    fig.tight_layout(pad=0)
    auxWidth = 26 * cm
    auxHeight = 15 * cm
    figure = plt.gcf()
    figure.set_size_inches(auxWidth, auxHeight)
    plt.tight_layout()
    # plt.savefig(r'%d.png' % i, dpi=300, transparent=True, bbox_inches='tight', bbox_extra_artists=(lgd,))
    plt.savefig('Laser.png', dpi=300, transparent=True, bbox_inches='tight')
    return

#def CalculateSMSR(x,y,L, height, thresh, prom):
def CalculateSMSR(x, y, height, prom, dist):
    x = np.array(x)
    y = np.array(y)
    peaksIndex, properties = signal.find_peaks(y, height=height, prominence=prom, distance=dist)
    peaks = y[peaksIndex]
    xPeaks = x[peaksIndex]
    #Sorting ascending
    peaksSorted = np.sort(peaks)
    kSorted = np.argsort(peaksSorted)
    peaksIndexSorted = peaksIndex[kSorted]
    kmax = kSorted[-1]
    if kmax == len(peaks)-1: #si el mayor está al final
        peaksDec = np.array([peaks[-1], peaks[-2]])
        xPeaksDec = np.array([xPeaks[-1], xPeaks[-2]])
    elif kmax==0: #si el mayor está al inicio
        peaksDec = np.array([peaks[0], peaks[1]])
        xPeaksDec = np.array([xPeaks[0], xPeaks[1]])
    else: #el mayor esta intermedio, comparar izq y derecha
        peaksRight = peaks[kmax + 1]
        peaksLeft = peaks[kmax - 1]
        if peaksRight>=peaksLeft:
            peaksDec = np.array([peaks[kmax], peaks[kmax + 1]])
            xPeaksDec = np.array([xPeaks[kmax], xPeaks[kmax+1]])
        else:
            peaksDec = np.array([peaks[kmax], peaks[kmax -1]])
            xPeaksDec = np.array([xPeaks[kmax], xPeaks[kmax - 1]])
    SMSR = abs(peaksDec[0] - peaksDec[1])
    return SMSR, peaksDec, xPeaksDec

def LaserStability3DInteractive(x,y,time):
    NOF = len(time)
    figS = go.Figure()
    for i in range(NOF):
        xi = x[i]
        yi = time[i] * np.ones(len(xi))
        zi = y[i]
        figS.add_trace(go.Scatter3d(x=xi,
                                    y=yi,
                                    z=zi,
                                    mode='lines',
                                    showlegend=False,
                                    marker=dict(
                                        size=12,
                                        opacity=0.8
                                        )))
    figS.update_layout(title="Stability")
    figS.show()
    return

def LaserStability3D(x, z, time,xRange):
    fig = plt.figure()
    ax = pl.subplot(projection='3d')
    cValue = []
    verts = []
    NS = len(time)
    #for i in range(NS-1,-1,-1):
    for i in range(NS):
        yi = [i] * len(x[i])
        #cValue.append(str(paramSel[i]))
        #cValue.append(str(paramSel[NS - 1 - i]))
        zi = z[i]
        Lz = len(zi)
        xp = np.array([x[i]])
        yp = np.array([yi])
        zp = np.array([zi])
        ax.plot_wireframe(xp, yp, zp, color='k',linewidth=1)
        #ax.plot3D(xi, ci, zi, color='k',linewidth=1)
    ax.set_xlabel('Wavelength (nm)',fontsize=14)
    ax.set_ylabel('Time(s)',fontsize=14)
    #plt.xticks(fontsize=12)
    #plt.yticks(fontsize=12)
    ax.set_zticks(list(range(-90,-9,10)),fontsize=20)
    pl.xticks(list(range(1545,1561,5)), ['1545', '1550', '1555', '1560'])
    pl.yticks(list(range(NS)), ['0','','','','','','','','','','80'])
    ax.set_zlabel('Output power (dBm)',fontsize=14)
    ax.set_xlim(xRange[0], xRange[1])
    ax.set_zlim(-80, -10)
    ax.view_init(elev=1., azim=-66)
    pl.show()
    pl.grid
    #Setting figure
    fig.tight_layout(pad=0)
    auxWidth = 26 * cm
    auxHeight = 15 * cm
    figure = pl.gcf()
    figure.set_size_inches(auxWidth, auxHeight)
    pl.tight_layout()
    pl.savefig('Stability.png', dpi=300, transparent=True, bbox_inches='tight')
    return


def PlotInteractiveLin(df1, paramSel, val):
    NOF = len(paramSel)
    col_names = df1.columns.values[1:NOF+1]
    paramStr = col_names.tolist()
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
    for i in range(NOF):
        B = df1[paramStr[i]]
        fig1.add_trace(go.Scatter(
            x=A,
            y=B,
            legendgroup = 'lgd'+str(i),
            name=paramStr[i],
            mode="lines",
            line_color=colorLegend[i],
            ),row=1, col=1)
    #fig1.update_layout(legend_title_text=paramTitle)
    # add val points
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
        AA = [paramSel[i]]*len(BB)
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
             #file = 'W000' + str(fileInit + i) + '.csv'
             file = 'W00' + str(fileInit + i) + '.CSV'
        else:
             if fileInit + i  < 100:
                #file = 'W00' + str(fileInit + i) + '.csv'
                file = 'W00' + str(fileInit + i) + '.CSV'
             else:
                #file = 'W0' + str(fileInit + i) + '.csv'
                file = 'W0' + str(fileInit + i) + '.CSV'
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

def TxRef(x1,y1,xRange):
    maxY1 = 0
    minY1 = min(y1)
    fig, ax = plt.subplots()
    ax.set_xlim(xRange)
    ax.set_ylim([minY1, maxY1])
    #ax.set_xlabel('Longitud de onda (nm)', fontsize=16
    ax.set_xlabel('Wavelength (nm)', fontsize=16)
    #ax.set_ylabel('Transmisión (dB)', fontsize=16)
    ax.set_ylabel('Transmission (dB)', fontsize=16)
    #plt.show()
    plt.plot(x1, y1, linewidth=0.8, color='k')
    fig.tight_layout(pad=0)
    auxWidth = 24 * cm
    auxHeight = 15 * cm
    figure = plt.gcf()
    figure.set_size_inches(auxWidth, auxHeight)
    plt.tight_layout()
    plt.savefig('TxRef', dpi=300,transparent=True, bbox_inches='tight')
    return


def TxParametric(df1, varControl):
    #legend title
    if varControl == 'Temp':
        title = r'$\mathrm{Temp.} (^{\circ}C)$'
    elif varControl == 'Curv':
        title = r'$\mathrm{Curv} (m^{-1})$'
    elif varControl == 'Torsion':
        title = r'$\mathrm{Torsion} (^{\circ})$'
        #title = r'$\mathrm{Temp} (^{\circ})$'
    else:
        title = ''
    col_names = df1.columns.values[2:]
    paramStr = col_names.tolist()
    NOF = len(paramStr)
    #df1 = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])]
    #Useful to see the insertion loss
    #maxY1 = df1[paramStr].max()
    maxY1 = 0
    minY1 = df1[paramStr].min()
    fig, ax = plt.subplots()
    for i in range(NOF):
        plt.plot(df1["Wavelength"], df1[paramStr[i]], linewidth=0.8)
    lgd = plt.legend(paramStr, fontsize=8,
                            title=title,
                            title_fontsize=12,
                            bbox_to_anchor=(1.1, 1),
                            loc='upper right',
                            fancybox=False)
    #SEt xlim,ylim
    xmin = min(df1["Wavelength"].tolist())
    xmax = max(df1["Wavelength"].tolist())
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([min(minY1), maxY1])
    ax.set_xlabel('Wavelength (nm)', fontsize=16)
    # ax.set_xlabel('Longitud de onda (nm)', fontsize=16)
    ax.set_ylabel('Transmission (dB)', fontsize=16)
    # ax.set_ylabel('Transmisión (dB)', fontsize=16)
    #Arrow indicating the tunning direction
    xOrigin = ( xmin + xmax ) / 2
    yOrigin = -1
    ax.annotate('', xy=(xOrigin, yOrigin), xycoords='data',
                xytext=(xOrigin-1, yOrigin), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                ec="k",
                                shrinkA=0, shrinkB=0))
    fig.tight_layout(pad=0)
    auxWidth = 24 * cm
    auxHeight = 15 * cm
    figure = plt.gcf()
    figure.set_size_inches(auxWidth, auxHeight)
    plt.tight_layout()
    #plt.savefig(r'%d.png'%i, dpi=300,transparent=True, bbox_inches='tight',bbox_extra_artists=(lgd,))
    plt.savefig('TxParamTempInc.png', dpi=300, transparent=True, bbox_inches='tight', bbox_extra_artists=(lgd,))
    return
