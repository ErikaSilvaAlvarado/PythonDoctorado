#Uses python3
import csv
import pandas as pd
import numpy as np
#import Matrix as mat
import matplotlib.pyplot as plt
from pylab import *
from scipy.fft import fft, ifft, fftfreq
from sklearn.linear_model import LinearRegression
pd.options.plotting.backend = "plotly"
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.signal import argrelextrema

font = {'size': 12,
        'stretch':'ultra-condensed',
        'weight': 'ultralight'
        }

cm = 1/2.54  # centimeters in inches

def ReadFolderTx(df, fileNumber, param, xRange):
    x = []; y = []
    NOF =len(fileNumber) # número de columnas
    for i in range(NOF):
        if fileNumber[0] + i  < 10:
             file = 'W000' + str(fileNumber[0] + i) + '.csv'
        else:
             if fileNumber[0] + i  < 100:
                file = 'W00' + str(fileNumber[0] + i) + '.csv'
             else:
                file = 'W0' + str(fileNumber[0] + i) + '.csv'
        dfi = pd.read_csv(file, skiprows=29,header=None, names=["Wavelength", str(param[i])])
        dfi = dfi[(dfi['Wavelength'] >= xRange[0]) & (dfi['Wavelength'] <= xRange[1])]
        df[str(param[i])] = dfi[str(param[i])] - df['ASE']
    return df


def ReadFolderPout(fileName, param):
    x = []; y = [];
    NOF =len(param) # número de columnas
    for i in range(NOF):
        if fileName[0] + i  < 10:
             file = 'W000' + str(fileName[0] + i) + '.csv'
        else:
             if fileName[0] + i  < 100:
                file = 'W00' + str(fileName[0] + i) + '.csv'
             else:
                file = 'W0' + str(fileName[0] + i) + '.csv'
        [xi, yi] = LoadFile(file, 29)
        if i == 0:
            df = pd.DataFrame(list(zip(xi, yi)), columns=['Wavelength'+str(i), str(param[i])])
        else:
            df['Wavelength'+str(i)] = xi
            df[str(param[i])] = yi
    return df

def ReadFolderPout(fileName, param):
    x = []; y = [];
    NOF =len(param) # número de columnas
    for i in range(NOF):
        if fileName[0] + i  < 10:
             file = 'W000' + str(fileName[0] + i) + '.csv'
        else:
             if fileName[0] + i  < 100:
                file = 'W00' + str(fileName[0] + i) + '.csv'
             else:
                file = 'W0' + str(fileName[0] + i) + '.csv'
        [xi, yi] = LoadFile(file, 29)
        if i == 0:
            df = pd.DataFrame(list(zip(xi, yi)), columns=['Wavelength'+str(i), str(param[i])])
        else:
            df['Wavelength'+str(i)] = xi
            df[str(param[i])] = yi
    return df

def ReadFolder(df, file_init, file_end,param):
    NOF = file_end - file_init + 1; x = []; y = [] #NOF: número de columnas
    for i in range(NOF):
        if file_init + i  < 10:
             file = 'W000' + str(file_init + i) + '.csv'
        else:
             if file_init + i  < 100:
                file = 'W00' + str(file_init + i) + '.csv'
             else:
                file = 'W0' + str(file_init + i) + '.csv'
        [xi, yi]= LoadFile(file,29) #Lista con los valores de wavelength y potencia en dBm
        df[str(param[i])] = yi - df['ASE']
    return df



def LoadFile(file,jump):
    with open(file, newline='') as file:
        reader = csv.reader(file, delimiter =',')

        for k in range(jump):
            next(reader)

        xi = []; yi = []

        for row in reader:
                xi.append(row[0])
                yi.append(row[1])
        xi = [float(i) for i in xi]
        yi = [float(i) for i in yi]

    return [xi,yi]



def DownSample(x,m):
    xDown = []
    i = 0
    while i <= len(x):
        if (i % m )==0:
             xDown.append(x[i])
        i = i+1
    return(xDown)

def Coef (X,Y,M):
    X = np.array(X)
    Y = np.array(Y)
    Z = np.zeros((M, M))
    B=np.zeros((M,1))
    for i in range (M):
        for j in range (M):
            Z[i,j]=np.sum(X**(i+j))
        aux=np.multiply(X**i, Y)
        B[i]=np.sum(aux)
    #a = mat.Multi(mat.Inverse(Z),B)
    pinvZ = mat.PseudoInverseMat(Z)
    a=mat.Multi(pinvZ,B)
    St=B[0]
    return (a)

def Error(X,Y,a):
    X = np.array(X)
    Y = np.array(Y)
    Fun = np.ones(X.shape) * a[len(a) - 1]
    for i in range(len(a) - 2, -1, -1):
        Fun=Fun* X + a[i]
    Sr= sum( (Y-Fun)**2)
    K=sum(Y)/len(Y)
    St=sum( (Y-K)**2 )
    r2=(St-Sr)/St
    return ([Sr,St,r2])

def Graphic(xmin, xmax, a, M,NS):
    # rango de x para graficar NS puntos
    xx = np.linspace(xmin, xmax, NS);
    # halla el y por el interpolacion
    yy = np.ones(xx.shape) * a[len(a) - 1]
    for i in range(M - 2, -1, -1):
        yy = yy * xx + a[i]
        # yy=sol[0]+sol[1]*xx+sol[2]*xx**2
    return ([xx,yy])

def PlotInteractive(df1, paramStr, paramTitle, val):
    colorLegend =['aqua', ' black', ' blue', ' blueviolet', ' brown', ' cadetblue', ' chocolate', ' coral',
                    ' cornflowerblue', ' crimson', ' darkblue', ' darkcyan', ' darkmagenta', ' darkorange', ' darkred',
                    ' darkseagreen', ' darkslategray', ' darkviolet', ' deeppink', ' deepskyblue', ' dodgerblue',
                    ' firebrick', ' forestgreen', ' fuchsia', ' gold', ' goldenrod', ' green', ' hotpink', ' indianred',
                    ' indigo', ' orangered', ' purple', ' rebeccapurple', ' red', ' saddlebrown', ' salmon',
                    ' seagreen', ' sienna', ' slateblue', ' steelblue', ' violet', ' yellowgreen', 'aqua', 'aquamarine',
                    'darkgoldenrod', 'darkorchid', 'darkslateblue', 'darkturquoise', 'greenyellow', 'navy',
                    'palevioletred', 'royalblue', 'sandybrown', 'turquoise']
    #df1 = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])]
    #col_names = df1.columns.values[2:]
    #paramStr = col_names.tolist()
    param = [int(x) for x in paramStr]
    A = df1["Wavelength"].tolist()
    fig1 = make_subplots(1,2)
    for i in range(len(paramStr)):
        B = df1[paramStr[i]]
        fig1.add_trace(go.Scatter(
            x=A,
            y=B,
            legendgroup = 'lgd'+paramStr[i],
            name=paramStr[i],
            mode="lines",
            line_color=colorLegend[i]
            ),row=1, col=1)

    for i in range(len(paramStr)):
        A1 = df1[~pd.isnull(df1[val + paramStr[i]])]['Wavelength'].tolist()
        B1 = df1[~pd.isnull(df1[val + paramStr[i]])][paramStr[i]].tolist()
        fig1.add_trace(go.Scatter(
            x=A1,
            y=B1,
            legendgroup = 'lgd'+paramStr[i],
            name =paramStr[i],
            mode ="markers",
            marker_color = colorLegend[i],
            showlegend=False
            ),row =1, col =1)
    for i in range(len(paramStr)):
        BB = df1[~pd.isnull(df1[val + paramStr[i]])]['Wavelength'].tolist()
        AA = [param[i]]*len(BB)
        fig1.add_trace(go.Scatter(
            x= AA,
            y=BB,
            legendgroup ='lgd' + paramStr[i],
            name =paramStr[i],
            mode ="markers",
            marker_color = colorLegend[i],
            showlegend=False
            ),row=1, col=2)
    fig1.update_layout(title=paramTitle)
    fig1.show()
    return

def PointsLinearity(df, xRange, paramStr, val):
        df1 = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])]
        if val == 'max':
            for i in range(len(paramStr)):
                df1['max' + paramStr[i]] = df1.iloc[argrelextrema(df1[paramStr[i]].values, np.greater_equal, order=15)[0]][paramStr[i]]

            maxY1 = df1[paramStr].max()
            kmax = df1[paramStr].idxmax()
            maxX1 = df1["Wavelength"].loc[kmax]
            maxX1 = maxX1.tolist()
            valX1 = maxX1
            valY1 = maxY1
        elif val == 'min':
            localMin = pd.DataFrame()
            for i in range(len(paramStr)):
                df1['min' + paramStr[i]] = df1.iloc[argrelextrema(df1[paramStr[i]].values, np.less_equal, order=15)[0]][paramStr[i]]

            minY1 = df1[paramStr].min()
            kmin = df1[paramStr].idxmin()
            minX1 = df1["Wavelength"].loc[kmin]
            minX1 = minX1.tolist()
            valX1 = minX1
            valY1 = minY1
        else:
            valY1 = df1[(df1[paramStr] >= val)][paramStr]
            kval = df1[(df1[paramStr] >= val)][paramStr].idxmin()
            valX1 = df1["Wavelength"].loc[kval].tolist()

        #return [valX1, valY1]
        return df1


def LinearSel(df,xRange, val, paramStr,paramTitle):
    eps = 0.05
    #col_names = df.columns.values[2:]
    #paramStr = col_names.tolist()
    #paramStr =  paramStr[0:32] #usar para 1545-1554 min
    #paramStr = paramStr[7:16]  # usar para 1545-1554 min
    #paramStr = paramStr[10:20]  # usar para 1555-1560 val= -13
    #paramStr = paramStr[0:12]  # usar para 1564-1571 val = -12
    #paramStr = paramStr[0:16]  # usar para 1545-1554 min
    #paramStr = paramStr[7:]  # usar para 1550-1553 max
    param = [int(x) for x in paramStr]
    fig, axs = plt.subplots(1, 2)
    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].tick_params(axis='y', labelsize=14)
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].tick_params(axis='y', labelsize=14)
    df1 = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])]

    abcisa = np.array(param).reshape((-1, 1))
    ordenada = np.array(valX1)
    model = LinearRegression().fit(abcisa, ordenada)
    slope_pm = 1000* model.coef_
    r_sq = model.score(abcisa, ordenada)
    y_pred = model.intercept_ + model.coef_ * abcisa
    fig.show()
    NS = len(paramStr)

    for i in range(NS):
        axs[0].plot(df1["Wavelength"], df1[paramStr[i]], linewidth=0.8)
        axs[1].scatter(param[i], valX1[i], marker='d')
    #axs[1].plot(param, y_pred.tolist(), color='blue')
    axs[0].set_xlim(xRange)

    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].tick_params(axis='y', labelsize=14)
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].tick_params(axis='y', labelsize=14)
    """
    listXticks0 = np.linspace(round(xRange[0],0), round(xRange[1],0), 5)
    axs[0].set_xticks(listXticks0.tolist())
    vals = axs[1].get_xticks()
    axs[1].set_xticklabels(['{:,d}'.format(x) for x in vals])
    """
    #axs[0].set_ylim(min(minY1) * 1.05, max(maxY1) * 0.95)
    #axs[0].set_xlabel('Longitud de onda (nm)', fontsize=16)
    axs[0].set_xlabel('Wavelength (nm)', fontsize=16)
    axs[0].set_ylabel('Transmission (dB)', fontsize=16)
    #axs[0].set_ylabel('Transmisión (dB)', fontsize=16)

    lgd = axs[0].legend(paramStr, fontsize=8,
                        title= paramTitle,
                        title_fontsize = 12,
                        bbox_to_anchor=(1.1, 1),
                        loc='upper right',
                        fancybox=False)
    xLoc =(xRange[0] + xRange[1]) / 2
    if val == 'max':
        yLoc = min(maxY1)-0.1
    elif val == 'min':
        yLoc = min(minY1)
    else:
        xLoc = min(valX1)
        yLoc =val
        val = str(val)

    if valX1[len(valX1)-1] - valX1[len(valX1)-2] > 0:
        axs[0].text(xLoc, yLoc, r'$\rightarrow$')
    else:
        axs[0].text(xLoc, yLoc, r'$\leftarrow$')

    #listYticks = np.linspace(round(min(minX1), 0), round(max(maxX1), 0), 5)
    #axs[1].set_yticks(listYticks)


    listYticks = axs[1].get_yticks()
    #xs[1].set_yticklabels(['{:,.1f}'.format(x) for x in listYticks])
    axs[1].set_xlabel(paramTitle, fontsize=16)
    axs[1].set_ylabel('Wavelength (nm)', fontsize=16)
    #axs[1].set_ylabel('Longitud de onda (nm)', fontsize=16)
    

    """"
    N = 1  # linear regression order
    M = N + 1
    a = Coef(param,valX1, M)
    aNpm = a[N] * 1000
    print(a[N])
    [Sr, St, r2] = Error(param, valX1, a)
    [xx, yy] = Graphic(param[0], param[NS - 1], a, M, NS)
    plt.plot(xx.tolist(), yy.tolist(), color='blue')
    """
    axs[1].set_xlim(param[0], param[NS - 1])
    #listXticks1 = np.array(param[0], param[NS - 1], 6)
    #axs[1].set_xticks(listXticks1)

    xLoc = (param[NS - 1] + param[0]) /3
    xLoc = 136
    yLoc = listYticks[len(listYticks) - 3]
    axs[1].text(xLoc, yLoc, r'$\frac{d\lambda}{dT}$= %.4f$pm/{^{\circ}C}$' % slope_pm, fontsize=10)
    fig.tight_layout(pad=0)
    auxWidth = 16 * cm
    auxHeight = 16 * cm
    figure = plt.gcf()
    figure.set_size_inches(auxWidth, auxHeight)
    plt.tight_layout()

    plt.savefig(str(xRange[0]) + "-" + str(xRange[1]) + val, dpi=300,transparent=True, bbox_inches='tight',bbox_extra_artists=(lgd,))
    return

def Transmission(df,xRange,paramTitle):
    col_names = df.columns.values[2:]
    paramStr = col_names.tolist()
    df1 = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])]
    maxY1 = df1[paramStr].max()
    minY1 = df1[paramStr].min()
    fig, ax = plt.subplots()
    ax.set_xlim(xRange)
    ax.set_ylim([min(minY1), max(maxY1)])
    #ax.set_xlabel('Longitud de onda (nm)', fontsize=16)
    ax.set_xlabel('Wavelength (nm)', fontsize=16)
    #ax.set_ylabel('Transmisión (dB)', fontsize=16)
    ax.set_ylabel('Transmission (dB)', fontsize=16)
    plt.show()
    plt.plot(df1["Wavelength"], df1[paramStr[0]], color='k', linewidth=0.8)
    fig.tight_layout(pad=0)
    auxWidth = 26 * cm
    auxHeight = 15 * cm
    figure = plt.gcf()
    figure.set_size_inches(auxWidth, auxHeight)
    plt.tight_layout()
    #plt.savefig(r'%d.png' % i, dpi=300, transparent=True, bbox_inches='tight', bbox_extra_artists=(lgd,))
    plt.savefig('Tx.png', dpi=300, transparent=True, bbox_inches='tight')
    """
        for i in range(len(paramStr)):
        plt.plot(df1["Wavelength"], df1[paramStr[i]], linewidth=0.8)

        lgd = plt.legend(paramStr[0:i+1], fontsize=8,
                            title= paramTitle,
                            title_fontsize=12,
                            bbox_to_anchor=(1.1, 1),
                            loc='upper right',
                            fancybox=False)
        xLoc = (df1["Wavelength"].min() + df1["Wavelength"].max()/2)
        yLoc = min(minY1)

        kmin = df1[paramStr].idxmin()
        minX1 = df1["Wavelength"].loc[kmin]
        minX1 = minX1.tolist()

        if minX1[2] - minX1[1] > 0:
            plt.text(xLoc, yLoc, r'$\rightarrow$')
        else:
            plt.text(xLoc, yLoc, r'$\leftarrow$')
    """
    return

def FastFourierPlot(df,xRange):
    df1 = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])]
    col_names = df1.columns.values[2:]
    paramStr = col_names.tolist()
    x = df['Wavelength'].tolist()
    N = len(x)
    y0 = df[paramStr[0]].tolist()
    dwav = x[1] - x[0]
    Y0 = fft(y0)
    sf = fftfreq(N, dwav)[:N // 2]
    plt.plot(sf, 2.0 / N * np.abs(Y0[0:N // 2]), 'k-')
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

    def Laser(df, xRange, paramTitle):
        col_names = df.columns.values[2:]
        paramStr = col_names.tolist()
        df1 = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])]
        maxY1 = df1[paramStr].max()
        minY1 = df1[paramStr].min()
        fig, ax = plt.subplots()
        ax.set_xlim(xRange)
        ax.set_ylim([min(minY1), max(maxY1)])
        # ax.set_xlabel('Longitud de onda (nm)', fontsize=16)
        ax.set_xlabel('Wavelength (nm)', fontsize=16)
        # ax.set_ylabel('Transmisión (dB)', fontsize=16)
        ax.set_ylabel('Output power (dBm)', fontsize=16)
        plt.show()
        plt.plot(df1["Wavelength"], df1[paramStr[0]], color='k', linewidth=0.8)

        fig.tight_layout(pad=0)
        auxWidth = 26 * cm
        auxHeight = 15 * cm
        figure = plt.gcf()
        figure.set_size_inches(auxWidth, auxHeight)
        plt.tight_layout()
        # plt.savefig(r'%d.png' % i, dpi=300, transparent=True, bbox_inches='tight', bbox_extra_artists=(lgd,))
        plt.savefig('Laser.png', dpi=300, transparent=True, bbox_inches='tight')

        fig, ax = plt.subplots()
        for i in range(len(paramStr)):
            plt.plot(df1["Wavelength"], df1[paramStr[i]], linewidth=0.8)

        lgd = plt.legend(paramStr, fontsize=8,
                                title= paramTitle,
                                title_fontsize=12,
                                bbox_to_anchor=(1.1, 1),
                                loc='upper right',
                                fancybox=False)
        xLoc = (df1["Wavelength"].min() + df1["Wavelength"].max()/2)
        yLoc = max(maxY1)

        kmax = df1[paramStr].idxmax()
        maxX1 = df1["Wavelength"].loc[kmax]

        if maxX1[2] - maxX1[1] > 0:
            plt.text(xLoc, yLoc, r'$\rightarrow$')
        else:
            lt.text(xLoc, yLoc, r'$\leftarrow$')
        fig.tight_layout(pad=0)
        auxWidth = 26 * cm
        auxHeight = 15 * cm
        figure = plt.gcf()
        figure.set_size_inches(auxWidth, auxHeight)
        plt.tight_layout()
        # plt.savefig(r'%d.png' % i, dpi=300, transparent=True, bbox_inches='tight', bbox_extra_artists=(lgd,))
        plt.savefig('LaserParam.png', dpi=300, transparent=True, bbox_inches='tight')
        return