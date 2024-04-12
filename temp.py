
"""
def DE_PSD(data, freq_params):
    '''
    compute DE and PSD
    --------
    input:  data [n*m]          n electrodes, m time points
            freq_params.stftn     frequency domain sampling rate
            freq_params.fStart    start frequency of each frequency band
            freq_params.fEnd      end frequency of each frequency band
            freq_params.window    window length of each sample point(seconds)
            freq_params.fs        original frequency
    output: psd,DE [n*l*k]        n electrodes, l windows, k frequency bands
    '''
    #initialize the parameters
    STFTN=freq_params['stftn']
    fStart=freq_params['fStart']
    fEnd=freq_params['fEnd']
    fs=freq_params['fs']
    window=freq_params['window']

    WindowPoints=fs*window

    fStartNum=np.zeros([len(fStart)],dtype=int)
    fEndNum=np.zeros([len(fEnd)],dtype=int)
    for i in range(0,len(freq_params['fStart'])):
        fStartNum[i]=int(fStart[i]/fs*STFTN)
        fEndNum[i]=int(fEnd[i]/fs*STFTN)

    #print(fStartNum[0],fEndNum[0])
    n=data.shape[0]
    m=data.shape[1]

    #print(m,n,l)
    psd = np.zeros([n,len(fStart)])
    de = np.zeros([n,len(fStart)])
    #Hanning window
    Hlength=window*fs
    #Hwindow=hanning(Hlength)
    Hwindow= np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength+1)) for n in range(1,Hlength+1)])

    WindowPoints=fs*window
    dataNow=data[0:n]
    for j in range(0,n):
        temp=dataNow[j]
        Hdata=temp*Hwindow
        FFTdata=fft(Hdata,STFTN)
        magFFTdata=abs(FFTdata[0:int(STFTN/2)])
        for p in range(0,len(fStart)):
            E = 0
            #E_log = 0
            for p0 in range(fStartNum[p]-1,fEndNum[p]):
                E=E+magFFTdata[p0]*magFFTdata[p0]
            #    E_log = E_log + log2(magFFTdata(p0)*magFFTdata(p0)+1)
            E = E/(fEndNum[p]-fStartNum[p]+1)
            psd[j][p] = E
            de[j][p] = math.log(100*E,2)
            #de(j,i,p)=log2((1+E)^4)
    
    return psd,de
"""



import os
import numpy as np
import math
import scipy.io as sio
from scipy.fftpack import fft,ifft



def DE_PSD(data, freq_params):
    '''
    compute DE and PSD
    --------
    input:  data [n*m]          n electrodes, m time points
            freq_params.stftn     frequency domain sampling rate
            freq_params.fStart    start frequency of each frequency band
            freq_params.fEnd      end frequency of each frequency band
            freq_params.window    window length of each sample point(seconds)
            freq_params.fs        original frequency
    output: psd,DE [n*l*k]        n electrodes, l windows, k frequency bands
    '''
    #initialize the parameters
    STFTN=freq_params['stftn']
    fStart=freq_params['fStart']
    fEnd=freq_params['fEnd']
    fs=freq_params['fs']
    window=freq_params['window']

    WindowPoints=fs*window

    fStartNum=np.zeros([fStart],dtype=int)
    fEndNum=np.zeros([fEnd],dtype=int)
    for i in range(0,fStart):
        fStartNum[i]=int(fStart[i]/fs*STFTN)
        fEndNum[i]=int(fEnd[i]/fs*STFTN)

    #print(fStartNum[0],fEndNum[0])
    n=data.shape[0]
    m=data.shape[1]

    #print(m,n,l)
    psd = np.zeros([n,fStart])
    de = np.zeros([n,fStart])
    #Hanning window
    Hlength=window*fs
    #Hwindow=hanning(Hlength)
    Hwindow= np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength+1)) for n in range(1,Hlength+1)])

    WindowPoints=fs*window
    dataNow=data[0:n]
    for j in range(0,n):
        temp=dataNow[j]
        Hdata=temp*Hwindow
        FFTdata=fft(Hdata,STFTN)
        magFFTdata=abs(FFTdata[0:int(STFTN/2)])
        for p in range(0,fStart):
            E = 0
            #E_log = 0
            for p0 in range(fStartNum[p]-1,fEndNum[p]):
                E=E+magFFTdata[p0]*magFFTdata[p0]
            #    E_log = E_log + log2(magFFTdata(p0)*magFFTdata(p0)+1)
            E = E/(fEndNum[p]-fStartNum[p]+1)
            psd[j][p] = E
            de[j][p] = math.log(100*E,2)
            #de(j,i,p)=log2((1+E)^4)
    
    return psd,de
