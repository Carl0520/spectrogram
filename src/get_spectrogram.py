#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:46:43 2018

@author: gaoyi
"""

import os
import pandas as pd

# Math
import numpy as np
import librosa

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display

from glob import glob
from tqdm import tqdm

from PIL import Image as im
import gc

''' make dir '''
dir_lst = ['spectrogram','mel','mfcc']
for d in dir_lst:
    if not os.path.exists('/home/gaoyi/BIIC2/spectrogram3/feature/'+d):
        os.makedirs('/home/gaoyi/BIIC2/spectrogram3/feature/'+d)
        

def save_img(S,root):
        plt.figure()
        librosa.display.specshow(S)
        plt.savefig(root,bbox_inches='tight')
        plt.close()
        gc.collect()

    
#session=1
#path = glob('/home/gaoyi/BIIC2/IEMOCAP/session/Session{}/*/*/*'.format(session))
#filepath = path[10]
#filepath2 = path[10]
#y, sr = librosa.load(filepath, sr=16000, mono=True)  
#S = librosa.feature.melspectrogram(y=y, sr=sr)
#librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
#
#S, phase = librosa.magphase(librosa.stft(y=y))
#librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max))


#%%
for session in range(3,4):    
    for d in dir_lst:
        if not os.path.exists('/home/gaoyi/BIIC2/spectrogram3/feature/'+d+'/Session{}'.format(session)):
            os.makedirs('/home/gaoyi/BIIC2/spectrogram3/feature/'+d+'/Session{}'.format(session))
        
    path = glob('/home/gaoyi/BIIC2/IEMOCAP/session/Session{}/*/*/*'.format(session))

    for filename in tqdm(path):

        basename = os.path.basename(filename)[:-4]
        samples ,sample_rate = librosa.load(filename)
#        spec_S, phase = librosa.magphase(librosa.stft(samples))

#        spec_S = librosa.amplitude_to_db(spec_S, ref=np.max)
#        save_img(spec_S,'/home/gaoyi/BIIC2/spectrogram3/feature/spectrogram/Session{}/'.format(session)+basename+'.png')

        mel_S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
        mel_S = librosa.power_to_db(mel_S, ref=np.max)
        save_img(mel_S,'/home/gaoyi/BIIC2/spectrogram3/feature/mel/Session{}/'.format(session)+basename+'.png')
#
#
#        mfcc = librosa.feature.mfcc(samples,sample_rate ,n_mfcc=13)
#        delta2_mfcc = librosa.feature.delta(mfcc, order=2)                                         
#        save_img(delta2_mfcc,'/home/gaoyi/BIIC2/spectrogram3/feature/mfcc/Session{}/'.format(session)+basename+'.png')
#            


        

        
#        for idx2 in len_mel:
#            dic_mel[idx2[1]]= log_S_speaker[:idx2[0]]
#            log_S_speaker = log_S_speaker[idx2[0]:]
#            
#        for idx3 in len_mfcc:
#            dic_mfcc[idx3[1]]=  mfcc_speaker[:idx3[0]]
#            mfcc_speaker =  mfcc_speaker[idx3[0]:]
#%%

dataframe = pd.read_csv('/home/gaoyi/BIIC2/ComParE2018_AtypicalAffect/lab/ComParE2018_AtypicalAffect.tsv', sep='\t')
happy_name = dataframe.loc[dataframe['emotion']=='happy']
happy_name  = happy_name ['file_name'].tolist()
path =  '/home/gaoyi/BIIC2/ComParE2018_AtypicalAffect/wav/'

dir_lst = ['spectrogram','mel','mfcc']
for d in dir_lst:
    if not os.path.exists('/home/gaoyi/BIIC2/spectrogram3/feature/interspeech_'+d):
        os.makedirs('/home/gaoyi/BIIC2/spectrogram3/feature/interspeech_'+d)
    
for filename in tqdm(happy_name):
    basename = filename
    filename = '/home/gaoyi/BIIC2/ComParE2018_AtypicalAffect/wav/' + basename
    samples ,sample_rate = librosa.load(filename)
#    spec_S, phase = librosa.magphase(librosa.stft(samples))

#    spec_S = librosa.amplitude_to_db(spec_S, ref=np.max)
#    save_img(spec_S,'/home/gaoyi/BIIC2/spectrogram3/feature/interspeech_spectrogram/'+basename+'.png')
#    
#    mel_S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
#    mel_S = librosa.power_to_db(mel_S, ref=np.max)
#    save_img(mel_S,'/home/gaoyi/BIIC2/spectrogram3/feature/interspeech_mel/'+basename+'.png')
    
    mfcc = librosa.feature.mfcc(samples,sample_rate ,n_mfcc=13)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)                                         
    save_img(delta2_mfcc,'/home/gaoyi/BIIC2/spectrogram3/feature/interspeech_mfcc/'+basename+'.png')
#
