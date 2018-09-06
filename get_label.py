#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 18:14:40 2018

@author: gaoyi
"""
import pandas as pd 
import os 
os.chdir('/home/gaoyi/BIIC2/spectrogram3/label')
labels = pd.read_csv('/home/gaoyi/BIIC2/IEMOCAP/IEMOCAP_Org_Emo.csv')
def get_session_label(session):
    f = open('0{}_labels.txt'.format(session),'a')
    for i in labels['Name_String']:
        if 'Ses0'+str(session) in i:
            label = labels[labels['Name_String'] == i]['Emotion'].values[0]
            if label in ['sad']:
                f.write(i[:-4]+',0\n')
            elif label in ['hap']:
                f.write(i[:-4]+',1\n')
            elif label in ['neu']:
                f.write(i[:-4]+',2\n')
            elif label in ['ang']:
                f.write(i[:-4]+',3\n')   
    
    f.close()
    
for s in range(1,6):
    get_session_label(s)
    
def get_happy_label():
    dataframe = pd.read_csv('/home/gaoyi/BIIC2/ComParE2018_AtypicalAffect/lab/ComParE2018_AtypicalAffect.tsv', sep='\t')
    happy_name = dataframe.loc[dataframe['emotion']=='happy']
    happy_name  = happy_name ['file_name'].tolist()
    f = open('happy_labels.txt','a')
    for i in happy_name:
        f.write(i[:-4]+',1\n')
    f.close()
        
get_happy_label()
    
    


