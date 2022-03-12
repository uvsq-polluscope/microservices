#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# In[1]:


import pandas as pd
import numpy as np
import math
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import math
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None

import skmob
from skmob.utils import constants, utils, gislib
import datetime
from skmob.privacy import attacks
from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.utils import constants

from skmob.preprocessing import filtering
# from Stop_detection_Hilbert import *
# from skmob_detection import *

#in general: engine = create_engine('dialect+driver://username:password@host:port/database'
engine = create_engine('postgresql://postgres:postgres@localhost:5432/'+"RECORD2")

# In[5]:
# constant lists:

indoor = ['Domicile', 'Bureau','Restaurant','Magasin', 'Gare', 'Indoor']
outdoor = ['Parc', 'Walk', 'Running', 'Vélo']
transport = ['Train', 'Métro', 'Voiture', 'Bus', 'Motorcycle']





# In[6]:


def encode_activity_stop(df):
    
    ### 1 stands for move and 0 stands for stop:
    df['_activity_'] = 1
    for i in range(len(df)):
        #if any(x in ['Domicile', 'Bureau','Restaurant','Magasin', 'Gare', 'indoor'] for x in df.loc[i].activity):
        if df.loc[i].activity in indoor:
            df.loc[i ,'_activity_'] = 0
            
    ### 1 stands for move and 0 stands for stop:
    df['_stops_'] = np.where(df.stop==-1,1,0)  
    
    return df


# In[7]

def splitting_2(df,time=300):
    
    
    df=df.sort_values(by=['time'])
    
    data = df.values
    dfs=[]
    start=0
    start_time=data[0][1]
    lendata = len(data)-1
    for i in range(lendata):
        data1 = data[i]
        data2 = data[i+1]
            
        if ((data2[1]-start_time).seconds>=time):
            dfs.append(data[start:(i+1)])
            start=i+1
            start_time=data2[1]            
    if start<lendata+1:
        dfs.append(data[start:-1])
    
    
    for i in range(len(dfs)):
        dfs[i] = pd.DataFrame(dfs[i], columns=df.columns)

        for x in dfs[i].columns:
            dfs[i][x]=dfs[i][x].astype(df[x].dtypes.name)
    
    return dfs

# In[6]:


def splitting(df,time=600):
    if 'time' in df.columns:
        df=df.sort_values(by=['time'])
    elif 'datetime' in df.columns:
        df=df.sort_values(by=['datetime'])
    
    data = df.values
    dfs=[]
    start=0
    start_time=data[0][0]
    lendata = len(data)-1
    for i in range(lendata):
        data1 = data[i]
        data2 = data[i+1]
            
        if ((data2[0]-start_time).seconds>=time):
            dfs.append(data[start:(i+1)])
            start=i+1
            start_time=data2[0]            
    if start<lendata+1:
        dfs.append(data[start:-1])
      
    for i in range(len(dfs)):
        dfs[i] = pd.DataFrame(dfs[i], columns=df.columns)
        for x in dfs[i].columns:
            dfs[i][x]=dfs[i][x].astype(df[x].dtypes.name)
    
    #return splitted data with activities and stops
    segment=[]
    activity=[]
    stops = []
    i=0
    for df in dfs:
        segment.append(df.iloc[0][0])
        activity.append(set(df.activity.unique()))
        stops.append(df.stops.median())
        i+=1

    res = pd.DataFrame({'segment':segment, 'activity': activity, 'stops':stops})
    return res


def precision_recall(df):
    
    TP = len(df[(df._activity_==0) & (df._stops_==0)])
    
    FP = len(df[(df._activity_==1) & (df._stops_==0)])
    
    FN = len(df[(df._activity_==0) & (df._stops_==1)])
             
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    return print('Precision : %s, recall : %s'%(precision, recall))


def infer_home_work(df):
    
    hour = []
    df_to_array = df.values
    lendf = len(df_to_array)
    for i in range(lendf):
        data_i = df_to_array[i]
        hour.append(dt.strftime(data_i[0], "%H:%M:%S"))
    df['time_h'] = hour
    
    #Infer 'Domicile' from the hour
    
    list_stops = set(df[(df.time_h >= '02:00:00') & (df.time_h <= '05:00:00')].stops.unique())
    for item in list_stops:
        if item !=-1:
            df.loc[df.stops==item, 'stops'] = 'Domicile'
    
    
    ##Infer 'Bureau' as one of the most visisted place (the first or second)
    
    df_groupby = df.groupby('stops').count().sort_values('uid',  ascending=False).reset_index()
    list_stops2 = list(df_groupby.stops)
    
    for item in list_stops2:
        if (item == 'Domicile') | (item == -1):
            pass
        else:
            df.loc[df.stops==item, 'stops'] = 'Bureau'
            break
            
    df.drop(['time_h'], axis=1, inplace=True)
    
    return df


def journals(id):
    data = pd.read_csv('detected_activities/participant_'+str(id)+'.csv')
    data_to_array = data.values
    lendata = len(data_to_array) -1
    liste = [data_to_array[0]]
    for i in range(lendata):
        datai = data_to_array[i]
        datai1 = data_to_array[i+1]

        if datai[3] != datai1[3]:
            liste.append(datai1)
            
    df = pd.DataFrame(data=liste, columns=data.columns) 
    df = df[['participant_virtual_id','timestamp',  'activity_stop']]
    df.rename({'activity_stop':'activity'}, inplace=True)
    
    return df
            
    
def prepare_data_for_plot(df):
    
    df.sort_values('timestamp', inplace=True)
    df['leaving_time'] = df['timestamp'] + dt.timedelta(minutes=10)
    df['activity_stop'] = np.where(df['activity_stop'] == 'transport', 'Transport',\
                            np.where(df['activity_stop'] == 'indoor', 'Indoor',\
                                    np.where(df['activity_stop'] == 'outdoor', 'Outdoor', df['activity_stop'] )))
    return df

def splitting2(df,time=300):
    df.reset_index(inplace=True)
    df['time'] = pd.to_datetime(df['time'] , format='%Y-%b-%d %H:%M:%S')
    
    df=df.sort_values(by=['time'])
    
    data = df.values
    dfs=[]
    start=0
    start_time=data[0][0]
    lendata = len(data)-1
    for i in range(lendata):
        data1 = data[i]
        data2 = data[i+1]
            
        if ((data2[0]-start_time).seconds>=time):
            dfs.append(data[start:(i+1)])
            start=i+1
            start_time=data2[0]            
    if start<lendata+1:
        dfs.append(data[start:-1])
      
    for i in range(len(dfs)):
        dfs[i] = pd.DataFrame(dfs[i], columns=df.columns).set_index('time')
        
        for x in dfs[i].columns:
            dfs[i][x]=dfs[i][x].astype(df[x].dtypes.name)
    
    return dfs