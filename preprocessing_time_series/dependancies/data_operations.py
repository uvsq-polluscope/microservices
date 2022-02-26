#!/usr/bin/env python
#imports
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

import pandas as pd
import numpy as np
import pandasql as ps
from datetime import timedelta

import matplotlib.pyplot as plt
import sys

from detecta import detect_cusum

import plotly
import plotly.graph_objects as go
from math import sin
from scipy.signal import find_peaks

import json
from math import isnan
import csv


#this function calculates the precision and recall when using cumsum
def precision_recall(df):
    TP = len(df[(df.activity_time.notnull()) & (df.cusum_time.notnull())].drop_duplicates('activity_time').drop_duplicates('cusum_time'))
    FN = len(df[df.cusum_time.isnull()]) + len(df[(df.duplicated('cusum_time')) & (df.cusum_time.notnull())])
    FP = len(df[df.activity_time.isnull()]) + len(df[(df.duplicated('activity_time')) & (df.activity_time.notnull())])
    if TP>0:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
    else:
        precision=0
        recall=0
    return precision, recall



#Adapting the Cumsum algorithm to our needs
def cusum_(df, variable, threshold, drift, show):
    df_variable = df.dropna(subset= [variable])
    df_variable.index = range(len(df_variable))
    df_variable['truth'] = np.zeros(len(df_variable))
    
    for i in range(len(df_variable)):
        if (df_variable.activity.iloc[i] != df_variable.activity.shift(1).iloc[i]) |        (df_variable.event.iloc[i] != df_variable.event.shift(1).iloc[i] ):
            df_variable.truth.iloc[i] = 1
            
    prec=[]
    rec=[]
    
    for i in range(len(threshold)):
        ta, tai, taf, amp = detect_cusum(df_variable[variable], threshold[i], drift[i], True, show)
        #
        df_variable['CUSUM'] = np.zeros(len(df_variable))
        for i in ta:
            df_variable.loc[i,"CUSUM"] = 1
        #
        ##Create a buffering interval of 5 minutes
        cusum = df_variable[df_variable.CUSUM == 1][["time", "activity", "event", "CUSUM"]]
        truth = df_variable[df_variable.truth == 1][["time", "activity", "event", "truth"]]
        truth["start_time"] = truth.time - timedelta(minutes = 5)
        truth["end_time"] = truth.time + timedelta(minutes = 5)
        sqlcode = '''
        select cusum.time as cusum_time,
        truth.time as activity_time,
        truth.activity,
        truth.event

        from truth
        left join cusum  
        on cusum.time between truth.start_time and truth.end_time

        UNION

        select cusum.time as cusum_time,
        truth.time as activity_time,
        truth.activity,
        truth.event

        from cusum
        left join truth 
        on cusum.time between truth.start_time and truth.end_time
        '''
        newdf = ps.sqldf(sqlcode,locals())
        #
        precision, recall = precision_recall(newdf)
        #
        prec.append(precision)
        rec.append(recall)
        
        #results = pd.DataFrame({'Threshold': threshold, 'Drift': drift, 'Precision': prec, 'Recall': rec})
    
    return df_variable, newdf, prec, rec



def Convert(lst): 
    return [ -i for i in lst ]



#Peaks detection by the help of "find_peaks" function we don't use it (here we should specify a height or a threshold)
def detect_peaks(df,columnName,show=True,height=-1):    
    '''This Function aims to detect the peaks in the data you should provide it a dataframe and the column name you are aiming
    to detect peaks in it. This function will return the indices of these peakes in the dataframe.'''
    time_series=df[columnName]
    time_series_conv=Convert(time_series)
    if height!=-1:
        indices_positive = find_peaks(time_series,height)[0]
        indices_negative=find_peaks(time_series_conv,height)[0]
    else:
        indices_positive = find_peaks(time_series,df[columnName].max()/4)[0]
        indices_negative=find_peaks(time_series_conv,df[columnName].max()/4)[0]  
        
    indices=np.append(indices_positive,indices_negative)
    
    if show:
        plotly.offline.init_notebook_mode(connected=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=time_series,
            mode='lines+markers',
            name='Original Plot'
        ))

        fig.add_trace(go.Scatter(
            x=indices,
            y=[time_series[j] for j in indices],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='cross'
            ),
            name='Detected Peaks'
        ))

        fig.show()
    return indices



#This function returns a new dataframe execluding the indices that are supposed to be peaks (also not used)
def remove_peaks(df,columnName,height=-1):
    '''This Function aims to remove the peaks in the data you should provide it a dataframe and the column name you are aiming
    to remove peaks in it. This function will call the detect_peaks function and will find the indices of the peakes in the 
    dataframe. Then it will return a new dataframe without peaks.'''
    indices=detect_peaks(df,columnName,height=height,show=False)
    new_df=df.drop(indices,axis=0)
    return new_df



def add_verification_column(df,values=['activity_time', 'Humidity', 'Temperature', 'BC']):
    '''This function will add a verification column for the provided dataframe'''
    res=df
    res.sort_values(values, inplace=True)
    
    ### Create a new column which takes only zeros
    res['verification'] = np.zeros(len(res))
    res['verification'] = pd.to_datetime(res['verification'], format='%Y-%m-%d %H:%M:%S')
    #### Define the format of the time columns
    for i in res.drop(['activity', 'event', 'verification'], axis=1).columns:
        res[i] = pd.to_datetime(res[i], format='%Y-%m-%d %H:%M:%S')    
    return res



def compute_verification_column(df,values):
    '''This function will call the add_verification_column function to intailize a verification column and then it will compute
    the verification for each row by inserting the nearest detected change time (to the activity time) into verification field'''
    res=add_verification_column(df,values)
    for i in range(len(res)):
        res['verification'].iloc[i] = res[res.drop(['activity', 'event','verification'], axis=1).columns].iloc[i].mean()

    return res



def merge_close_changes(res,delay=5):
    for i in range(len(res)):
        for j in res.drop(['verification','activity','activity_time', 'event'], axis=1).columns:            
            if (res[j].iloc[i] - res.shift(1)[j].iloc[i] <= timedelta(minutes=delay)):
                res['verification'].iloc[i] = res['verification'].shift(1).iloc[i]
                if res[j].shift(1).isnull().iloc[i]:
                    res[j].iloc[i-1] = res[j].iloc[i] 
    res.drop_duplicates(['verification'], inplace=True)
    return res



def get_true_detected_changes(df,attribute):
    '''This function will return all the change points detected when an activity happened'''
    df_new = pd.DataFrame(df[(df[attribute].notnull())& (df.activity_time.notnull())])
    return df_new



def get_false_detected_changes(df,attribute):
    '''This function will return all the time of change points detected but no activity happened'''
    df_new = pd.DataFrame(df[(df[attribute].notnull())& (df.activity_time.isnull())])
    return df_new



def get_all_detected_changes(df,attribute):
    '''This function will return all the time of change points detected '''
    df_new = pd.DataFrame(df[(df[attribute].notnull())])
    return df_new



def scatter_plot(df,x,y):
    '''This function will plot df[y] values along with df[x] as a scatter'''
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(df[x], df[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y+'Values')
    plt.show()



def get_concentration_along_changes(original_df,df,attributes,values=[],changes="All"):
    '''This function will return all the specified values (if not specified it will return all columns of the original data)
    as a dataframe were the change occurs in the attributes provided and the returned dataframe contains the measurements of
    the other attributes. At the end the function will return a new dataframe containing only the measurements when a change
    occured.'''
    df_attribute=pd.DataFrame()
    for attribute in attributes:
        if changes=="All":        
            df_attribute=pd.concat([df_attribute,get_all_detected_changes(df,attribute)])
        elif changes==True:
            df_attribute=pd.concat([df_attribute,get_true_detected_changes(df,attribute)])
        else:
            df_attribute=pd.concat([df_attribute,get_false_detected_changes(df,attribute)])
    if not values:
        df_data = pd.DataFrame(columns=original_df.columns)
    else:
        df_data = pd.DataFrame(columns=values)
    for attribute in attributes:
        for i in original_df["time"]:
            for j in df_attribute[attribute]:
                if i==j: 
                    if not values:
                        df_data=pd.concat([df_data,original_df.loc[original_df["time"] == i]])
                    else:
                        df_data=pd.concat([df_data,original_df.loc[original_df["time"] == i][values]])
    df_data.drop_duplicates(inplace=True)
    return df_data



def get_range_concentration_along_changes(original_df,df,attributes,values=[],changes="All",minutes=1):
    '''This function will return all the specified values (if not specified it will return all columns of the original data)
    as a dataframe were the change occurs in the attributes provided and the returned dataframe contains the measurements of
    the other attributes. At the end the function will return a new dataframe containing only the measurements when a change
    occured.'''
    df_attribute=pd.DataFrame()
    for attribute in attributes:
        if changes=="All":        
            df_attribute=pd.concat([df_attribute,get_all_detected_changes(df,attribute)])
        elif changes==True:
            df_attribute=pd.concat([df_attribute,get_true_detected_changes(df,attribute)])
        else:
            df_attribute=pd.concat([df_attribute,get_false_detected_changes(df,attribute)])
    if not values:
        df_data = pd.DataFrame(columns=original_df.columns)
    else:
        df_data = pd.DataFrame(columns=values)
    for attribute in attributes:
        for i in original_df["time"]:
            for j in df_attribute[attribute]:
                if i==j: 
                    if not values:
                        df_data=pd.concat([df_data,original_df.loc[(original_df["time"] >= i-timedelta(minutes = minutes))&(original_df["time"] <= i+timedelta(minutes = minutes))]])
                    else:
                        df_data=pd.concat([df_data,original_df.loc[(original_df["time"] >= i-timedelta(minutes = minutes))&(original_df["time"] <= i+timedelta(minutes = minutes))][values]])
    df_data.drop_duplicates(inplace=True)
    return df_data



#Detects peaks over a given threshold
def peaks_detection(df,columnName,threshold,show=False):
    indices=[]
    for i in range(1,len(df[columnName])-1):
        if abs(df[columnName].iloc[i+1]-df[columnName].iloc[i])>=threshold and (df["time"].iloc[i+1]-df["time"].iloc[i]).seconds==60:
            indices.append(i)
            indices.append(i+1)
    if show==True:
        plotly.offline.init_notebook_mode(connected=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df[columnName],
            mode='lines+markers',
            name='Original Plot'
        ))

        fig.add_trace(go.Scatter(
            x=indices,
            y=[df[columnName][j] for j in indices],
            mode='markers',
            marker=dict(
            size=8,
            color='red',
            symbol='cross'
            ),
            name='Detected Peaks'
        ))

        fig.show()
    return indices



#Remove the indices detected as peaks from the dataframe using the above function
def peakes_removing(df,columnName,threshold,show=False):
    indices=peaks_detection(df,columnName,threshold,show)
    df_new=df.copy()
    df_new.drop(indices,inplace=True)
    return df_new



def split_dataframe(df,columnName="BC",time=3600,show=False):
    '''
    This function is used to split the df into several dfs when the difference between 2 consecutive records is more than a specified time
    '''
    indices=[]
    for i in range(1,len(df)-1):
        if (df["time"].iloc[i+1]-df["time"].iloc[i]).seconds>time:
            indices.append(i+1)
    if show==True:
        plotly.offline.init_notebook_mode(connected=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df[columnName],
            mode='lines+markers',
            name='Original Plot'
        ))

        fig.add_trace(go.Scatter(
            x=indices,
            y=[df[columnName][j] for j in indices],
            mode='markers',
            marker=dict(
            size=8,
            color='red',
            symbol='cross'
            ),
            name='Detected Peaks'
        ))

        fig.show()
    dfs=[]
    if len(indices)==0:
        dfs.append(df)
    else:        
        for i in range(len(indices)-1):
            if len(dfs)==0:
                dfs.append(df[0:indices[i]])
                dfs.append(df[indices[i]:indices[i+1]])
            else:
                dfs.append(df[indices[i]:indices[i+1]])                  
        if len(indices)==1:
            dfs.append(df[0:indices[0]])
            dfs.append(df[indices[0]:])
        else:
            dfs.append(df[indices[i+1]:])
        
    return dfs



def mean_peaks_removing(df,columnName,window_size=10,show=False):
    indices=mean_peaks_detection(df,columnName,window_size,show)[0]
    df_new=df.copy()
    df_new.drop(indices,inplace=True)
    return df_new



def mean_peaks_replacing(df,columnName,window_size=10,show=False):
    peaks=mean_peaks_detection(df,columnName,window_size,show)
    indices=peaks[0]
    avg_map=peaks[1]
    df_new=df.copy()
    for key in avg_map:
        df_new.replace(df_new.iloc[key][columnName],avg_map[key],inplace=True)
    return df_new




def get_concentration_along_peaks(df,columnName,minutes=1):
    indices=mean_peaks_detection(df,columnName,show=False)[0]
    df_data=pd.DataFrame()
    for i in indices:
        j=df.iloc[i]["time"]
        df_data=pd.concat([df_data,df.loc[(df["time"] >= j-timedelta(minutes = minutes))&(df["time"] <= j+timedelta(minutes = minutes))]])
    return df_data




def spikes_detection(df_,columnNames,columnName,approach=3,threshold=None,show_all=False,show_plot=False):
    aggregation=[]
    avg_map={}
    values=['activity_time', 'Humidity', 'Temperature']
    df_temperature, newdf_t, prec_t, rec_t = cusum_(df_, 'Temperature', [0.6], [0.05], False)
    df_humidity, newdf_h, prec_h, rec_h = cusum_(df_, 'Humidity', [4], [0.05], False)
    if len(df_['NO2'].dropna())>0:
        df_NO2, newdf_NO2, prec_NO2, rec_NO2 = cusum_(df_, 'NO2', [15], [15], False)
        values.append('NO2')
    BC_size=len(df_["BC"].dropna())
    if BC_size>0:
        df_BC, newdf_BC, prec_BC, rec_BC = cusum_(df_, 'BC', [900], [500], False)    
        values.append('BC')
    TH1 = ps.sqldf(sqlcode1,locals())
    TH2 = ps.sqldf(sqlcode2,locals())
    if BC_size>0:
        TH3 = ps.sqldf(sqlcode3,locals())
    else:
        TH3=TH2
    
    res2=compute_verification_column(TH3,values)
    res2.drop(res2[res2.activity_time == df_.sort_values(['time']).time.iloc[0]].index, inplace = True)
    res2.drop_duplicates('verification', inplace = True)
    new_df=get_concentration_along_changes(df_,res2,columnNames).index.to_list()
        
    if approach==3:
#         indices=mean_peaks_detection(df_,columnName,show=False)[0]
#         indices=mean_peaks_detection__(df_,columnName,show=False)[0]
        indices,avg_map=mean_peaks_detection(df_,columnName,show=False)
    else:
        if approach==2:
            if threshold==None:
                indices=peaks_detection__(df_,columnName,show=False)
            else:
                indices=peaks_detection__(df_,columnName,threshold=threshold,show=False)            
        else:
            if approach==1:
                if threshold==None:
                    indices=peaks_detection_(df_,columnName,threshold=2000,show=False)
                else:
                    indices=peaks_detection_(df_,columnName,threshold=threshold,show=False)
    if show_all==True:
        show_all_(df_,columnName,indices,new_df)
    else:
        if show_plot ==True:
            show_plot_(df_,columnName,indices,new_df)
    for k in new_df:
        for m in indices:
            if k==m:
                aggregation.append(k)
    return [new_df,indices,aggregation,avg_map]



def peakes_removing_(df,columnName,threshold,show=False):
    indices=peaks_detection_(df,columnName,threshold,show)
    df_new=df.copy()
    df_new.drop(indices,inplace=True)
    return df_new    



def peaks_detection__(df_,columnName,threshold=1000,show=False):
    indices=[]
    dfs=split_dataframe(df_,time=60,show=False)
    for df in dfs:
        if len(df)<10:            
            df['columnName_Mean'] = df[columnName].rolling(window=len(df), center=True ).mean()
            temp=df.dropna(subset=["columnName_Mean"])
            for i in range(len(df[columnName])):
                if abs(abs(df[columnName].iloc[i])-abs(temp['columnName_Mean'].values[0]))>=threshold:
                        indices.append(df[columnName].index[i])
        else:
            df['columnName_Mean'] = df[columnName].rolling(window=10, center=True ).mean()
            for i in range(len(df[columnName])):
                if i==0 or i==1:
                    if abs(abs(df[columnName].iloc[i])-abs(df['columnName_Mean'].iloc[i+2]))>=threshold:
                        indices.append(df[columnName].index[i])
                else:
                    if i==df.index[-1] or i==df.index[-2]:
                        if abs(abs(df[columnName].iloc[i])-abs(df['columnName_Mean'].iloc[i-2]))>=threshold:
                            indices.append(df[columnName].index[i])
                    else:
                        if abs(abs(df[columnName].iloc[i])-abs(df['columnName_Mean'].iloc[i]))>=threshold:
                            indices.append(df[columnName].index[i])
    if show==True:
        plotly.offline.init_notebook_mode(connected=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df_[columnName],
            mode='lines+markers',
            name='Original Plot'
        ))

        fig.add_trace(go.Scatter(
            x=indices,
            y=[df_[columnName][j] for j in indices],
            mode='markers',
            marker=dict(
            size=8,
            color='red',
            symbol='cross'
            ),
            name='Detected Peaks'
        ))

        fig.show()
    return indices



def peakes_removing__(df,columnName,threshold=1000,show=False):
    indices=peaks_detection__(df,columnName,threshold,show)
    df_new=df.copy()
    df_new.drop(indices,inplace=True)
    return df_new    



def show_plot_(df_,columnName,indices,new_df,aggregation,below_above_threshold):
    plotly.offline.init_notebook_mode(connected=True)
    fig = go.Figure()    
    fig.add_trace(go.Scatter(
        y=df_[columnName],
        mode='lines+markers',
        name=columnName+' Plot',
    ))

    fig.add_trace(go.Scatter(
        x=indices,
        y=[df_[columnName][j] for j in indices],
        mode='markers',
        marker=dict(
        size=8,
        color='red',
        symbol='square-open-dot'
        ),
        name='Detected Peaks'
    ))
    fig.add_trace(go.Scatter(
        x=new_df,
        y=[df_[columnName][j] for j in new_df],
        mode='markers',
        marker=dict(
        size=8,
        color='green',
        symbol='cross'
        ),
        name='Detected Changes'
    ))
    fig.add_trace(go.Scatter(
        x=aggregation,
        y=[df_[columnName][j] for j in aggregation],
        mode='markers',
        marker=dict(
        size=8,
        color='black',
        symbol='circle'
        ),
        name='Aggregation'
    ))
    
    fig.add_trace(go.Scatter(
        x=below_above_threshold,
        y=[df_[columnName][j] for j in below_above_threshold],
        mode='markers',
        marker=dict(
        size=8,
        color='yellow',
        symbol='circle'
        ),
        name='Below/Above Threshold'
    ))
    fig.show()




def show_all_(df_,columnName,indices,new_df):
    plotly.offline.init_notebook_mode(connected=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        y=df_[columnName],
        mode='lines+markers',
        name=str(columnName)+' Plot',        
    ),secondary_y=False,)
    
    fig.add_trace(go.Scatter(
        y=df_["Temperature"],
        mode='lines+markers',    
        name="Temperature Plot"
    ),secondary_y=True,)
        
    fig.add_trace(go.Scatter(
        y=df_["Humidity"],
        mode='lines+markers',            
        name="Humidity Plot"
    ),secondary_y=True,)

    fig.add_trace(go.Scatter(
        x=indices,
        y=[df_[columnName][j] for j in indices],
        mode='markers',
        marker=dict(
        size=8,
        color='red',
        symbol='square-open-dot'
        ),
        name='Detected Peaks'
    ))
    fig.add_trace(go.Scatter(
        x=new_df,
        y=[df_[columnName][j] for j in new_df],
        mode='markers',
        marker=dict(
        size=8,
        color='green',
        symbol='cross'
        ),
        name='Detected Peaks'
    ))

    fig.add_trace(go.Scatter(
        x=new_df,
        y=[df_["Temperature"][j] for j in new_df],
        mode='markers',
        marker=dict(
        size=8,
        color='green',
        symbol='cross'
        ),
        name='Detected Peaks'
    ),secondary_y=True)
    fig.show()



def mean_peaks_replacing_negative_replaced_by_abs(df,columnName,window_size=10,show=False):
    peaks=mean_peaks_detection_negative_replaced_by_abs(df,columnName,window_size,show)
    indices=peaks[0]
    avg_map=peaks[1]
    df_new=df.copy()
    df_new["Replaced "+columnName]=np.zeros(len(df_new))
    for key in avg_map:
        df_new.replace(df_new.iloc[key][columnName],avg_map[key],inplace=True)
        df_new["Replaced "+columnName].iloc[key]=1
    return df_new



def calculate_avtivity_time(df__):
    df_=df__.copy()
    dfs=split_dataframe(df_,time=360,show=False)
    indices=[]
    for df in dfs:        
        df["truth"]=np.zeros(len(df))
        for i in range(len(df)):
            if (df.activity.iloc[i] != df.activity.shift(1).iloc[i]) |(df.event.iloc[i] != df.event.shift(1).iloc[i] ):
                df["truth"].iloc[i-1]=1
                df["truth"].iloc[i]=1
        indices+=df[df["truth"]==1].index.to_list()
    return indices




def spikes_detection_validation_with_activities(df_,columnName,approach=3,threshold=None,show_all=False,show_plot=False):
    aggregation=[]
    new_df=calculate_avtivity_time(df_)
    avg_map={}    
    if approach==3:
        indices,avg_map=mean_peaks_detection(df_,columnName,show=False)
    else:
        if approach==2:
            if threshold==None:
                indices=peaks_detection__(df_,columnName,show=False)
            else:
                indices=peaks_detection__(df_,columnName,threshold=threshold,show=False)            
        else:
            if approach==1:
                if threshold==None:
                    indices=peaks_detection_(df_,columnName,threshold=2000,show=False)
                else:
                    indices=peaks_detection_(df_,columnName,threshold=threshold,show=False)
    if show_all==True:
        show_all_(df_,columnName,indices,new_df)
    else:
        if show_plot ==True:
            show_plot_(df_,columnName,indices,new_df)
    for k in new_df:
        for m in indices:
            if k==m:
                aggregation.append(k)
    return [new_df,indices,aggregation,avg_map]



def spikes_detection_validation_with_changes(df_,columnNames,columnName,approach=3,window_size=10,threshold=None,show_all=False,show_plot=False,time=360):
    aggregation=[]
    avg_map={}
    values=['activity_time', 'Humidity', 'Temperature']
    df_temperature, newdf_t, prec_t, rec_t = cusum_(df_, 'Temperature', [0.6], [0.05], False)
    df_humidity, newdf_h, prec_h, rec_h = cusum_(df_, 'Humidity', [4], [0.05], False)
    if len(df_["NO2"].dropna())>0:
        values.append('NO2')
        df_NO2, newdf_NO2, prec_NO2, rec_NO2 = cusum_(df_, 'NO2', [15], [15], False)
    BC_size=len(df_["BC"].dropna())
    if BC_size>0:
        values.append('BC')
        df_BC, newdf_BC, prec_BC, rec_BC = cusum_(df_, 'BC', [900], [500], False)    
    TH1 = ps.sqldf(sqlcode1,locals())
    TH2 = ps.sqldf(sqlcode2,locals())
    if BC_size>0:
        TH3 = ps.sqldf(sqlcode3,locals())
    else:
        TH3=TH2    
    
    res2=compute_verification_column(TH3,values)
    res2.drop(res2[res2.activity_time == df_.sort_values(['time']).time.iloc[0]].index, inplace = True)
    res2.drop_duplicates('verification', inplace = True)
    new_df=get_concentration_along_changes(df_,res2,columnNames).index.to_list()
    
    if approach==3:
        indices,avg_map=mean_peaks_detection(df_,columnName,window_size,show=False)
    else:
        if approach==2:
            if threshold==None:
                indices=peaks_detection__(df_,columnName,show=False)
            else:
                indices=peaks_detection__(df_,columnName,threshold=threshold,show=False)            
        else:
            if approach==1:
                if threshold==None:
                    indices=peaks_detection_(df_,columnName,threshold=2000,show=False)
                else:
                    indices=peaks_detection_(df_,columnName,threshold=threshold,show=False)
    if show_all==True:
        show_all_(df_,columnName,indices,new_df)
    else:
        if show_plot ==True:
            show_plot_(df_,columnName,indices,new_df)
    for k in new_df:
        for m in indices:
            if k==m:
                aggregation.append(k)
    return [new_df,indices,aggregation,avg_map]



def spikes_detection_along_range_changes(df_,columnNames,columnName,approach=3,threshold=None,show_all=False,show_plot=False):
    aggregation=[]
    avg_map={}
    values=['activity_time', 'Humidity', 'Temperature']
    df_temperature, newdf_t, prec_t, rec_t = cusum_(df_, 'Temperature', [0.6], [0.05], False)
    df_humidity, newdf_h, prec_h, rec_h = cusum_(df_, 'Humidity', [4], [0.05], False)
    if len(df_['NO2'].dropna())>0:
        values.append('NO2')
        df_NO2, newdf_NO2, prec_NO2, rec_NO2 = cusum_(df_, 'NO2', [15], [15], False)
    BC_size=len(df_["BC"].dropna())
    if BC_size>0:
        values.append('BC')
        df_BC, newdf_BC, prec_BC, rec_BC = cusum_(df_, 'BC', [900], [500], False)    
    TH1 = ps.sqldf(sqlcode1,locals())
    TH2 = ps.sqldf(sqlcode2,locals())
    if BC_size>0:
        TH3 = ps.sqldf(sqlcode3,locals())
    else:
        TH3=TH2
    
    res2=compute_verification_column(TH3,values)
    res2.drop(res2[res2.activity_time == df_.sort_values(['time']).time.iloc[0]].index, inplace = True)
    res2.drop_duplicates('verification', inplace = True)
    new_df=get_range_concentration_along_changes(df_,res2,columnNames,minutes=5)
    df_change_index=new_df.index.to_list()

    if approach==3:
        indices,avg_map=mean_peaks_detection(new_df,columnName,show=False)
    else:
        if approach==2:
            if threshold==None:
                indices=peaks_detection__(df_,columnName,show=False)
            else:
                indices=peaks_detection__(df_,columnName,threshold=threshold,show=False)            
        else:
            if approach==1:
                if threshold==None:
                    indices=peaks_detection_(df_,columnName,threshold=2000,show=False)
                else:
                    indices=peaks_detection_(df_,columnName,threshold=threshold,show=False)
    if show_all==True:
        show_all_(df_,columnName,indices,df_change_index)
    else:
        if show_plot ==True:
            show_plot_(df_,columnName,indices,df_change_index)
    for k in df_change_index:
        for m in indices:
            if k==m:
                aggregation.append(k)
    return [df_change_index,indices,aggregation,avg_map]




def spikes_removal(df,indices):
    df_new=df.copy()
    df_new.drop(indices,inplace=True)
    print("Data loss ",((len(df)-len(df_new))/len(df))*100,"%")
    return df_new




def spikes_replacement(df,columnName,avg_map):
    df_new=df.copy()
    df_new["Replaced "+columnName]=np.zeros(len(df_new))
    for key in avg_map:
        df_new.replace(df_new.iloc[key][columnName],avg_map[key],inplace=True)
        df_new["Replaced "+columnName].iloc[key]=1
    return df_new




sqlcode1 = '''
select newdf_t.cusum_time as Temperature,
newdf_h.cusum_time as Humidity,
newdf_t.activity_time,
newdf_t.activity,
newdf_t.event

from newdf_t
left join newdf_h 
on newdf_t.activity_time = newdf_h.activity_time

UNION

select newdf_t.cusum_time as Temperature,
newdf_h.cusum_time as Humidity,
newdf_h.activity_time,
newdf_h.activity,
newdf_h.event

from newdf_h
left join newdf_t 
on newdf_h.activity_time = newdf_t.activity_time

'''


###Temperature & humidity & NO2

sqlcode2 = '''
select newdf_NO2.cusum_time as NO2,
TH1.* 

from TH1
left join newdf_NO2
on newdf_NO2.activity_time = TH1.activity_time

UNION

select newdf_NO2.cusum_time as NO2,
TH1.temperature,
TH1.humidity,
newdf_NO2.activity_time,
newdf_NO2.activity,
newdf_NO2.event

from newdf_NO2
left join TH1 
on TH1.activity_time = newdf_NO2.activity_time

'''


###Temperature & humidity & NO2 & BC

sqlcode3 = '''
select newdf_BC.cusum_time as BC,
TH2.* 

from TH2
left join newdf_BC
on TH2.activity_time = newdf_BC.activity_time

UNION

select newdf_BC.cusum_time as BC,
TH2.NO2,
TH2.temperature,
TH2.humidity,
newdf_BC.activity_time,
newdf_BC.activity,
newdf_BC.event

from newdf_BC
left join TH2
on newdf_BC.activity_time = TH2.activity_time

'''



def data_segmentation(df,thresholds={"temperature_th":0.6,"humidity_th":4,"NO2_th":15,"BC_th":900,"PM10_th":9,"PM2.5_th":15,"PM1.0_th":11,"speed_th":1},drift={"temperature_drift":0.05,"humidity_drift":0.05,"NO2_drift":15,"BC_drift":500,"PM10_drift":0.05,"PM2.5_drift":0.03,"PM1.0_drift":0.5,"speed_drift":0.1},remove_peaks=False,columnName=None,delay=5,buffer=5):    
       
        ###Temperature & humidity

    sqlcode1 = '''
    select newdf_t.cusum_time as Temperature,
    newdf_h.cusum_time as Humidity,
    newdf_t.activity_time,
    newdf_t.activity,
    newdf_t.event

    from newdf_t
    left join newdf_h 
    on newdf_t.activity_time = newdf_h.activity_time

    UNION

    select newdf_t.cusum_time as Temperature,
    newdf_h.cusum_time as Humidity,
    newdf_h.activity_time,
    newdf_h.activity,
    newdf_h.event

    from newdf_h
    left join newdf_t 
    on newdf_h.activity_time = newdf_t.activity_time

    '''


    ###Temperature & humidity & NO2

    sqlcode2 = '''
    select newdf_NO2.cusum_time as NO2,
    TH1.* 

    from TH1
    left join newdf_NO2
    on newdf_NO2.activity_time = TH1.activity_time

    UNION

    select newdf_NO2.cusum_time as NO2,
    TH1.temperature,
    TH1.humidity,
    newdf_NO2.activity_time,
    newdf_NO2.activity,
    newdf_NO2.event

    from newdf_NO2
    left join TH1 
    on TH1.activity_time = newdf_NO2.activity_time

    '''


    ###Temperature & humidity & NO2 & BC

    sqlcode3 = '''
    select newdf_BC.cusum_time as BC,
    TH2.* 

    from TH2
    left join newdf_BC
    on TH2.activity_time = newdf_BC.activity_time

    UNION

    select newdf_BC.cusum_time as BC,
    TH2.NO2,
    TH2.temperature,
    TH2.humidity,
    newdf_BC.activity_time,
    newdf_BC.activity,
    newdf_BC.event

    from newdf_BC
    left join TH2
    on newdf_BC.activity_time = TH2.activity_time

    '''

    ###Temperature & humidity & NO2 & BC & PM10

    sqlcode4 = '''
    select newdf_PM10.cusum_time as PM10,
    TH3.* 

    from TH3
    left join newdf_PM10
    on TH3.activity_time = newdf_PM10.activity_time

    UNION

    select newdf_PM10.cusum_time as PM10,
    TH3.BC,
    TH3.NO2,
    TH3.temperature,
    TH3.humidity,
    newdf_PM10.activity_time,
    newdf_PM10.activity,
    newdf_PM10.event

    from newdf_PM10
    left join TH3
    on newdf_PM10.activity_time = TH3.activity_time

    '''

    ###Temperature & humidity & NO2 & BC & PM10 & PM2.5

    sqlcode5 = '''
    select newdf_PM25.cusum_time as PM25,
    TH4.* 

    from TH4
    left join newdf_PM25
    on TH4.activity_time = newdf_PM25.activity_time

    UNION

    select newdf_PM25.cusum_time as PM25,
    TH4.PM10,
    TH4.BC,
    TH4.NO2,
    TH4.temperature,
    TH4.humidity,
    newdf_PM25.activity_time,
    newdf_PM25.activity,
    newdf_PM25.event

    from newdf_PM25
    left join TH4
    on newdf_PM25.activity_time = TH4.activity_time

    '''

    ###Temperature & humidity & NO2 & BC & PM10 & PM2.5 & PM1.0

    sqlcode6 = '''
    select newdf_PM1.cusum_time as "PM1.0",
    TH5.* 

    from TH5
    left join newdf_PM1
    on TH5.activity_time = newdf_PM1.activity_time

    UNION

    select newdf_PM1.cusum_time as "PM1.0",
    TH5.PM25,
    TH5.PM10,
    TH5.BC,
    TH5.NO2,
    TH5.temperature,
    TH5.humidity,
    newdf_PM1.activity_time,
    newdf_PM1.activity,
    newdf_PM1.event

    from newdf_PM1
    left join TH5
    on newdf_PM1.activity_time = TH5.activity_time

    '''

    ###Temperature & humidity & NO2 & BC & PM10 & PM2.5 & PM1.0 & speed

    sqlcode7 = '''
    select newdf_speed.cusum_time as Speed,
    TH6.* 

    from TH6
    left join newdf_speed
    on TH6.activity_time = newdf_speed.activity_time

    UNION

    select newdf_speed.cusum_time as Speed,
    TH6."PM1.0",
    TH6.PM25,
    TH6.PM10,
    TH6.BC,
    TH6.NO2,
    TH6.temperature,
    TH6.humidity,
    newdf_speed.activity_time,
    newdf_speed.activity,
    newdf_speed.event

    from newdf_speed
    left join TH6
    on newdf_speed.activity_time = TH6.activity_time

    '''


    ###Temperature & humidity
    sqlcode11 = '''
    select newdf_t.cusum_time as temperature,
    newdf_h.cusum_time as humidity,
    newdf_t.activity_time,
    newdf_t.activity,
    newdf_t.event

    from newdf_t
    left join newdf_h 
    on newdf_t.activity_time = newdf_h.activity_time

    UNION

    select newdf_t.cusum_time as temperature,
    newdf_h.cusum_time as humidity,
    newdf_h.activity_time,
    newdf_h.activity,
    newdf_h.event

    from newdf_h
    left join newdf_t 
    on newdf_h.activity_time = newdf_t.activity_time

    '''

    ###Temperature & humidity & PM10

    sqlcode12 = '''
    select newdf_PM10.cusum_time as PM10,
    TH1.* 

    from TH1
    left join newdf_PM10
    on newdf_PM10.activity_time = TH1.activity_time

    UNION

    select newdf_PM10.cusum_time as PM10,
    TH1.temperature,
    TH1.humidity,
    newdf_PM10.activity_time,
    newdf_PM10.activity,
    newdf_PM10.event

    from newdf_PM10
    left join TH1 
    on TH1.activity_time = newdf_PM10.activity_time

    '''


    ###Temperature & humidity & PM10 & PM2.5

    sqlcode13 = '''
    select newdf_PM25.cusum_time as PM25,
    TH2.* 

    from TH2
    left join newdf_PM25
    on TH2.activity_time = newdf_PM25.activity_time

    UNION

    select newdf_PM25.cusum_time as PM25,
    TH2.PM10,
    TH2.temperature,
    TH2.humidity,
    newdf_PM25.activity_time,
    newdf_PM25.activity,
    newdf_PM25.event

    from newdf_PM25
    left join TH2
    on newdf_PM25.activity_time = TH2.activity_time

    '''

    ###Temperature & humidity & PM10 & PM2.5 & PM1.0

    sqlcode14 = '''
    select newdf_PM1.cusum_time as "PM1.0",
    TH3.* 

    from TH3
    left join newdf_PM1
    on TH3.activity_time = newdf_PM1.activity_time

    UNION

    select newdf_PM1.cusum_time as "PM1.0",
    TH3.PM25,
    TH3.PM10,
    TH3.temperature,
    TH3.humidity,
    newdf_PM1.activity_time,
    newdf_PM1.activity,
    newdf_PM1.event

    from newdf_PM1
    left join TH3
    on newdf_PM1.activity_time = TH3.activity_time

    '''

    ###Temperature & humidity & PM10 & PM2.5 & PM1.0 & NO2

    sqlcodeNO2 = '''
    select newdf_NO2.cusum_time as NO2,
    TH4.* 

    from TH4
    left join newdf_NO2
    on TH4.activity_time = newdf_NO2.activity_time

    UNION

    select newdf_NO2.cusum_time as NO2,
    TH4.PM10,
    TH4.PM25,
    TH4."PM1.0",
    TH4.temperature,
    TH4.humidity,
    newdf_NO2.activity_time,
    newdf_NO2.activity,
    newdf_NO2.event

    from newdf_NO2
    left join TH4
    on newdf_NO2.activity_time = TH4.activity_time
    '''


    ###Temperature & humidity & PM10 & PM2.5 & PM1.0 & BC

    sqlcodeBC = '''
    select newdf_BC.cusum_time as BC,
    TH4.* 

    from TH4
    left join newdf_BC
    on TH4.activity_time = newdf_BC.activity_time

    UNION

    select newdf_BC.cusum_time as BC,
    TH4.PM10,
    TH4.PM25,
    TH4."PM1.0",
    TH4.temperature,
    TH4.humidity,
    newdf_BC.activity_time,
    newdf_BC.activity,
    newdf_BC.event

    from newdf_BC
    left join TH4
    on newdf_BC.activity_time = TH4.activity_time
    '''
    sqlcodeSpeed_BC = '''
    select newdf_speed.cusum_time as Speed,
    TH5.* 

    from TH5
    left join newdf_speed
    on TH5.activity_time = newdf_speed.activity_time

    UNION

    select newdf_speed.cusum_time as Speed,
    TH5."PM1.0",
    TH5.PM25,
    TH5.PM10,
    TH5.BC,
    TH5.temperature,
    TH5.humidity,
    newdf_speed.activity_time,
    newdf_speed.activity,
    newdf_speed.event

    from newdf_speed
    left join TH5
    on newdf_speed.activity_time = TH5.activity_time

    '''

    sqlcodeSpeed = '''
    select newdf_speed.cusum_time as Speed,
    TH4.* 

    from TH4
    left join newdf_speed
    on TH4.activity_time = newdf_speed.activity_time

    UNION

    select newdf_speed.cusum_time as Speed,
    TH4."PM1.0",
    TH4.PM25,
    TH4.PM10,
    TH4.temperature,
    TH4.humidity,
    newdf_speed.activity_time,
    newdf_speed.activity,
    newdf_speed.event

    from newdf_speed
    left join TH4
    on newdf_speed.activity_time = TH4.activity_time

    '''


    sqlcodeSpeed_NO2 = '''
    select newdf_speed.cusum_time as Speed,
    TH5.* 

    from TH5
    left join newdf_speed
    on TH5.activity_time = newdf_speed.activity_time

    UNION

    select newdf_speed.cusum_time as Speed,
    TH5."PM1.0",
    TH5.PM25,
    TH5.PM10,
    TH5.NO2,
    TH5.temperature,
    TH5.humidity,
    newdf_speed.activity_time,
    newdf_speed.activity,
    newdf_speed.event

    from newdf_speed
    left join TH5
    on newdf_speed.activity_time = TH5.activity_time
    '''
    df2=df.copy()
    if len(df2.dropna())==0:
        return [df2,0,0]
    if remove_peaks==True and columnName!=None and len(df2.dropna(subset=[columnName]))>0:
        df2=spikes_removal(df2,spikes_detection_validation_with_changes(df2,["Temperature"],columnName)[1])
        df2.reset_index(inplace=True)
    df2['truth'] = np.zeros(len(df2))
    df2.sort_values(['time'], inplace=True)
    for i in range(len(df2)):
        if (df2.activity.iloc[i] != df2.activity.shift(1).iloc[i]) |        (df2.event.iloc[i] != df2.event.shift(1).iloc[i]):
            df2.truth.iloc[i] = 1
    df2['truth'].iloc[0]=0
    
    df_temperature, newdf_t, prec_t, rec_t = cusum_(df2, 'Temperature', [thresholds["temperature_th"]], [drift["temperature_drift"]], False)
    df_humidity, newdf_h, prec_h, rec_h = cusum_(df2, 'Humidity', [thresholds["humidity_th"]], [drift["humidity_drift"]], False)
    if len(df2.dropna(subset=["NO2"]))>0:
        df_NO2, newdf_NO2, prec_NO2, rec_NO2 = cusum_(df2, 'NO2',  [thresholds["NO2_th"]], [drift["NO2_drift"]], False)
    if len(df2.dropna(subset=["BC"]))>0:
        df_BC, newdf_BC, prec_BC, rec_BC = cusum_(df2, 'BC',  [thresholds["BC_th"]], [drift["BC_drift"]], False)
    df_PM10, newdf_PM10, prec_PM10, rec_PM10 = cusum_(df2, 'PM10',  [thresholds["PM10_th"]], [drift["PM10_drift"]], False)
    df_PM25, newdf_PM25, prec_PM25, rec_PM25  = cusum_(df2, 'PM2.5',  [thresholds["PM2.5_th"]], [drift["PM2.5_drift"]], False)
    df_PM1, newdf_PM1, prec_PM1, rec_PM1  = cusum_(df2, 'PM1.0', [thresholds["PM1.0_th"]], [drift["PM1.0_drift"]], False)
    if len(df2.dropna(subset=["vitesse(m/s)"]))>0:
        df_speed, newdf_speed, prec_speed, rec_speed  = cusum_(df2, 'vitesse(m/s)', [thresholds["speed_th"]], [drift["speed_drift"]], False)
    
    # DFs = {x:pd.DataFrame() for x in dfnames}
    if  len(df2.dropna(subset=["BC"]))>0 and len(df2.dropna(subset=["NO2"]))>0 and len(df2.dropna(subset=["vitesse(m/s)"]))>0:
        TH1 = ps.sqldf(sqlcode1,locals())
        TH2 = ps.sqldf(sqlcode2,locals())
        TH3 = ps.sqldf(sqlcode3,locals())
        TH4 = ps.sqldf(sqlcode4,locals())
        TH5 = ps.sqldf(sqlcode5,locals())
        TH6 = ps.sqldf(sqlcode6,locals())
        TH7 = ps.sqldf(sqlcode7,locals())
    else:
        if len(df2.dropna(subset=["BC"]))==0 and len(df2.dropna(subset=["NO2"]))==0 and len(df2.dropna(subset=["vitesse(m/s)"]))==0:
            TH1 = ps.sqldf(sqlcode11,locals())
            TH2 = ps.sqldf(sqlcode12,locals())
            TH3 = ps.sqldf(sqlcode13,locals())
            TH4 = ps.sqldf(sqlcode14,locals())
            TH7=TH4
        else:
            if  len(df2.dropna(subset=["NO2"]))==0 and len(df2.dropna(subset=["vitesse(m/s)"]))==0:
                TH1 = ps.sqldf(sqlcode11,locals())
                TH2 = ps.sqldf(sqlcode12,locals())
                TH3 = ps.sqldf(sqlcode13,locals())
                TH4 = ps.sqldf(sqlcode14,locals())
                TH5 = ps.sqldf(sqlcodeBC,locals())
                TH7=TH5
            else:
                if  len(df2.dropna(subset=["BC"]))==0 and len(df2.dropna(subset=["vitesse(m/s)"]))==0:
                    TH1 = ps.sqldf(sqlcode11,locals())
                    TH2 = ps.sqldf(sqlcode12,locals())
                    TH3 = ps.sqldf(sqlcode13,locals())
                    TH4 = ps.sqldf(sqlcode14,locals())
                    TH5 = ps.sqldf(sqlcodeNO2,locals())
                    TH7=TH5
                else:
                    if  len(df2.dropna(subset=["NO2"]))==0:
                        TH1 = ps.sqldf(sqlcode11,locals())
                        TH2 = ps.sqldf(sqlcode12,locals())
                        TH3 = ps.sqldf(sqlcode13,locals())
                        TH4 = ps.sqldf(sqlcode14,locals())
                        TH5 = ps.sqldf(sqlcodeBC,locals())
                        TH6 = ps.sqldf(sqlcodeSpeed_BC,locals())
                        TH7=TH6
                    else:                        
                        if  len(df2.dropna(subset=["BC"]))==0:
                            TH1 = ps.sqldf(sqlcode11,locals())
                            TH2 = ps.sqldf(sqlcode12,locals())
                            TH3 = ps.sqldf(sqlcode13,locals())
                            TH4 = ps.sqldf(sqlcode14,locals())
                            TH5 = ps.sqldf(sqlcodeNO2,locals())
                            TH6 = ps.sqldf(sqlcodeSpeed_NO2,locals())
                            TH7=TH6
                        else:
                            if len(df2.dropna(subset=["BC"]))==0 and len(df2.dropna(subset=["NO2"]))==0 and len(df2.dropna(subset=["vitesse(m/s)"]))>0:
                                TH1 = ps.sqldf(sqlcode11,locals())
                                TH2 = ps.sqldf(sqlcode12,locals())
                                TH3 = ps.sqldf(sqlcode13,locals())
                                TH4 = ps.sqldf(sqlcode14,locals())
                                TH5 = ps.sqldf(sqlcodeSpeed,locals())
                                TH7=TH5
        


    truth = df2[df2.truth==1][['time','activity']]
    truth['start_date']= truth.time - timedelta(minutes = buffer)
    truth['end_date'] = truth.time + timedelta(minutes = buffer)
    
    values=list(TH7.drop(['activity','event','activity_time'],axis=1).columns)
    reversed_values=Reverse(values)
    res=compute_verification_column(TH7,reversed_values)
#     res['verification'] = pd.to_datetime(res['verification'], format='%Y-%m-%d %H:%M:%S')
#     res.drop(res[res.activity_time == df2.sort_values(['time']).time.iloc[0]].index, inplace = True)
#     res.drop_duplicates('verification', inplace = True)
#     res.sort_values(['verification'], inplace = True)
    
#     for i in range(len(res)):
#         if (res['verification'].iloc[i] - res['verification'].shift(1).iloc[i] < timedelta(minutes=5)) & \
#         ((res.activity_time.isnull().iloc[i]) | (res.activity_time.shift(1).isnull().iloc[i])):
#             res['verification'].iloc[i] = res['verification'].shift(1).iloc[i]
#             for j in res.drop(['activity_time', 'activity', 'event', 'verification'], axis=1).columns:
#                 if res[j].shift(1).isnull().iloc[i]:
#                     res[j].iloc[i-1] = res[j].iloc[i]   
    res=merge_close_changes(res,delay=delay)
    
#     res = res.sort_values(['verification', 'activity_time']).drop_duplicates('verification', keep='first')
    sqlcode ='''
        select res.*,
        truth.time as truth

        from res
        left join truth
        on res.verification between truth.start_date and truth.end_date

    '''

    res = ps.sqldf(sqlcode,locals())
    res.drop(['activity','event'],axis=1,inplace=True)
    FP_ = len(res[(res.activity_time.isnull()) & (res.verification != 0)])
    TP = len(res[(res.activity_time.notnull()) & (res.verification != 0)])
    FP = len(res[(res.activity_time.isnull()) & (res.verification != 0)])
    FN = len(res[(res.activity_time.notnull()) & (res.verification == 0)])
    if TP>0:
        precision=TP / (TP + FP_)
        recall=TP / (TP + FN)
        print( 'Precision: ', TP / (TP + FP_))
        print('Recall :', TP / (TP + FN))
    else:    
        precision=0
        recall=0
        print( 'Precision: 0')
        print('Recall : 0')


    return [res,precision,recall]



def segmentation(df_ids,number_of_attempts=None,remove_peaks=False,columnName=None,folder_name='segmented_data',buffer=5,delay=5):
    final_result=[]
    for i in range(len(df_ids)):        
        if i==number_of_attempts:
            break    
        df=get_postgres_data(kit_id=df_ids["kit_id"].iloc[i],participant_id=df_ids["participant_id"].iloc[i])
        if remove_peaks==True and columnName!=None and len(df.dropna(subset=[columnName]))>0:
            res,precision,recall=data_segmentation(df,remove_peaks=remove_peaks,columnName=columnName,buffer=buffer,delay=delay)
        else:
            res,precision,recall=data_segmentation(df,buffer=buffer,delay=delay)

        df_ids['start_date']=df_ids['start_date'].astype(str)
        df_ids['end_date']=df_ids['end_date'].astype(str)
        meta_data={"kit_id":df_ids["kit_id"].iloc[i].tolist(),"participant_id":df_ids["participant_id"].iloc[i].tolist(),
                   "participant_virtual_id":df_ids["participant_virtual_id"].iloc[i],
                  "start_date":df_ids["start_date"].iloc[i],"end_date":df_ids["end_date"].iloc[i],"precision":precision,"recall":recall}
        js = json.dumps(meta_data)
        f_meta = open(folder_name+"/meta_data/kit:"+str(meta_data["kit_id"])+",participant_virtual_id:"+str(meta_data["participant_virtual_id"])+"-meta-data.json","w")
        f_meta.write(js)
        f_meta.close()
        res.to_csv(folder_name+"/data/kit:"+str(meta_data["kit_id"])+",participant_virtual_id:"+str(meta_data["participant_virtual_id"])+"-data.csv")
        final_result.append(res)        



def apply_segmentation(campaign_id=1,remove_peaks=False,columnName=None,folder_name="segmented_data",delay=5,buffer=5):
    df_ids=get_all_participantIDs_and_kitIDs(campaign_id=campaign_id)
    result=segmentation(df_ids,columnName=columnName,folder_name=folder_name,delay=delay,buffer=buffer)
    return result




def get_segments_data(kit_id,participant_virtual_id,folder="segmented_data"):
    data_path=folder+"/data/kit:"+str(kit_id)+",participant_virtual_id:"+str(participant_virtual_id)+"-data.csv"
    meta_path=folder+"/meta_data/kit:"+str(kit_id)+",participant_virtual_id:"+str(participant_virtual_id)+"-meta-data.json"
    with open(meta_path) as json_file:
        data = json.load(json_file)
    df=pd.read_csv(data_path)
    return [data,df]




def Reverse(lst): 
    return [ele for ele in reversed(lst)]




def plotXY(df_,columnNameX,columnNameY):
    plotly.offline.init_notebook_mode(connected=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_[columnNameX],
        y=df_[columnNameY],
        mode='lines+markers',
        name='Original Plot'
    ))
    fig.show()




def mean_peaks_detection(df__,columnName,min_threshold=-2000,max_threshold=50000,window_size=10,show=False):
    indices=[]
    indices2=[]
    avg_map={}
    detected=0
    df_=df__.copy()    
    
    if min_threshold==None or max_threshold==None:
        print("Please make sure that you have entered correctly the min and max thresholds")
        return
    
    if "original_index" not in df_.columns:
        df_["original_index"]=df_.index
    dfs=split_dataframe(df_,time=360,show=False)
    
    for df in dfs:       
        if len(df)<=window_size:
            if len(df.dropna(subset=[columnName]))>0:
                df=df.dropna(subset=[columnName])
                df['columnName_Mean'] = df[columnName].rolling(window=len(df.dropna(subset=[columnName])), center=True ).mean()
                temp=df.dropna(subset=["columnName_Mean"])
                threshold=1/2*abs(temp['columnName_Mean'].values[0])
                th2=threshold
                th=abs(temp['columnName_Mean'].values[0])
    #             threshold=(5)*threshold
                for i in range(len(df[columnName])):
                    if (abs(df[columnName].iloc[i])>th and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(1).iloc[df[columnName].index[i]])>=th2 and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(-1).iloc[df[columnName].index[i]])>=th2 and ((df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>0) or (df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<0))) or (df__[columnName].iloc[df[columnName].index[i]]<min_threshold and df__[columnName].iloc[df[columnName].index[i]]<0) or (df__[columnName].iloc[df[columnName].index[i]]>max_threshold and df__[columnName].iloc[df[columnName].index[i]]>0):
                        indices2.append(int(df["original_index"].iloc[i]))
                        avg_map[int(df["original_index"].iloc[i])]=temp['columnName_Mean'].values[0]
                        indices.append(df[columnName].index[i])
    #                     avg_map[df[columnName].index[i]]=temp['columnName_Mean'].values[0]
                        detected+=1
        else:
            df['columnName_Mean'] = df[columnName].rolling(window=window_size, center=True ).mean()
            for i in range(len(df[columnName])):                   
                margin=int(window_size/2)
                if i<=window_size/2: 
                    if len(df.dropna(subset=[columnName]))>0:
                        threshold=(3)*abs(df['columnName_Mean'].iloc[i+margin])
                        th2=(1/2)*threshold
                        th=(4)*abs(df['columnName_Mean'].iloc[i+margin])
    #                     threshold=(2)*threshold    
                        if (abs(df[columnName].iloc[i])>th and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(1).iloc[df[columnName].index[i]])>=th2 and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(-1).iloc[df[columnName].index[i]])>=th2 and ((df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>0) or (df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<0))) or (df__[columnName].iloc[df[columnName].index[i]]<min_threshold and df__[columnName].iloc[df[columnName].index[i]]<0) or (df__[columnName].iloc[df[columnName].index[i]]>max_threshold and df__[columnName].iloc[df[columnName].index[i]]>0):
                            indices2.append(int(df["original_index"].iloc[i]))
                            avg_map[int(df["original_index"].iloc[i])]=df['columnName_Mean'].iloc[i+margin]
                            indices.append(df[columnName].index[i])
    #                         avg_map[df[columnName].index[i]]=df['columnName_Mean'].iloc[i+margin]
                            detected+=1
                else:
                    if i<=df.index[-margin]: 
                        if len(df.dropna(subset=[columnName]))>0:                              
                            threshold=(3)*abs(df['columnName_Mean'].iloc[i-margin])
                            th2=(1/2)*threshold
                            th=(4)*abs(df['columnName_Mean'].iloc[i-margin])
    #                         threshold=(2)*threshold

                            if ((abs(df[columnName].iloc[i])>th and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(1).iloc[df[columnName].index[i]])>=th2 and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(-1).iloc[df[columnName].index[i]])>=th2 and ((df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>0) or (df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<0))) or (df__[columnName].iloc[df[columnName].index[i]]<min_threshold and df__[columnName].iloc[df[columnName].index[i]]<0) or (df__[columnName].iloc[df[columnName].index[i]]>max_threshold and df__[columnName].iloc[df[columnName].index[i]]>0)) or (df__[columnName].iloc[df[columnName].index[i]]<min_threshold and df__[columnName].iloc[df[columnName].index[i]]<0) or (df__[columnName].iloc[df[columnName].index[i]]>max_threshold and df__[columnName].iloc[df[columnName].index[i]]>0):
                                indices2.append(int(df["original_index"].iloc[i]))
                                avg_map[int(df["original_index"].iloc[i])]=df['columnName_Mean'].iloc[i-margin]                       
                                indices.append(df[columnName].index[i])
    #                             avg_map[df[columnName].index[i]]=df['columnName_Mean'].iloc[i-margin]
                                detected+=1
                    else:
                        if len(df.dropna(subset=[columnName]))>0:
                            threshold=(3)*abs(df['columnName_Mean'].iloc[i])
                            th2=(1/2)*threshold
                            th=(4)*abs(df['columnName_Mean'].iloc[i])
    #                         threshold=(2)*threshold

                            if (abs(df[columnName].iloc[i])>th and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(1).iloc[df[columnName].index[i]])>=th2 and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(-1).iloc[df[columnName].index[i]])>=th2 and ((df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>0) or (df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<0))) or (df__[columnName].iloc[df[columnName].index[i]]<min_threshold and df__[columnName].iloc[df[columnName].index[i]]<0) or (df__[columnName].iloc[df[columnName].index[i]]>max_threshold and df__[columnName].iloc[df[columnName].index[i]]>0):
                                indices2.append(int(df["original_index"].iloc[i]))
                                avg_map[int(df["original_index"].iloc[i])]=df['columnName_Mean'].iloc[i]                       
                                indices.append(df[columnName].index[i])
    #                             avg_map[df[columnName].index[i]]=df['columnName_Mean'].iloc[i]
                                detected+=1
    if show==True:
        plotly.offline.init_notebook_mode(connected=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df_[columnName],
            mode='lines+markers',
            name='Original Plot'
        ))

        fig.add_trace(go.Scatter(
            x=indices,
            y=[df_[columnName][j] for j in indices],
            mode='markers',
            marker=dict(
            size=8,
            color='red',
            symbol='cross'
            ),
            name='Detected Peaks'
        ))

        fig.show()
    return [indices,avg_map,detected,indices2]




def getDistanceFromLatLonInM(lon1,lat1,lon2,lat2):
    R = 6371000 # Radius of the earth in km
    dLat = radians(lat2-lat1)
    dLon = radians(lon2-lon1)
    rLat1 = radians(lat1)
    rLat2 = radians(lat2)
    a = sin(dLat/2) * sin(dLat/2) + cos(rLat1) * cos(rLat2) * sin(dLon/2) * sin(dLon/2) 
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c # Distance in km
    return d




def getSpeedFromLonLat(lon1,lat1,lon2,lat2):
    if lon1==None or lon2==None or lat1==None or lat2==None:
        return np.nan
    return getDistanceFromLatLonInM(lon1,lat1,lon2,lat2)/60



def splitting(df,time=300):
    df=df.sort_values(by=['time'])
    dfs=[]
    start=0
    start_time=df.iloc[0]["time"]
    print("hello")
    for i in range(len(df)):
        print(df.shift(-1).iloc[i]["time"])
        if ((df.shift(-1).iloc[i]["time"]-start_time).seconds>=time):
            dfs.append(df.iloc[start:(i+1)])
            start=i+1
            start_time=df.shift(-1).iloc[i]["time"]            
    if start<len(df):
        dfs.append(df[start:-1])
    return dfs


def getDistanceFromLatLonInKM(lon1,lat1,lon2,lat2):
    R = 6371 # Radius of the earth in km
    dLat = radians(lat2-lat1)
    dLon = radians(lon2-lon1)
    rLat1 = radians(lat1)
    rLat2 = radians(lat2)
    a = sin(dLat/2) * sin(dLat/2) + cos(rLat1) * cos(rLat2) * sin(dLon/2) * sin(dLon/2) 
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c # Distance in km
    return d




def getSpeedFromLonLat_KM_per_H(lon1,lat1,lon2,lat2):
    return getDistanceFromLatLonInKM(lon1,lat1,lon2,lat2)/0.0166667


def spikes_detection_validation_with_changes_negative_replaced_by_abs(df_,columnNames,columnName,min_threshold,max_threshold,q=0.9999,interval=2,approach=3,window_size=10,threshold=None,show_all=False,show_plot=False,time=360):
    aggregation=[]
    below_above_threshold=[]
    avg_map={}
    values=['activity_time', 'Humidity', 'Temperature']
    df_["original_index"]=df_.index
    df_temperature, newdf_t, prec_t, rec_t = cusum_(df_, 'Temperature', [0.6], [0.05], False)
    df_humidity, newdf_h, prec_h, rec_h = cusum_(df_, 'Humidity', [4], [0.05], False)
    if len(df_["NO2"].dropna()):
        values.append("NO2")
        df_NO2, newdf_NO2, prec_NO2, rec_NO2 = cusum_(df_, 'NO2', [15], [15], False)
    BC_size=len(df_["BC"].dropna())
    if BC_size>0:
        values.append("BC")
        df_BC, newdf_BC, prec_BC, rec_BC = cusum_(df_, 'BC', [900], [500], False)    
    TH1 = ps.sqldf(sqlcode1,locals())
    if len(df_["NO2"].dropna())>0:
        TH2 = ps.sqldf(sqlcode2,locals())
    else:
        TH2=TH1
    if BC_size>0 and len(df_["NO2"].dropna())>0:
        TH3 = ps.sqldf(sqlcode3,locals())
    else:
        if BC_size>0:
            TH3 = ps.sqldf(sqlcode4,locals())
        else:
            TH3=TH2    

    
    res2=compute_verification_column(TH3,values)
    res2.drop(res2[res2.activity_time == df_.sort_values(['time']).time.iloc[0]].index, inplace = True)
    res2.drop_duplicates('verification', inplace = True)
    new_df=get_concentration_along_changes(df_,res2,columnNames,changes=False).index.to_list()
    new_df_range=get_range_concentration_along_changes(df_,res2,columnNames,changes=False,minutes=interval).index.to_list()
    
    if approach==3:
        indices=[]
        indices2=[]  
        index,avg_map,detected,index2=mean_peaks_detection_negative_replaced_by_abs(df_,columnName,min_threshold=min_threshold,max_threshold=max_threshold,q=q,window_size=window_size,show=False)
        indices+=index
        indices2+=index2

    else:
        if approach==2:
            if threshold==None:
                indices=peaks_detection__(df_,columnName,show=False)
            else:
                indices=peaks_detection__(df_,columnName,threshold=threshold,show=False)            
        else:
            if approach==1:
                if threshold==None:
                    indices=peaks_detection_(df_,columnName,threshold=2000,show=False)
                else:
                    indices=peaks_detection_(df_,columnName,threshold=threshold,show=False)
    for k in new_df_range:
        for m in indices:
            if k==m :
                aggregation.append(m)
            if (df_[columnName].iloc[m]<min_threshold and df_[columnName].iloc[m]<0) or (df_[columnName].iloc[m]>max_threshold and df_[columnName].iloc[m]>0):
                below_above_threshold.append(m)
    if show_all==True:
        show_all_(df_,columnName,indices2,new_df)
    else:
        if show_plot ==True:
            show_plot_(df_,columnName,indices2,new_df,list(set(aggregation)),list(set(below_above_threshold)))
    return [new_df,indices,list(set(aggregation)),avg_map]


def mean_peaks_detection_negative_replaced_by_abs(df__,columnName,min_threshold=-2000,max_threshold=50000,window_size=10,q=0.9999,show=False):
    indices=[]
    indices2=[]
    avg_map={}
    detected=0
    df_=df__.copy()    
    quantile=df_[columnName].quantile(q)
    
    if min_threshold==None or max_threshold==None:
        print("Please make sure that you have entered correctly the min and max thresholds")
        return
    #to make the max value included
    max_threshold=max_threshold-1
    
    if "original_index" not in df_.columns:
        df_["original_index"]=df_.index
    for i in range(len(df_)):
        if df_[columnName].iloc[i]<0:
            df_[columnName].replace(df_[columnName].iloc[i],abs(df_[columnName].iloc[i]),inplace=True)
#     dfs=split_dataframe(df_,time=360,show=False)
    dfs=splitting(df_)
    for df in dfs:       
        if len(df)<=window_size:
            if len(df.dropna(subset=[columnName]))>0:
                df=df.dropna(subset=[columnName])
                df['columnName_Mean'] = df[columnName].rolling(window=len(df.dropna(subset=[columnName])), center=True ).mean()   
#                 print('columnName_Mean',df['columnName_Mean'])
                temp=df.dropna(subset=["columnName_Mean"])  
                threshold=1/2*abs(temp['columnName_Mean'].values[0])
                th2=(2)*threshold
                th=abs(temp['columnName_Mean'].values[0])
    #             threshold=(5)*threshold
                for i in range(len(df[columnName])):    
                    
#                     print(df['time'].iloc[i])
#                     print(i,df[columnName].iloc[i])
#                     print(df__[columnName].shift(-1).iloc[df[columnName].index[i]])
#                     print(df__[columnName].iloc[df[columnName].index[i]])
                    if (df[columnName].iloc[i]>th and abs(df[columnName].iloc[i])>quantile and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(1).iloc[df[columnName].index[i]])>=th2 and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(-1).iloc[df[columnName].index[i]])>=th2 and ((df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>0) or (df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<0))) or (df__[columnName].iloc[df[columnName].index[i]]<min_threshold and df__[columnName].iloc[df[columnName].index[i]]<0) or (df__[columnName].iloc[df[columnName].index[i]]>max_threshold and df__[columnName].iloc[df[columnName].index[i]]>0) :
                        indices2.append(int(df["original_index"].iloc[i]))
                        avg_map[int(df["original_index"].iloc[i])]=temp['columnName_Mean'].values[0]
                        indices.append(df[columnName].index[i])
    #                     avg_map[df[columnName].index[i]]=temp['columnName_Mean'].values[0]
                        detected+=1
        else:
            df['columnName_Mean'] = df[columnName].rolling(window=window_size, center=True ).mean()
            for i in range(len(df[columnName])):                   
                margin=int(window_size/2)
                if i<=window_size/2:
                    if len(df.dropna(subset=[columnName]))>0:
                        threshold=(3)*abs(df['columnName_Mean'].iloc[i+margin])
                        th2=(1/2)*threshold
                        th=(4)*abs(df['columnName_Mean'].iloc[i+margin])
    #                     threshold=(2)*threshold    
                        if (df[columnName].iloc[i]>th and abs(df[columnName].iloc[i])>quantile and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(1).iloc[df[columnName].index[i]])>=th2 and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(-1).iloc[df[columnName].index[i]])>=th2 and ((df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>0) or (df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<0))) or (df__[columnName].iloc[df[columnName].index[i]]<min_threshold and df__[columnName].iloc[df[columnName].index[i]]<0) or (df__[columnName].iloc[df[columnName].index[i]]>max_threshold and df__[columnName].iloc[df[columnName].index[i]]>0) :
                            indices2.append(int(df["original_index"].iloc[i]))
                            avg_map[int(df["original_index"].iloc[i])]=df['columnName_Mean'].iloc[i+margin]
                            indices.append(df[columnName].index[i])
    #                         avg_map[df[columnName].index[i]]=df['columnName_Mean'].iloc[i+margin]
                            detected+=1
                else:
                    if i<=df.index[-margin]: 
                        if len(df.dropna(subset=[columnName]))>0:                        
                            threshold=(3)*abs(df['columnName_Mean'].iloc[i-margin])
                            th2=(1/2)*threshold
                            th=(4)*abs(df['columnName_Mean'].iloc[i-margin])
    #                         threshold=(2)*threshold
                            if i>=8971:
                                print(df['time'].iloc[i],df__[columnName].iloc[i])
                                print(i,df[columnName].iloc[i])
                                print(df__[columnName].shift(-1).iloc[df[columnName].index[i]])
                                print(df__[columnName].iloc[df[columnName].index[i]])
                            if (df[columnName].iloc[i]>th and abs(df[columnName].iloc[i])>quantile and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(1).iloc[df[columnName].index[i]])>=th2 and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(-1).iloc[df[columnName].index[i]])>=th2 and ((df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>0) or (df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<0))) or (df__[columnName].iloc[df[columnName].index[i]]<min_threshold and df__[columnName].iloc[df[columnName].index[i]]<0) or (df__[columnName].iloc[df[columnName].index[i]]>max_threshold and df__[columnName].iloc[df[columnName].index[i]]>0) :
                                indices2.append(int(df["original_index"].iloc[i]))
                                avg_map[int(df["original_index"].iloc[i])]=df['columnName_Mean'].iloc[i-margin]                       
                                indices.append(df[columnName].index[i])
    #                             avg_map[df[columnName].index[i]]=df['columnName_Mean'].iloc[i-margin]
                                detected+=1
                    else:
                        if len(df.dropna(subset=[columnName]))>0:
                            threshold=(3)*abs(df['columnName_Mean'].iloc[i])
                            th2=(1/2)*threshold
                            th=(4)*abs(df['columnName_Mean'].iloc[i])
#                         threshold=(2)*threshold
                            print(df[columnName])
                            print(df[columnName].iloc[i])
                            if (df[columnName].iloc[i]>th and abs(df[columnName].iloc[i])>quantile and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(1).iloc[df[columnName].index[i]])>=th2 and abs(df__[columnName].iloc[df[columnName].index[i]]-df__[columnName].shift(-1).iloc[df[columnName].index[i]])>=th2 and ((df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]>0) or (df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<df__[columnName].shift(-1).iloc[df[columnName].index[i]] and df__[columnName].iloc[df[columnName].index[i]]<0))) or (df__[columnName].iloc[df[columnName].index[i]]<min_threshold and df__[columnName].iloc[df[columnName].index[i]]<0) or (df__[columnName].iloc[df[columnName].index[i]]>max_threshold and df__[columnName].iloc[df[columnName].index[i]]>0) :
                                indices2.append(int(df["original_index"].iloc[i]))
                                avg_map[int(df["original_index"].iloc[i])]=df['columnName_Mean'].iloc[i]                       
                                indices.append(df[columnName].index[i])
    #                             avg_map[df[columnName].index[i]]=df['columnName_Mean'].iloc[i]
                                detected+=1
    if show==True :
        plotly.offline.init_notebook_mode(connected=True)
        fig = go.Figure()
#         fig=go.FigureWidget()
        fig.add_trace(go.Scatter(
            y=df__[columnName],
            mode='lines+markers',
            name='Original Plot'
        ))

        fig.add_trace(go.Scatter(
            x=indices,
            y=[df__[columnName][j] for j in indices],
            mode='markers',
            marker=dict(
            size=8,
            color='red',
            symbol='cross'
            ),
            name='Detected Peaks'
        ))
        
        fig.show()
    return [indices,avg_map,detected,indices2]

def mean_peaks_removing_all_peaks_negative_replaced_by_abs(df_,columnName,indices,show=False):        
    df_new=df_.copy()
    df_new["Replaced"]=np.zeros(len(df_new))
    for key in indices:    
        df_new["Replaced"].iloc[key]=1
        df_new.replace(df_new.iloc[key][columnName],np.nan,inplace=True)    
        
    if show==True:
        plotly.offline.init_notebook_mode(connected=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df_[columnName],
            mode='lines+markers',
            name='Original Plot'
        ))

        fig.add_trace(go.Scatter(
            x=indices,
            y=[df[columnName][j] for j in indices],
            mode='markers',
            marker=dict(
            size=8,
            color='red',
            symbol='cross'
            ),
            name='Detected Peaks'
        ))

        fig.show()
    return df_new



    engine = db.create_engine(url) 
    connection = engine.connect()
    metadata = db.MetaData()
    data = db.Table(table_name, metadata, autoload=True, autoload_with=engine)
    
    query = db.insert(data) 
    c=0    
    for df in dfs:
        df = df.astype('object')
        values_list = []
        df=df.rename(columns={'vitesse(m/s)': 'Speed'})
        c+=1
        print("Storing Data frame number: "+str(c))
        for i in range(len(df)-1):
            d=df.iloc[i].to_dict()            
            values_list.append(d)
            
        ResultProxy = connection.execute(query,values_list)        
        
    connection.close()    
    print("#################END##############")