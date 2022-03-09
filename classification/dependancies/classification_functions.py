#!/usr/bin/env python


#imports
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn
import pickle
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import select
from sqlalchemy.sql import text
from pandas import Timestamp
from pandas import Timedelta
from pandas import NaT
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import log
#import skmob
#from skmob.preprocessing import filtering
from math import radians
from math import cos
from math import asin
from math import sqrt
from math import atan2
#from skmob.measures.individual import jump_lengths, radius_of_gyration, home_location
from math import radians, cos, sin, asin, sqrt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statistics import mean
from plotly.subplots import make_subplots
import os
from model_training import *
import matplotlib.patches as mpatches



dictionary_RECORD={'Walk': 1,
 'Bus': 2,
 'Bureau': 3,
 'Restaurant': 4,
 'Domicile': 5,
 'Vélo': 6,
 'Voiture': 7,
 'Magasin': 8,
 'Métro': 9,
 'Gare': 10,
 'Motorcycle': 11,
 'Running': 12,
 'Parc': 13,
 'Indoor': 15,
 'Cinéma': 14,
 'Train': 16,
'No Data': -1}

dictionary_VGP={
    'Rue':1,
    'Bus': 2,
     'Bureau': 3,
     'Restaurant': 4,
     'Domicile': 5,
     'Voiture': 6,
     'Magasin': 7,
     'Train': 8    
}




'''Performs classification for a given participant by providint the participant_virtual_id (for VGP campaign)'''
def classification(participant_virtual_id,model_path='./models/models_RF/'):
    data=get_preprocessed_data(participant_virtual_id=participant_virtual_id)
    data.sort_values('time', inplace=True)
    dfs=splitting(data)
    labels=predict_labels_raw_data_RF(dfs,model_path=model_path,classes_removed=False)
    
    gps=get_Temperature_data_VGP(participant_virtual_id=participant_virtual_id)[["time","lon_1","lat_1","lon_2","lat_2","activity"]]
    tdf = skmob.TrajDataFrame(gps.dropna(subset=["lat_1","lon_1"]), latitude='lat_1', longitude='lon_1',  datetime='time')
    ftdf = filtering.filter(tdf, max_speed_kmh=90)
    
    filename_model=model_path+"Speed_model.sav"
    # load the model from disk
    Speed_model = pickle.load(open(filename_model, 'rb'))
    filename_model=model_path+"BC_model.sav"
    # load the model from disk
    BC_model = pickle.load(open(filename_model, 'rb'))
    
    classes={}
    for key,value in labels.items():
        classes[key]=get_key(most_frequent(list(value)),dictionary={'Rue': 1, 'Bus': 2, 'Bureau': 3, 'Restaurant': 4, 'Domicile': 5, 'Voiture': 6, 'Magasin': 7, 'Train': 8, 'Voiture_arrêt':6, "Domicile_tiers":5, "Train_arrêt":8, "Bureau_tiers":3})
    
    classes = correct_annotation(classes)
    
    c=validate_results(data,classes)
    classes=pd.DataFrame({'participant_virtual_id': participant_virtual_id, 'timestamp': list(classes.keys()), 'activity':list(classes.values())})
    classes['timestamp'] = pd.to_datetime(classes['timestamp'])
    classes.sort_values(['timestamp'], inplace=True)    
    
    new_activities,new_dictionary=correct_annotation_using_GPS(data,classes,ftdf,Speed_model,BC_model,participant_virtual_id=participant_virtual_id)
    if len(new_dictionary)>0:
        c=validate_results(data,new_dictionary)
    print("***********************************************")
    return new_activities




def classification_RECORD(participant_virtual_id,model_path='./models/models_RF/'):
    data=get_Temperature_data_RECORD(participant_virtual_id=participant_virtual_id)
    data.sort_values('time', inplace=True)
    dfs=splitting(data)
    labels=predict_labels_raw_data_RF(dfs,model_path=model_path,classes_removed=False)
    classes={}
    for key,value in labels.items():
        classes[key]=get_key(most_frequent(list(value)),dictionary={'Rue': 1, 'Bus': 2, 'Bureau': 3, 'Restaurant': 4, 'Domicile': 5, 'Voiture': 6, 'Magasin': 7, 'Train': 8, 'Voiture_arrêt':6, "Domicile_tiers":5, "Train_arrêt":8, "Bureau_tiers":3})
    
    classes = correct_annotation(classes)
    c=validate_results(data,classes)
    classes=pd.DataFrame({'participant_virtual_id': participant_virtual_id, 'timestamp': list(classes.keys()), 'activity':list(classes.values())})
    classes['timestamp'] = pd.to_datetime(classes['timestamp'])
    classes.sort_values(['timestamp'], inplace=True)    
    
    return classes




def correct_annotation_VGP(dictionary): ###Classes as dictionary
    new_dictionary={}
#     prev_key=None
    
    for key,value in dictionary.items():
        if value == 'Domicile' or value=='Domicile_tiers':
            dateTime=datetime.strptime(key,'%Y-%m-%d %H:%M:%S')
            weekday=dateTime.weekday()
#             print(weekday)
            if weekday<5:
                time_in_day=dateTime.time()
                if datetime.strptime('08:30:00','%H:%M:%S').time()<=time_in_day and time_in_day<=datetime.strptime('17:30:00','%H:%M:%S').time():
                    new_dictionary[key]='Bureau'
                else:
                    new_dictionary[key]='Domicile'
            else:
                new_dictionary[key]='Domicile'
        else:            
            if value== 'Bureau' or value== 'Bureau_tiers':
                dateTime=datetime.strptime(key,'%Y-%m-%d %H:%M:%S')
                weekday=dateTime.weekday()
                if weekday<5:
                    time_in_day=dateTime.time()
                    if datetime.strptime('08:00:00','%H:%M:%S').time()>=time_in_day or time_in_day>=datetime.strptime('17:30:00','%H:%M:%S').time():
                        new_dictionary[key]='Domicile'
                    else:
                        new_dictionary[key]='Bureau'
                else:
                    new_dictionary[key]='Domicile'
            else:
                if value == "Voiture_arrêt":
                    new_dictionary[key]="Voiture"
                else:
                    if value == "Train_arrêt":
                        new_dictionary[key]="Train"
                    else:                        
                        new_dictionary[key]=value
        
        if value!='Domicile':
            dateTime=datetime.strptime(key,'%Y-%m-%d %H:%M:%S')
            time_in_day=dateTime.time()
            if time_in_day>=datetime.strptime('01:00:00','%H:%M:%S').time() and time_in_day<=datetime.strptime('05:00:00','%H:%M:%S').time():
                new_dictionary[key]='Domicile'
            else:
                new_dictionary[key]=value
    return new_dictionary




def validate_results_RECORD(df,predicted_labels,correct_annotations=False,activities=['Walk', 'Bus', 'Bureau', 'Restaurant', 'Domicile', 'Vélo','Voiture', 'Magasin', 'Métro','Gare','Motorcycle','Running','Parc']):
    real_activities={}
    score=[]
    false=0
    true=0
    if correct_annotations==True:
        predicted_labels=correct_annotation(predicted_labels)
    for key,value in predicted_labels.items():
        activity=df[df["time"]==str(key)]["activity"].to_list()[0]
        real_activities[key]=activity
        if activity in activities:
            if activity==value:
                true+=1
                score.append(1)
            else:
                false+=1
                score.append(0)
    if len(score)>0:
        print("True percentage :",true/len(score))
        print("False percentage :",false/len(score))
    if correct_annotations==True:
        return real_activities,score,predicted_labels
    return real_activities,score




def validate_accuracy_per_class_RECORD(df,predicted_labels,correct_annotations=False,activities=['Walk', 'Bus', 'Bureau', 'Restaurant', 'Domicile', 'Vélo','Voiture', 'Magasin', 'Métro','Gare','Motorcycle','Running','Parc']):
    real_activities={}
    score=[]
    
    score_walk=[]
    score_Bus=[]
    score_Bureau=[]
    score_Restaurant=[]
    score_Domicile=[]
    score_Velo=[]
    score_Voiture=[]
    score_Magasin=[]
    score_Metro=[]
    score_Gare=[]
    score_Motorcycle=[]
    score_Running=[]
    score_Parc=[]
    
    scores={}
    
    false=0
    true=0
    if correct_annotations==True:
        predicted_labels=correct_annotation(predicted_labels)
    for key,value in predicted_labels.items():
        activity=df[df["time"]==str(key)]["activity"].to_list()[0]
        real_activities[key]=activity
        if activity in activities:
            if activity==value:
                true+=1
                score.append(1)
                if activity=='Walk':
                    score_walk.append(1)
                if activity=='Bus':
                    score_Bus.append(1)
                if activity=='Bureau':
                    score_Bureau.append(1)
                if activity=='Restaurant':
                    score_Restaurant.append(1)
                if activity=='Domicile':
                    score_Domicile.append(1)
                if activity=='Vélo':
                    score_Velo.append(1)
                if activity=='Voiture':
                    score_Voiture.append(1)
                if activity=='Magasin':
                    score_Magasin.append(1)
                if activity=='Métro':
                    score_Metro.append(1)
                if activity=='Gare':
                    score_Gare.append(1)
                if activity=='Motorcycle':
                    score_Motorcycle.append(1)
                if activity=='Running':
                    score_Running.append(1)
                if activity=='Parc':
                    score_Parc.append(1)
                
            else:
                false+=1
                score.append(0)
                if activity=='Walk':
                    score_walk.append(0)
                if activity=='Bus':
                    score_Bus.append(0)
                if activity=='Bureau':
                    score_Bureau.append(0)
                if activity=='Restaurant':
                    score_Restaurant.append(0)
                if activity=='Domicile':
                    score_Domicile.append(0)
                if activity=='Vélo':
                    score_Velo.append(0)
                if activity=='Voiture':
                    score_Voiture.append(0)
                if activity=='Magasin':
                    score_Magasin.append(0)
                if activity=='Métro':
                    score_Metro.append(0)
                if activity=='Gare':
                    score_Gare.append(0)
                if activity=='Motorcycle':
                    score_Motorcycle.append(0)
#                     print(value)
#                     print("***************")
                if activity=='Running':
                    score_Running.append(0)
                if activity=='Parc':
                    score_Parc.append(0)
    
    if len(score)>0:
        scores['Accuracy']=true/len(score)
    if len(score_walk)>0:
        scores['Walk']=len([i for i in score_walk if i==1])/len(score_walk)
#         print('Walk',len(score_walk))
    if len(score_Bus)>0:
        scores['Bus']=len([i for i in score_Bus if i==1])/len(score_Bus)
#         print('Bus',len(score_Bus))
    
    if len(score_Bureau)>0:
        scores['Bureau']=len([i for i in score_Bureau if i==1])/len(score_Bureau)
#         print('Bureau',len(score_Bureau))
        
    if len(score_Restaurant)>0:
        scores['Restaurant']=len([i for i in score_Restaurant if i==1])/len(score_Restaurant)
#         print('Restaurant',len(score_Restaurant))
    if len(score_Domicile)>0:
        scores['Domicile']=len([i for i in score_Domicile if i==1])/len(score_Domicile)
#         print('Domicile',len(score_Domicile))
        
    if len(score_Velo)>0:
        scores['Velo']=len([i for i in score_Velo if i==1])/len(score_Velo)
#         print('Velo',len(score_Velo))
    if len(score_Voiture)>0:
        scores['Voiture']=len([i for i in score_Voiture if i==1])/len(score_Voiture)
#         print('Voiture',len(score_Voiture))
    if len(score_Magasin)>0:
        scores['Magasin']=len([i for i in score_Magasin if i==1])/len(score_Magasin)
#         print('Magasin',len(score_Magasin))
    if len(score_Metro)>0:
        scores['Metro']=len([i for i in score_Metro if i==1])/len(score_Metro)
#         print('Metro',len(score_Metro))
    if len(score_Gare)>0:
        scores['Gare']=len([i for i in score_Gare if i==1])/len(score_Gare)
#         print('Gare',len(score_Gare))
    if len(score_Motorcycle)>0:
        scores['Motorcycle']=len([i for i in score_Motorcycle if i==1])/len(score_Motorcycle)
#         print('Motorcycle',len(score_Motorcycle))
    if len(score_Running)>0:
        scores['Running']=len([i for i in score_Running if i==1])/len(score_Running)
#         print('Running',len(score_Running))
    if len(score_Parc)>0:
        scores['Parc']=len([i for i in score_Parc if i==1])/len(score_Parc)
#         print('Parc',len(score_Parc))
            
    print("Participant",df.iloc[0]['participant_virtual_id'])
    for key,value in scores.items():
        if key=='Accuracy':
            print(key,np.round(value,3)*100)
            print('-----------------------')
        else:
            print(key,np.round(value,3)*100)
        
    
    return scores




def calculate_timespent_in_microenvironments_all_over_campaign(df):    
    df.rename(columns={"timestamp":"time"},inplace=True)
    activity_time=calculate_timespent_in_microenvironments(df)
    activity_time_all_campaign={}
    for key,value in activity_time.items():
        summ=None
        for e in value:
            if summ==None:
                summ=e
            else:
                summ+=e
        activity_time_all_campaign[key]=summ
    return activity_time_all_campaign
        



def calculate_timespent_in_microenvironments(df):    
    print("Participant ---------------",df.iloc[0]['participant_virtual_id'])
    start=df.iloc[0]['time']
    activity_time={}
    for activity in list(set(df.dropna(subset=['activity'])['activity'])):
        activity_time[activity]=[]
    for i in range(len(df)):
        if (df.iloc[i]['activity']!=df.shift(-1).iloc[i]['activity'] or (df.shift(-1).iloc[i]['time']-start).days>2) and df.iloc[i]['activity']!=None:
            if (df.shift(-1).iloc[i]['time']-start).days>2:
                activity_time[df.iloc[i]['activity']].append(start-start)
            else:
                activity_time[df.iloc[i]['activity']].append(df.shift(-1).iloc[i]['time']-start)
            start=df.shift(-1).iloc[i]['time']
    return activity_time
        




def calculate_timespent_in_microenvironments_all_over_campaign_predictions(df):
    activities_timespent={}
    for activity in df["activity"]:
        if activity not in activities_timespent.keys():
            activities_timespent[activity]=1
        else:
            activities_timespent[activity]+=1
    for key,value in activities_timespent.items():
        print(key,value)
        activities_timespent[key]=display_time(value*600)
    return activities_timespent






def display_time(seconds, granularity=4):
    intervals = (
    ('w', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
    )
    result = []
    time=[]
    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
            time.append(value)
    return pd.Timedelta(', '.join(result[:granularity]))
    



def splitting(df,time=300):
    df=df.sort_values(by=['time'])
    dfs=[]
    start=0
    start_time=df.iloc[0]["time"]
    for i in range(len(df)):
        if ((df.shift(-1).iloc[i]["time"]-start_time).seconds>=time):
            dfs.append(df.iloc[start:(i+1)])
            start=i+1
            start_time=df.shift(-1).iloc[i]["time"]            
    if start<len(df):
        dfs.append(df[start:-1])
    return dfs




####Plots
def plot_activities(activities,df,plot_Speed=False):
    annotations=get_annotations_for_prediction_time(df=df,activities=activities)
    fig = make_subplots(
        specs=[[{"secondary_y": True}]])  
    
    if plot_Speed:
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['Speed'],
            mode='lines+markers',
            name='Original Plot'
                ),secondary_y=True)
    fig.add_trace(go.Scatter(
                x=activities['timestamp'][activities['activity']=='Walk'],
                y=activities['activity'][activities['activity']=='Walk'],
                mode='markers',
                marker=dict(
                size=8,
                color='red',
                symbol='cross'
                ),
                name='Walk'
            ))
    
    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Walk'],
                y=annotations['activity'][annotations['activity']=='User annotation Walk'],
                mode='markers',
                marker=dict(
                size=8,
                color='red',
                symbol='square'
                ),
                name='Walk Annotation'
            ))
    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Rue'],
                y=annotations['activity'][annotations['activity']=='User annotation Rue'],
                mode='markers',
                marker=dict(
                size=8,
                color='red',
                symbol='square'
                ),
                name='Rue Annotation'
            ))

    fig.add_trace(go.Scatter(
                x=activities['timestamp'][activities['activity']=='Bus'],
                y=activities['activity'][activities['activity']=='Bus'],
                mode='markers',
                marker=dict(
                size=8,
                color='blue',
                symbol='cross'
                ),
                name='Bus'
            ))

    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Bus'],
                y=annotations['activity'][annotations['activity']=='User annotation Bus'],
                mode='markers',
                marker=dict(
                size=8,
                color='blue',
                symbol='square'
                ),
                name='Bus Annotation'
            ))

    fig.add_trace(go.Scatter(
                x=activities['timestamp'][activities['activity']=='Bureau'],
                y=activities['activity'][activities['activity']=='Bureau'],
                mode='markers',
                marker=dict(
                size=8,
                color='green',
                symbol='cross'
                ),
                name='Bureau'
            ))
    
    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Bureau'],
                y=annotations['activity'][annotations['activity']=='User annotation Bureau'],
                mode='markers',
                marker=dict(
                size=8,
                color='green',
                symbol='square'
                ),
                name='Bureau Annotation'
            ))
    

    fig.add_trace(go.Scatter(
                x=activities['timestamp'][activities['activity']=='Restaurant'],
                y=activities['activity'][activities['activity']=='Restaurant'],
                mode='markers',
                marker=dict(
                size=8,
                color='gray',
                symbol='cross'
                ),
                name='Restaurant'
            ))

    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Restaurant'],
                y=annotations['activity'][annotations['activity']=='User annotation Restaurant'],
                mode='markers',
                marker=dict(
                size=8,
                color='gray',
                symbol='square'
                ),
                name='Restaurant Annotation'
            ))
    
    fig.add_trace(go.Scatter(
                x=activities['timestamp'][activities['activity']=='Domicile'],
                y=activities['activity'][activities['activity']=='Domicile'],
                mode='markers',
                marker=dict(
                size=8,
                color='black',
                symbol='cross'
                ),
                name='Domicile'
            ))

    
    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Domicile'],
                y=annotations['activity'][annotations['activity']=='User annotation Domicile'],
                mode='markers',
                marker=dict(
                size=8,
                color='black',
                symbol='square'
                ),
                name='Domicile Annotation'
            ))
    
    fig.add_trace(go.Scatter(
                x=activities['timestamp'][activities['activity']=='Voiture'],
                y=activities['activity'][activities['activity']=='Voiture'],
                mode='markers',
                marker=dict(
                size=8,
                color='purple',
                symbol='cross'
                ),
                name='Voiture'
            ))

    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Voiture'],
                y=annotations['activity'][annotations['activity']=='User annotation Voiture'],
                mode='markers',
                marker=dict(
                size=8,
                color='purple',
                symbol='square'
                ),
                name='Voiture Annotation'
            ))

    fig.add_trace(go.Scatter(
                x=activities['timestamp'][activities['activity']=='Magasin'],
                y=activities['activity'][activities['activity']=='Magasin'],
                mode='markers',
                marker=dict(
                size=8,
                color='white',
                symbol='cross'
                ),
                name='Magasin'
            ))
    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Magasin'],
                y=annotations['activity'][annotations['activity']=='User annotation Magasin'],
                mode='markers',
                marker=dict(
                size=8,
                color='white',
                symbol='square'
                ),
                name='Magasin Annotation'
            ))
    fig.add_trace(go.Scatter(
                x=activities['timestamp'][activities['activity']=='Gare'],
                y=activities['activity'][activities['activity']=='Gare'],
                mode='markers',
                marker=dict(
                size=8,
                color='brown',
                symbol='cross'
                ),
                name='Gare'
            ))
    
    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Gare'],
                y=annotations['activity'][annotations['activity']=='User annotation Gare'],
                mode='markers',
                marker=dict(
                size=8,
                color='brown',
                symbol='square'
                ),
                name='Gare Annotation'
    ))
    
    fig.add_trace(go.Scatter(
                x=activities['timestamp'][activities['activity']=='Métro'],
                y=activities['activity'][activities['activity']=='Métro'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='cross'
                ),
                name='Métro'
            ))
    
    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Métro'],
                y=annotations['activity'][annotations['activity']=='User annotation Métro'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='square'
                ),
                name='Métro Annotation'
            ))
    
    fig.add_trace(go.Scatter(
                x=activities['timestamp'][activities['activity']=='Vélo'],
                y=activities['activity'][activities['activity']=='Vélo'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='cross'
                ),
                name='Vélo'
            ))
    
    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Vélo'],
                y=annotations['activity'][annotations['activity']=='User annotation Vélo'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='square'
                ),
                name='Vélo Annotation'
            ))
    fig.add_trace(go.Scatter(
                x=activities['timestamp'][activities['activity']=='Running'],
                y=activities['activity'][activities['activity']=='Running'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='cross'
                ),
                name='Running'
            ))
        
    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Running'],
                y=annotations['activity'][annotations['activity']=='User annotation Running'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='square'
                ),
                name='Running Annotation'
            ))
    
    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Cinéma'],
                y=annotations['activity'][annotations['activity']=='User annotation Cinéma'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='square'
                ),
                name='Cinéma Annotation'
            ))
    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Train'],
                y=annotations['activity'][annotations['activity']=='User annotation Train'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='square'
                ),
                name='Train Annotation'
            ))
    fig.add_trace(go.Scatter(
                x=activities['timestamp'][activities['activity']=='Motorcycle'],
                y=activities['activity'][activities['activity']=='Motorcycle'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='cross'
                ),
                name='Running'
            ))
    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Motorcycle'],
                y=annotations['activity'][annotations['activity']=='User annotation Motorcycle'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='square'
                ),
                name='Motorcycle Annotation'
            ))
    fig.add_trace(go.Scatter(
                x=activities['timestamp'][activities['activity']=='Parc'],
                y=activities['activity'][activities['activity']=='Parc'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='cross'
                ),
                name='Running'
            ))
    fig.add_trace(go.Scatter(
                x=annotations['timestamp'][annotations['activity']=='User annotation Parc'],
                y=annotations['activity'][annotations['activity']=='User annotation Parc'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='square'
                ),
                name='Motorcycle Annotation'
            ))
    
    fig.update_layout(width=1200, height=600)
    fig.show()



def plot_annotations(df,plot_Speed=False):
#     annotations=get_annotations_for_prediction_time(df=df,activities=activities)
    fig = make_subplots(
        specs=[[{"secondary_y": True}]])  
    
    if plot_Speed:
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['Speed'],
            mode='lines+markers',
            name='Original Plot'
                ),secondary_y=True)
    fig.add_trace(go.Scatter(
                x=df['time'][df['activity']=='Walk'],
                y=df['activity'][df['activity']=='Walk'],
                mode='markers',
                marker=dict(
                size=8,
                color='red',
                symbol='cross'
                ),
                name='Walk'
            ))
    
    
    fig.add_trace(go.Scatter(
                x=df['time'][df['activity']=='Bus'],
                y=df['activity'][df['activity']=='Bus'],
                mode='markers',
                marker=dict(
                size=8,
                color='blue',
                symbol='cross'
                ),
                name='Bus'
            ))

    
    fig.add_trace(go.Scatter(
                x=df['time'][df['activity']=='Bureau'],
                y=df['activity'][df['activity']=='Bureau'],
                mode='markers',
                marker=dict(
                size=8,
                color='green',
                symbol='cross'
                ),
                name='Bureau'
            ))
    
    

    fig.add_trace(go.Scatter(
                x=df['time'][df['activity']=='Restaurant'],
                y=df['activity'][df['activity']=='Restaurant'],
                mode='markers',
                marker=dict(
                size=8,
                color='gray',
                symbol='cross'
                ),
                name='Restaurant'
            ))

    
    fig.add_trace(go.Scatter(
                x=df['time'][df['activity']=='Domicile'],
                y=df['activity'][df['activity']=='Domicile'],
                mode='markers',
                marker=dict(
                size=8,
                color='black',
                symbol='cross'
                ),
                name='Domicile'
            ))

    
    
    fig.add_trace(go.Scatter(
                x=df['time'][df['activity']=='Voiture'],
                y=df['activity'][df['activity']=='Voiture'],
                mode='markers',
                marker=dict(
                size=8,
                color='purple',
                symbol='cross'
                ),
                name='Voiture'
            ))

    
    fig.add_trace(go.Scatter(
                x=df['time'][df['activity']=='Magasin'],
                y=df['activity'][df['activity']=='Magasin'],
                mode='markers',
                marker=dict(
                size=8,
                color='white',
                symbol='cross'
                ),
                name='Magasin'
            ))
    
    
    fig.add_trace(go.Scatter(
                x=df['time'][df['activity']=='Running'],
                y=df['activity'][df['activity']=='Running'],
                mode='markers',
                marker=dict(
                size=8,
                color='brown',
                symbol='cross'
                ),
                name='Running'
            ))
    
    
    fig.add_trace(go.Scatter(
                x=df['time'][df['activity']=='Métro'],
                y=df['activity'][df['activity']=='Métro'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='cross'
                ),
                name='Métro'
            ))
    
    fig.add_trace(go.Scatter(
                x=df['time'][df['activity']=='Vélo'],
                y=df['activity'][df['activity']=='Vélo'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='square'
                ),
                name='Vélo Annotation'
            ))
    
    fig.add_trace(go.Scatter(
                x=df['time'][df['activity']=='Motorcycle'],
                y=df['activity'][df['activity']=='Motorcycle'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='square'
                ),
                name='Motorcycle Annotation'
            ))
    
    fig.add_trace(go.Scatter(
                x=df['time'][df['activity']=='Gare'],
                y=df['activity'][df['activity']=='Gare'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='square'
                ),
                name='Gare Annotation'
            ))
    
    fig.add_trace(go.Scatter(
                x=df['time'][df['activity']=='Parc'],
                y=df['activity'][df['activity']=='Parc'],
                mode='markers',
                marker=dict(
                size=8,
                color='Yellow',
                symbol='square'
                ),
                name='Parc Annotation'
            ))
    
    fig.update_layout(width=1200, height=600)
    fig.show()




def plot_Speed(df):
    fig = make_subplots(
        specs=[[{"secondary_y": True}]])  
    
    if plot_Speed:
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['Speed'],
            mode='lines+markers',
            name='Original Plot'
                ))
    
    fig.update_layout(width=1200, height=600)
    fig.show()




#######




def get_annotations_for_prediction_time(df,activities):    
    timestamps=[]
    annotations=[]
    for i in range(len(activities['timestamp'])):
        timestamps.append(activities['timestamp'].iloc[i])
        annotation=df["activity"][df['time']==activities['timestamp'].iloc[i]].values[0]
        if annotation!=None:
            annotations.append("User annotation "+annotation)
        else:
            annotations.append(annotation)
            
    
    user_annotations=pd.DataFrame({'timestamp': timestamps, 'activity':annotations})
    user_annotations['timestamp'] = pd.to_datetime(user_annotations['timestamp'])
    user_annotations.sort_values(['timestamp'], inplace=True)    
    return user_annotations




def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r




def check_location(lon1, lat1, lon2, lat2,radius):
    a = haversine(lon1, lat1, lon2, lat2)
    if a <= radius:
        return True
    else:
        return False




def correct_annotation_1(df,classes,ftdf,Speed_model=None,BC_model=None,dictionary={'Rue': 1, 'Bus': 2, 'Bureau': 3, 'Restaurant': 4, 'Domicile': 5, 'Voiture': 6, 'Magasin': 7, 'Train': 8, 'Voiture_arrêt':6, "Domicile_tiers":5, "Train_arrêt":8, "Bureau_tiers":3},place="Domicile"): ###Classes as dictionary
    new_dictionary={}
    dfs=splitting(df)
    hl_df = home_location(ftdf[ftdf["activity"]==place])
# print(hl_df.head())
    
    for i in range(len(classes)):
        if len(ftdf[ftdf["datetime"]==classes["time"].iloc[i]]["lng"])>0 and len(ftdf[ftdf["datetime"]==classes["time"].iloc[i]]["lat"])>0:
#             print(classes["timestamp"].iloc[i],classes["activity"].iloc[i])
            is_in_location=check_location(lon1=hl_df["lng"].iloc[0],lat1=hl_df["lat"].iloc[0],lon2=ftdf[ftdf["datetime"]==classes["time"].iloc[i]]["lng"].iloc[0],lat2=ftdf[ftdf["datetime"]==classes["time"].iloc[i]]["lat"].iloc[0])
            if is_in_location==True and classes["activity"].iloc[i]!=place:
                new_dictionary[str(classes["time"].iloc[i])]=place
# #                 print(is_in_location,classes["activity"].iloc[i])
#                 S=dfs[i].dropna(subset=["Speed"])["Speed"].values
#                 BC=dfs[i].dropna(subset=["BC"])["BC"].values
#                 result=[]
#                 if len(S)>0:
# #                     S=calculate_mean_std([S])
# #                     nsamples, nx, ny = [S].shape
# #                     train_Temperature = [S].reshape((nsamples,nx*ny))
#                     s_result=Speed_model.predict([S])
#                     result.append(s_result)
# #                     print("s_result",s_result)
# #                     print("Real annotation",ftdf["activity"].iloc[i])
#                 if len(BC)>0:
# #                     BC=calculate_mean_std([BC])
                    
#                     BC_result=BC_model.predict([BC])
# #                     print("BC_result",BC_result)
#                     result.append(BC_result)
                
#                 if len(result)>0:
#                     if get_key(most_frequent(result),dictionary=dictionary) in ["Domicile","Bureau","Restaurant","Magasin"]:                        
#                         new_dictionary[str(classes["timestamp"].iloc[i])]=place
#                     else:
#                         new_dictionary[str(classes["timestamp"].iloc[i])]=get_key(most_frequent(result),dictionary=dictionary)
                print("Corrected")
            else:
                new_dictionary[str(classes["time"].iloc[i])]=classes["activity"].iloc[i]
        else:
            new_dictionary[str(classes["time"].iloc[i])]=classes["activity"].iloc[i]

#         else:
#             print("No coordinates")
            print("=============================================")
    return new_dictionary




def correct_annotation_using_GPS(df,classes,ftdf,Speed_model=None,BC_model=None,dictionary={'Rue': 1, 'Bus': 2, 'Bureau': 3, 'Restaurant': 4, 'Domicile': 5, 'Voiture': 6, 'Magasin': 7, 'Train': 8, 'Voiture_arrêt':6, "Domicile_tiers":5, "Train_arrêt":8, "Bureau_tiers":3},participant_virtual_id=''):
    new_activities=[]
    new_dictionary={}
    if len(ftdf[ftdf["activity"]=='Domicile'])>=len(ftdf)/4:
        print("Checking Domicile...")
        new_activities=correct_annotation_1(df,classes,ftdf,Speed_model,BC_model,dictionary,place="Domicile")
        new_dictionary=new_activities
        new_activities=pd.DataFrame({'participant_virtual_id': participant_virtual_id, 'time': list(new_activities.keys()), 'activity':list(new_activities.values())})
        new_activities['time'] = pd.to_datetime(new_activities['time'])
        new_activities.sort_values(['time'], inplace=True)    
    
    if len(ftdf[ftdf["activity"]=='Bureau'])>=len(ftdf)/5:
        print("Checking Bureau...")
        if len(new_activities)>0:
            new_activities=correct_annotation_1(df,new_activities,ftdf,Speed_model,BC_model,dictionary,place="Bureau")
        else:
            new_activities=correct_annotation_1(df,classes,ftdf,Speed_model,BC_model,dictionary,place="Bureau")
        
        new_dictionary=new_activities
        new_activities=pd.DataFrame({'participant_virtual_id': participant_virtual_id, 'time': list(new_activities.keys()), 'activity':list(new_activities.values())})
        new_activities['time'] = pd.to_datetime(new_activities['time'])
        new_activities.sort_values(['time'], inplace=True)    
    
    if len(new_activities)==0:
        return classes,new_dictionary
    return new_activities,new_dictionary




####Classification main functions



def prepare_new_dataset_with_speed(predicted_Temperature,predicted_proba_Temperature,predicted_Humidity,predicted_proba_Humidity,predicted_NO2,predicted_proba_NO2,predicted_BC,predicted_proba_BC,predicted_PM1,predicted_proba_PM1,predicted_PM25,predicted_proba_PM25,predicted_PM10,predicted_proba_PM10,predicted_Speed,predicted_proba_Speed,classes_removed=False):    
    train_data_RF=[]
    pt=[]
    ptp=[]
    ph=[]
    php=[]
    pn=[]
    pnp=[]
    pb=[]
    pbp=[]
    pp1=[]
    pp1p=[]
    pp25=[]
    pp25p=[]
    pp10=[]
    pp10p=[]
    ps=[]
    psp=[]
    
    for t,tp,h,hp in zip(predicted_Temperature,predicted_proba_Temperature,predicted_Humidity,predicted_proba_Humidity):
        pt.append(t[0])
        ptp.append(tp[0])
        ph.append(h[0])
        php.append(hp[0])
    
    for n,np in zip(predicted_NO2,predicted_proba_NO2):
        pn.append(n[0])
        pnp.append(np[0])
    
    for b,bp in zip(predicted_BC,predicted_proba_BC):
        pb.append(b[0])
        pbp.append(bp[0])
        
    if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
        for p1,p1p in zip(predicted_PM1,predicted_proba_PM1):
            pp1.append(p1[0])
            pp1p.append(p1p[0])

        for p25,p25p in zip(predicted_PM25,predicted_proba_PM25):
            pp25.append(p25[0])
            pp25p.append(p25p[0])

        for p10,p10p in zip(predicted_PM10,predicted_proba_PM10):
            pp10.append(p10[0])
            pp10p.append(p10p[0])
        
    for s,sp in zip(predicted_Speed,predicted_proba_Speed):
        ps.append(s[0])
        psp.append(sp[0])
        
    
    predicted_Temperature=pt
    predicted_proba_Temperature=ptp
    predicted_Humidity=ph
    predicted_proba_Humidity=php
    predicted_NO2=pn
    predicted_proba_NO2=pnp
    predicted_BC=pb
    predicted_proba_BC=pbp
    predicted_PM1=pp1
    predicted_proba_PM1=pp1p
    predicted_PM25=pp25
    predicted_proba_PM25=pp25p
    predicted_PM10=pp10
    predicted_proba_PM10=pp10p
    predicted_Speed=ps
    predicted_proba_Speed=psp
    
#     if len(predicted_proba_Temperature)>0:
#         l=len(predicted_proba_Temperature)
#     else:
#         if len(predicted_proba_NO2)>0:
#             l=len(predicted_proba_NO2)
    l=max([len(predicted_proba_Temperature),len(predicted_proba_Humidity),len(predicted_proba_NO2),len(predicted_proba_BC),len(predicted_proba_PM1),len(predicted_proba_PM25),len(predicted_proba_PM10),len(predicted_Speed)])
    
    for i in range(l):
        if classes_removed==False:
            a=[]
            if len(predicted_Temperature)>0:
                a.append(predicted_Temperature[i])
            if len(predicted_Humidity)>0:
                a.append(predicted_Humidity[i])
            if len(predicted_NO2)>0:
                a.append(predicted_NO2[i])
            if len(predicted_BC)>0:
                a.append(predicted_BC[i])
            if len(predicted_PM1)>0:
                a.append(predicted_PM1[i])
            if len(predicted_PM25)>0:
                a.append(predicted_PM25[i])
            if len(predicted_PM10)>0:
                a.append(predicted_PM10[i])
            if len(predicted_Speed)>0:
                a.append(predicted_Speed[i])
            
            if len(predicted_Temperature)>0:    
                a.append(predicted_proba_Temperature[i][int(predicted_Temperature[i])-1])
            if len(predicted_Humidity)>0:
                a.append(predicted_proba_Humidity[i][int(predicted_Humidity[i])-1])
            if len(predicted_NO2)>0:
                a.append(predicted_proba_NO2[i][int(predicted_NO2[i])-1])
            if len(predicted_BC)>0:
                a.append(predicted_proba_BC[i][int(predicted_BC[i])-1])
            if len(predicted_PM1)>0:
                a.append(predicted_proba_PM1[i][int(predicted_PM1[i])-1])
            if len(predicted_PM25)>0:
                a.append(predicted_proba_PM25[i][int(predicted_PM25[i])-1])
            if len(predicted_PM10)>0:
                a.append(predicted_proba_PM10[i][int(predicted_PM10[i])-1])    
            if len(predicted_Speed)>0:
                a.append(predicted_proba_Speed[i][int(predicted_Speed[i])-1])    
            train_data_RF.append(a)
    return train_data_RF




def prepare_new_dataset(predicted_Temperature,predicted_proba_Temperature,predicted_Humidity,predicted_proba_Humidity,predicted_NO2,predicted_proba_NO2,predicted_BC,predicted_proba_BC,predicted_PM1,predicted_proba_PM1,predicted_PM25,predicted_proba_PM25,predicted_PM10,predicted_proba_PM10,classes_removed=False):    
    train_data_RF=[]
    pt=[]
    ptp=[]
    ph=[]
    php=[]
    pn=[]
    pnp=[]
    pb=[]
    pbp=[]
    pp1=[]
    pp1p=[]
    pp25=[]
    pp25p=[]
    pp10=[]
    pp10p=[]
    
    for t,tp,h,hp in zip(predicted_Temperature,predicted_proba_Temperature,predicted_Humidity,predicted_proba_Humidity):
        pt.append(t[0])
        ptp.append(tp[0])
        ph.append(h[0])
        php.append(hp[0])
    
    for n,np in zip(predicted_NO2,predicted_proba_NO2):
        pn.append(n[0])
        pnp.append(np[0])
    
    for b,bp in zip(predicted_BC,predicted_proba_BC):
        pb.append(b[0])
        pbp.append(bp[0])
    if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
        for p1,p1p in zip(predicted_PM1,predicted_proba_PM1):
            pp1.append(p1[0])
            pp1p.append(p1p[0])

        for p25,p25p in zip(predicted_PM25,predicted_proba_PM25):
            pp25.append(p25[0])
            pp25p.append(p25p[0])

        for p10,p10p in zip(predicted_PM10,predicted_proba_PM10):
            pp10.append(p10[0])
            pp10p.append(p10p[0])
        
    
    predicted_Temperature=pt
    predicted_proba_Temperature=ptp
    predicted_Humidity=ph
    predicted_proba_Humidity=php
    predicted_NO2=pn
    predicted_proba_NO2=pnp
    predicted_BC=pb
    predicted_proba_BC=pbp
    predicted_PM1=pp1
    predicted_proba_PM1=pp1p
    predicted_PM25=pp25
    predicted_proba_PM25=pp25p
    predicted_PM10=pp10
    predicted_proba_PM10=pp10p

#     if len(predicted_proba_Temperature)>0:
#         l=len(predicted_proba_Temperature)
#     else:        
#         if len(predicted_proba_Humidity)>0:
#             l=len(predicted_proba_NO2)
    l=max([len(predicted_proba_Temperature),len(predicted_proba_Humidity),len(predicted_proba_NO2),len(predicted_proba_BC),len(predicted_proba_PM1),len(predicted_proba_PM25),len(predicted_proba_PM10)])
    
    
    for i in range(l):
        if classes_removed==False:
            a=[]
            if len(predicted_Temperature)>0:
                a.append(predicted_Temperature[i])
            if len(predicted_Humidity)>0:
                a.append(predicted_Humidity[i])
            if len(predicted_NO2)>0:
                a.append(predicted_NO2[i])
            if len(predicted_BC)>0:
                a.append(predicted_BC[i])
            if len(predicted_PM1)>0:
                a.append(predicted_PM1[i])
            if len(predicted_PM25)>0:
                a.append(predicted_PM25[i])
            if len(predicted_PM10)>0:
                a.append(predicted_PM10[i])
                
            if len(predicted_Temperature)>0:
                a.append(predicted_proba_Temperature[i][int(predicted_Temperature[i])-1])
            if len(predicted_Humidity)>0:
                a.append(predicted_proba_Humidity[i][int(predicted_Humidity[i])-1])
            if len(predicted_NO2)>0:
                a.append(predicted_proba_NO2[i][int(predicted_NO2[i])-1])
            if len(predicted_BC)>0:
                a.append(predicted_proba_BC[i][int(predicted_BC[i])-1])
            if len(predicted_PM1)>0:
                a.append(predicted_proba_PM1[i][int(predicted_PM1[i])-1])
            if len(predicted_PM25)>0:
                a.append(predicted_proba_PM25[i][int(predicted_PM25[i])-1])
            if len(predicted_PM10)>0:
                a.append(predicted_proba_PM10[i][int(predicted_PM10[i])-1])    
            train_data_RF.append(a)

    return train_data_RF




def predict_labels_raw_data_RF_(dfs,model_path='./models/new_models/',classes_removed=False):
    Temperature_model,Humidity_model,NO2_model,BC_model,PM1_model,PM25_model,PM10_model=load_models(model_path=model_path,basic_models=True)
    labels_dictionay={}
    labels_Temperature={}
    labels_Humidity={}
    labels_NO2={}
    labels_BC={}
    labels_PM1={}
    labels_PM25={}
    labels_PM10={}
    labels_Speed={}
    z=0

    for df_test in dfs:        
        if len(df_test.dropna(subset=["Speed"]))>1:
            
            Speed_model,multi_view_model,multi_view_model_without_BC,multi_view_model_without_NO2,multi_view_model_without_BC_NO2,multi_view_model_without_PMS,multi_view_model_without_BC_PMS,multi_view_model_only_Temperature_Humidity=load_models(model_path=model_path,with_speed=True)
            
            if len(df_test)>0:
                Temperature_data,Humidity_data,NO2_data,BC_data,PM1_data,PM25_data,PM10_data,Speed_data=prepare_set_with_speed(df_test)
            if len(df_test)>0:

#                 predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,calculate_mean_std(Temperature_data))
                predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,Temperature_data)
                print("Temperature")

#                 predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,calculate_mean_std(Humidity_data))
                predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,Humidity_data)
                print("Humidity")

#                 predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,calculate_mean_std(NO2_data))
                predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,NO2_data)
                print("NO2")

#                 predicted_BC,predicted_proba_BC=predict_view(BC_model,calculate_mean_std(BC_data))
                predicted_BC,predicted_proba_BC=predict_view(BC_model,BC_data)
                print("BC")

#                 predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,calculate_mean_std(PM1_data))
                predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,PM1_data)
                print("PM1.0")

#                 predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,calculate_mean_std(PM25_data))
                predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,PM25_data)
                print("PM2.5")

#                 predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,calculate_mean_std(PM10_data))
                predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,PM10_data)
                print("PM10")

#                 predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,calculate_mean_std(Speed_data))
                predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,Speed_data)
                print("Speed")

                print("counter",z)
                print("===============================================================================")
                z+=1

                new_data = prepare_new_dataset_with_speed(predicted_Temperature,predicted_proba_Temperature,predicted_Humidity,predicted_proba_Humidity,predicted_NO2,predicted_proba_NO2,predicted_BC,predicted_proba_BC,predicted_PM1,predicted_proba_PM1,predicted_PM25,predicted_proba_PM25,predicted_PM10,predicted_proba_PM10,predicted_Speed,predicted_proba_Speed,classes_removed=classes_removed)
                print(new_data)

                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                    labels=multi_view_model.predict(new_data)
                else:
                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                        labels=multi_view_model_without_BC.predict(new_data)
                    else:
                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                            labels=multi_view_model_without_NO2.predict(new_data)
                        else:
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                labels=multi_view_model_without_BC_NO2.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 and len(predicted_Speed)>0:
                                    labels=multi_view_model_without_PMS.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2) and len(predicted_Speed)>0:
                                        print(multi_view_model_without_BC_PMS)
                                        labels=multi_view_model_without_BC_PMS.predict(new_data)
                                    else:                                
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_Speed)>0:
                                            labels=multi_view_model_only_Temperature_Humidity.predict(new_data)


#                 labels_dictionay[str(df_test["time"].iloc[0])]=labels.tolist()
                time=str(df_test["time"].iloc[0])    
                labels_dictionay[time]=labels.tolist()
                
                if len(predicted_Temperature)>0:
                    labels_Temperature[time]=predicted_Temperature[0].tolist()
                else:
                    labels_Temperature[time]=[-1]
                
                if len(predicted_Humidity)>0:
                    labels_Humidity[time]=predicted_Humidity[0].tolist()
                else:
                    labels_Humidity[time]=[-1]
                    
                if len(predicted_NO2)>0:    
                    labels_NO2[time]=predicted_NO2[0].tolist()
                else:
                    labels_NO2[time]=[-1]
                    
                if len(predicted_BC)>0:    
                    labels_BC[time]=predicted_BC[0].tolist()
                else:
                    labels_BC[time]=[-1]
                
                if len(predicted_PM1)>0:    
                    labels_PM1[time]=predicted_PM1[0].tolist()
                else:
                    labels_PM1[time]=[-1]
                
                if len(predicted_PM25)>0:    
                    labels_PM25[time]=predicted_PM25[0].tolist()
                else:
                    labels_PM25[time]=[-1]
                
                if len(predicted_PM10)>0:    
                    labels_PM10[time]=predicted_PM10[0].tolist()
                else:
                    labels_PM10[time]=[-1]
                    
                if len(predicted_Speed)>0:    
                    labels_Speed[time]=predicted_Speed[0].tolist()
                else:
                    labels_Speed[time]=[-1]
#                 labels_dictionay[str(df_test["time"].iloc[0])]=[most_frequent(new_data[0])]
        else:            
            multi_view_model,multi_view_model_without_BC,multi_view_model_without_NO2,multi_view_model_without_BC_NO2,multi_view_model_without_PMS,multi_view_model_without_BC_PMS,multi_view_model_only_Temperature_Humidity=load_models(model_path=model_path,with_speed=False,basic_models=False)
            if len(df_test)>1:
                Temperature_data,Humidity_data,NO2_data,BC_data,PM1_data,PM25_data,PM10_data=prepare_set(df_test)
            if len(df_test)>1:
#                 print("T",calculate_mean_std(Temperature_data))
#                 predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,calculate_mean_std(Temperature_data))
                predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,Temperature_data)
                print("Temperature")
        
#                 predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,calculate_mean_std(Humidity_data))
                predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,Humidity_data)
                print("Humidity")

#                 predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,calculate_mean_std(NO2_data))
                predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,NO2_data)
                print("NO2")

#                 predicted_BC,predicted_proba_BC=predict_view(BC_model,calculate_mean_std(BC_data))
                predicted_BC,predicted_proba_BC=predict_view(BC_model,BC_data)
                print("BC")
                

#                 predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,calculate_mean_std(PM1_data))
                predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,PM1_data)
                print("PM1.0")

#                 predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,calculate_mean_std(PM25_data))
                predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,PM25_data)
                print("PM2.5")

#                 predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,calculate_mean_std(PM10_data))
                predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,PM10_data)
                print("PM10")

#                 predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,Speed_data)
#                 print("Speed")

                print("counter",z)
                print("===============================================================================")
                z+=1

                new_data = prepare_new_dataset(predicted_Temperature,predicted_proba_Temperature,predicted_Humidity,predicted_proba_Humidity,predicted_NO2,predicted_proba_NO2,predicted_BC,predicted_proba_BC,predicted_PM1,predicted_proba_PM1,predicted_PM25,predicted_proba_PM25,predicted_PM10,predicted_proba_PM10,classes_removed=classes_removed)
                print(new_data)

                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                    labels=multi_view_model.predict(new_data)
                else:
                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                        labels=multi_view_model_without_BC.predict(new_data)
                    else:
                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                            labels=multi_view_model_without_NO2.predict(new_data)
                        else:
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                labels=multi_view_model_without_BC_NO2.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 :
                                    labels=multi_view_model_without_PMS.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2):
                                        labels=multi_view_model_without_BC_PMS.predict(new_data)
                                    else:                                
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0:
                                            labels=multi_view_model_only_Temperature_Humidity.predict(new_data)

                time=str(df_test["time"].iloc[0])    
                labels_dictionay[time]=labels.tolist()
                if len(predicted_Temperature)>0:
                    labels_Temperature[time]=predicted_Temperature[0].tolist()
                else:
                    labels_Temperature[time]=[-1]
                
                if len(predicted_Humidity)>0:
                    labels_Humidity[time]=predicted_Humidity[0].tolist()
                else:
                    labels_Humidity[time]=[-1]
                    
                if len(predicted_NO2)>0:    
                    labels_NO2[time]=predicted_NO2[0].tolist()
                else:
                    labels_NO2[time]=[-1]
                    
                if len(predicted_BC)>0:    
                    labels_BC[time]=predicted_BC[0].tolist()
                else:
                    labels_BC[time]=[-1]
                
                if len(predicted_PM1)>0:    
                    labels_PM1[time]=predicted_PM1[0].tolist()
                else:
                    labels_PM1[time]=[-1]
                
                if len(predicted_PM25)>0:    
                    labels_PM25[time]=predicted_PM25[0].tolist()
                else:
                    labels_PM25[time]=[-1]
                
                if len(predicted_PM10)>0:    
                    labels_PM10[time]=predicted_PM10[0].tolist()
                else:
                    labels_PM10[time]=[-1]
                
                labels_Speed[time]=[-1]
                
                
                    
                
#                 labels_dictionay[str(df_test["time"].iloc[0])]=[most_frequent(new_data[0])]
            
    
    return labels_dictionay,labels_Temperature,labels_Humidity,labels_NO2,labels_BC,labels_PM1,labels_PM25,labels_PM10,labels_Speed




def fill_values(df,columns):
    train_Temperature=[]
    train_Humidity=[]
    train_NO2=[]
    train_BC=[]
    train_PM1=[]
    train_PM25=[]
    train_PM10=[]
    train_Speed=[]
    sample_Temperature=[]
    sample_Humidity=[]
    sample_NO2=[]
    sample_BC=[]
    sample_PM1=[]
    sample_PM25=[]
    sample_PM10=[]    
    sample_Speed=[] 
    
    if len(df)>0:
        c=0
        sample_Temperature=[]
        sample_Humidity=[]
        sample_NO2=[]
        sample_BC=[]
        sample_PM1=[]
        sample_PM25=[]
        sample_PM10=[]
        sample_Speed=[]
        
        for i in range(len(df)):
#             print(sample_Temperature)
            if "Temperature" in columns:
                sample_Temperature.append(df["Temperature"].iloc[i])
            if "Humidity" in columns:
                sample_Humidity.append(df["Humidity"].iloc[i])
            if "NO2" in columns:
                sample_NO2.append(df["NO2"].iloc[i])
            if "BC" in columns:
                sample_BC.append(df["BC"].iloc[i])
            if "PM1.0" in columns:
                sample_PM1.append(df["PM1.0"].iloc[i])
            if "PM2.5" in columns:
                sample_PM25.append(df["PM2.5"].iloc[i])
            if "PM10" in columns:
                sample_PM10.append(df["PM10"].iloc[i])
            if "Speed" in columns:
                sample_Speed.append(df["Speed"].iloc[i])
        
#         print("sample_Temperature",sample_Temperature)
        
        if "Temperature" in columns:
            while len(sample_Temperature)!=5:
                print("appending mean")
                sample_Temperature.append(mean(sample_Temperature))
            train_Temperature.append(sample_Temperature)
        if "Humidity" in columns:                    
            while len(sample_Humidity)!=5:
                sample_Humidity.append(mean(sample_Humidity))
            train_Humidity.append(sample_Humidity)
        if "NO2" in columns:
            while len(sample_NO2)!=5:
                sample_NO2.append(mean(sample_NO2))
            train_NO2.append(sample_NO2)
        if "BC" in columns:
            while len(sample_BC)!=5:
                sample_BC.append(mean(sample_BC))
            train_BC.append(sample_BC)
        if "PM1.0" in columns:
            while len(sample_PM1)!=5:
                sample_PM1.append(mean(sample_PM1))
            train_PM1.append(sample_PM1)
        if "PM2.5" in columns:
            while len(sample_PM25)!=5:
                sample_PM25.append(mean(sample_PM25))
            train_PM25.append(sample_PM25)
        if "PM10" in columns:
            while len(sample_PM10)!=5:
                sample_PM10.append(mean(sample_PM10))
            train_PM10.append(sample_PM10)
        if "Speed" in columns:
            while len(sample_Speed)!=5:
                sample_Speed.append(mean(sample_Speed))
            train_Speed.append(sample_Speed)
                
                
#     print(train_Temperature)
        
    return train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed




def prepare_set_(df1):    
    df=df1.dropna(subset=["Temperature","Humidity","NO2","BC","PM1.0","PM2.5","PM10"])
    columns_to_drop=remove_column(df1,columns_to_drop=["Temperature","Humidity","NO2","BC","PM1.0","PM2.5","PM10"])
    if len(df)==1:        
        df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)

    if len(df)>0:
        train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop) 
    else:
        df=df1.dropna(subset=["Temperature","Humidity","NO2","PM1.0","PM2.5","PM10"])            
        columns_to_drop=remove_column(df1,columns_to_drop=["Temperature","Humidity","NO2","PM1.0","PM2.5","PM10"])
        if len(df)==1:
            df=drop_one_row(df,df1,columns_to_drop=["Temperature","Humidity","NO2","PM1.0","PM2.5","PM10"])
        if len(df)>0 and (len(df.dropna(subset=['BC']))==0 or len(df.dropna(subset=['BC']))==1):    
            train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop) 
        else:
            df=df1.dropna(subset=["Temperature","Humidity","BC","PM1.0","PM2.5","PM10"])       
            columns_to_drop=remove_column(df1,columns_to_drop=["Temperature","Humidity","BC","PM1.0","PM2.5","PM10"])
            if len(df)==1:
                df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)
            if len(df)>0 and (len(df.dropna(subset=['NO2']))==0 or len(df.dropna(subset=['NO2']))==1):
                train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop) 
            else:
                df=df1.dropna(subset=["Temperature","Humidity","PM1.0","PM2.5","PM10"])       
                columns_to_drop=remove_column(df1,columns_to_drop=["Temperature","Humidity","PM1.0","PM2.5","PM10"])
                if len(df)==1:
                    df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)
                if len(df)>0 and (len(df.dropna(subset=['NO2','BC']))==0 or len(df.dropna(subset=['NO2','BC']))==1):
                    train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop) 
                else:
                    df=df1.dropna(subset=["Temperature","Humidity","NO2","BC"]) 
                    columns_to_drop=remove_column(df1,columns_to_drop=["Temperature","Humidity","NO2","BC"])
                    if len(df)==1:
                        df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)
                    if len(df)>0 and (len(df.dropna(subset=['PM1.0','PM2.5','PM10']))==0 or len(df.dropna(subset=['PM1.0','PM2.5','PM10']))==1):
                        train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop) 
                    else:
                        df=df1.dropna(subset=["Temperature","Humidity"])     
                        columns_to_drop=remove_column(df1,columns_to_drop=["Temperature","Humidity"])
                        if len(df)==1 and (len(df.dropna(subset=['NO2','BC','PM1.0','PM2.5','PM10']))==0 or len(df.dropna(subset=['NO2','BC','PM1.0','PM2.5','PM10']))==1):
                            df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)
                        p="Temperature and Humidity"
                        if len(df)>0:
                            train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop) 
                        else:
                            df=df1.dropna(subset=["BC","NO2"])    
#                             print(df)
                            columns_to_drop=remove_column(df1,columns_to_drop=["NO2","BC"])
#                             print(columns_to_drop)
                            if len(df)==1 and (len(df.dropna(subset=['Temperature','Humidity','PM1.0','PM2.5','PM10']))==0 or len(df.dropna(subset=['Temperature','Humidity','PM1.0','PM2.5','PM10']))==1):
                                df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)
                            p="BC and NO2"
                            if len(df)>0:
                                train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop) 
                            else:
                                df=df1.dropna(subset=["NO2"])    
#                                 print(df)
                                columns_to_drop=remove_column(df1,columns_to_drop=["NO2"])
                                print(columns_to_drop)
                                if len(df)==1 and (len(df.dropna(subset=['Temperature','Humidity','BC','PM1.0','PM2.5','PM10']))==0 or len(df.dropna(subset=['Temperature','Humidity','BC','PM1.0','PM2.5','PM10']))==1):
                                    df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)
                                p="NO2"
                                if len(df)>0:
                                    train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop) 
                                else:    
                                    print("len(Temp)",df.dropna(subset=['Temperature']))
                                    print("len(Hum)",df.dropna(subset=['Humidity']))
                                    print("len(NO2)",df.dropna(subset=['NO2']))
                                    print("len(BC)",df.dropna(subset=['BC']))
                                    print("len(PM1)",df.dropna(subset=['PM1.0']))
                                    print("len(Speed)",df.dropna(subset=['Speed']))

#     print(p)
#     print("s",train_Temperature)
#     nsamples, nx, ny = train_Temperature.shape
#     train_Temperature = train_Temperature.reshape((nsamples,nx*ny))
    
#     nsamples, nx, ny = train_Humidity.shape
#     train_Humidity = train_Humidity.reshape((nsamples,nx*ny))
    
#     nsamples, nx, ny = train_NO2.shape
#     train_NO2 = train_NO2.reshape((nsamples,nx*ny))
    
#     nsamples, nx, ny = train_BC.shape
#     train_BC = train_BC.reshape((nsamples,nx*ny))
    
#     nsamples, nx, ny = train_PM1.shape
#     train_PM1 = train_PM1.reshape((nsamples,nx*ny))
    
#     nsamples, nx, ny = train_PM25.shape
#     train_PM25 = train_PM25.reshape((nsamples,nx*ny))
    
#     nsamples, nx, ny = train_PM10.shape
#     train_PM10 = train_PM10.reshape((nsamples,nx*ny))
    
    return train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10




def prepare_set_with_speed_(df1):    
    if len(df1.dropna(subset=["Speed"]))>0:
        df=df1.dropna(subset=["Temperature","Humidity","NO2","BC","PM1.0","PM2.5","PM10","Speed"])
        columns_to_drop=remove_column(df1,columns_to_drop=["Temperature","Humidity","NO2","BC","PM1.0","PM2.5","PM10","Speed"])
        if len(df)==1:            
            df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)
    
        if len(df)>0:
            p="all dimensions"
            train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop)
        else:
            df=df1.dropna(subset=["Temperature","Humidity","NO2","PM1.0","PM2.5","PM10","Speed"]) 
            p="BC not found"
            columns_to_drop=remove_column(df1,columns_to_drop=["Temperature","Humidity","NO2","PM1.0","PM2.5","PM10","Speed"])
            if len(df)==1:                
                df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)
                
            if len(df)>0 and (len(df.dropna(subset=['BC']))==0 or len(df.dropna(subset=['BC']))==1):  
                train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop)
            else:
                df=df1.dropna(subset=["Temperature","Humidity","BC","PM1.0","PM2.5","PM10","Speed"])   
                p="NO2 not found"
                columns_to_drop=remove_column(df1,columns_to_drop=["Temperature","Humidity","BC","PM1.0","PM2.5","PM10","Speed"])
                if len(df)==1:
                    df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)
                if len(df)>0 and (len(df.dropna(subset=['NO2']))==0 or len(df.dropna(subset=['NO2']))==1):
                    train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop)
                else:                    
                    df=df1.dropna(subset=["Temperature","Humidity","PM1.0","PM2.5","PM10","Speed"])
                    p="BC and NO2 not found"
                    columns_to_drop=remove_column(df1,columns_to_drop=["Temperature","Humidity","PM1.0","PM2.5","PM10","Speed"])
                    if len(df)==1:
                        df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)
                    if len(df)>0  and (len(df.dropna(subset=['BC','NO2']))==0 or len(df.dropna(subset=['BC','NO2']))==1):
                        train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop)
                    else:
                        df=df1.dropna(subset=["Temperature","Humidity","NO2","BC","Speed"])
                        columns_to_drop=remove_column(df1,columns_to_drop=["Temperature","Humidity","NO2","BC","Speed"])
                        if len(df)==1:
                            df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)
                        p="PMS not found"
                        if len(df)>0 and (len(df.dropna(subset=['PM1.0','PM2.5','PM10']))==0 or len(df.dropna(subset=['PM1.0','PM2.5','PM10']))==1):
                            train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop)
                        else:
                            df=df1.dropna(subset=["Temperature","Humidity","Speed"])   
                            columns_to_drop=remove_column(df1,columns_to_drop=["Temperature","Humidity","Speed"])
                            if len(df)==1:
                                df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)
                            p="Temperature and Humidity"
                            if len(df)>0 and (len(df.dropna(subset=['NO2','BC','PM1.0','PM2.5','PM10']))==0 or len(df.dropna(subset=['NO2','BC','PM1.0','PM2.5','PM10']))==1):
                                train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop) 
                            else:
                                df=df1.dropna(subset=["NO2","BC","Speed"])   
                                columns_to_drop=remove_column(df1,columns_to_drop=["NO2","BC","Speed"])
                                if len(df)==1:
                                    df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)
                                p="NO2 and BC"
                                if len(df)>0 and (len(df.dropna(subset=['Temperature','Humidity','PM1.0','PM2.5','PM10']))==0 or len(df.dropna(subset=['Temperature','Humidity','PM1.0','PM2.5','PM10']))==1):
                                    train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop) 
                                else:
                                    df=df1.dropna(subset=["NO2","Speed"])   
                                    columns_to_drop=remove_column(df1,columns_to_drop=["NO2","Speed"])
                                    if len(df)==1:
                                        df=drop_one_row(df,df1,columns_to_drop=columns_to_drop)
                                    p="NO2"
                                    if len(df)>0 and (len(df.dropna(subset=['Temperature','Humidity',"BC",'PM1.0','PM2.5','PM10']))==0 or len(df.dropna(subset=['Temperature','Humidity',"BC",'PM1.0','PM2.5','PM10']))==1):
                                        train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed=fill_values(df,columns=columns_to_drop) 
                                    else:
                                        print("len(Temp)",df.dropna(subset=['Temperature']))
                                        print("len(Hum)",df.dropna(subset=['Humidity']))
                                        print("len(NO2)",df.dropna(subset=['NO2']))
                                        print("len(BC)",df.dropna(subset=['BC']))
                                        print("len(PM1)",df.dropna(subset=['PM1.0']))
                                        print("len(Speed)",df.dropna(subset=['Speed']))


#     print("p",train_Temperature)
#     nsamples, nx, ny = train_Temperature.shape
#     train_Temperature = train_Temperature.reshape((nsamples,nx*ny))
    
#     nsamples, nx, ny = train_Humidity.shape
#     train_Humidity = train_Humidity.reshape((nsamples,nx*ny))
    
#     nsamples, nx, ny = train_NO2.shape
#     train_NO2 = train_NO2.reshape((nsamples,nx*ny))
    
#     nsamples, nx, ny = train_BC.shape
#     train_BC = train_BC.reshape((nsamples,nx*ny))
    
#     nsamples, nx, ny = train_PM1.shape
#     train_PM1 = train_PM1.reshape((nsamples,nx*ny))
    
#     nsamples, nx, ny = train_PM25.shape
#     train_PM25 = train_PM25.reshape((nsamples,nx*ny))
    
#     nsamples, nx, ny = train_PM10.shape
#     train_PM10 = train_PM10.reshape((nsamples,nx*ny))
    
#     nsamples, nx, ny = train_Speed.shape
#     train_Speed = train_Speed.reshape((nsamples,nx*ny))
    
    return train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed




def load_models(model_path,basic_models=False,with_speed=False):
    if basic_models==True:
        model_names=["Temperature_model.sav","Humidity_model.sav","NO2_model.sav","BC_model.sav","PM1_model.sav","PM25_model.sav","PM10_model.sav"]
        filename_model=model_path+model_names[0]
        # load the model from disk
        Temperature_model = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[1]
        # load the model from disk
        Humidity_model = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[2]
        # load the model from disk
        NO2_model = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[3]
        # load the model from disk
        BC_model = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[4]
        # load the model from disk
        PM1_model = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[5]
        # load the model from disk
        PM25_model = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[6]
        # load the model from disk
        PM10_model = pickle.load(open(filename_model, 'rb'))
        
        return Temperature_model,Humidity_model,NO2_model,BC_model,PM1_model,PM25_model,PM10_model
    
    if with_speed==True:
        model_names=["Speed_model.sav","multi-view_model_with_Speed.sav","multi-view_model_without_BC_with_Speed.sav","multi-view_model_without_NO2_with_Speed.sav","multi-view_model_without_BC_NO2_with_Speed.sav","multi-view_model_only_Temperature_Humidity_with_Speed.sav","multi-view_model_without_PMS_with_Speed.sav","multi-view_model_without_BC_PMS_with_Speed.sav"]
        filename_model=model_path+model_names[0]
        # load the model from disk
        Speed_model = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[1]
        # load the model from disk
        multi_view_model = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[2]
        # load the model from disk
        multi_view_model_without_BC = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[3]
        # load the model from disk
        multi_view_model_without_NO2 = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[4]
        # load the model from disk
        multi_view_model_without_BC_NO2 = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[6]
        # load the model from disk
        multi_view_model_without_PMS = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[7]
        # load the model from disk
        multi_view_model_without_BC_PMS = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[5]
        # load the model from disk
        multi_view_model_only_Temperature_Humidity = pickle.load(open(filename_model, 'rb'))
        
        return Speed_model,multi_view_model,multi_view_model_without_BC,multi_view_model_without_NO2,multi_view_model_without_BC_NO2,multi_view_model_without_PMS,multi_view_model_without_BC_PMS,multi_view_model_only_Temperature_Humidity
    
    
    if with_speed==False and basic_models==False:
        model_names=["multi-view_model.sav","multi-view_model_without_BC.sav","multi-view_model_without_NO2.sav","multi-view_model_without_BC_NO2.sav","multi-view_model_only_Temperature_Humidity.sav","multi-view_model_without_PMS.sav","multi-view_model_without_BC_PMS.sav"]
        
        filename_model=model_path+model_names[0]
        # load the model from disk
        multi_view_model = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[1]
        # load the model from disk
        multi_view_model_without_BC = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[2]
        # load the model from disk
        multi_view_model_without_NO2 = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[3]
        # load the model from disk
        multi_view_model_without_BC_NO2 = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[5]
        # load the model from disk
        multi_view_model_without_PMS = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[6]
        # load the model from disk
        multi_view_model_without_BC_PMS = pickle.load(open(filename_model, 'rb'))

        filename_model=model_path+model_names[4]
        # load the model from disk
        multi_view_model_only_Temperature_Humidity = pickle.load(open(filename_model, 'rb'))
        
        return multi_view_model,multi_view_model_without_BC,multi_view_model_without_NO2,multi_view_model_without_BC_NO2,multi_view_model_without_PMS,multi_view_model_without_BC_PMS,multi_view_model_only_Temperature_Humidity


def load_new_models(model_path,with_speed=False):
    suffix='_with_Speed'
    extension='.sav'
    if with_speed==True:
        model_names=['multi-view_model_only_BC_NO2','multi-view_model_only_BC_NO2_PMS','multi-view_model_only_BC_PMS','multi-view_model_only_NO2_PMS','multi-view_model_only_PMS', 'multi-view_model_without_Temperature','multi-view_model_without_Temperature_NO2','multi-view_model_without_Temperature_BC','multi-view_model_without_Temperature_NO2_BC','multi-view_model_without_Temperature_PMS','multi-view_model_only_Humidity_NO2','multi-view_model_only_Humidity_BC','multi-view_model_without_NO2_PMS','multi-view_model_without_Humidity_PMS','multi-view_model_without_Humidity_BC','multi-view_model_only_Temperature_NO2','multi-view_model_without_Humidity_NO2','multi-view_model_only_Temperature_BC','multi-view_model_only_Temperature_PMS','multi-view_model_only_Humidity_PMS','multi-view_model_only_Humidity','multi-view_model_only_BC','multi-view_model_only_NO2','multi-view_model_only_Temperature']
    else:
        model_names=['multi-view_model_only_BC_NO2','multi-view_model_only_BC_NO2_PMS','multi-view_model_only_BC_PMS','multi-view_model_only_NO2_PMS','multi-view_model_only_PMS', 'multi-view_model_without_Temperature','multi-view_model_without_Temperature_NO2','multi-view_model_without_Temperature_BC','multi-view_model_without_Temperature_NO2_BC','multi-view_model_without_Temperature_PMS','multi-view_model_only_Humidity_NO2','multi-view_model_only_Humidity_BC','multi-view_model_without_NO2_PMS','multi-view_model_without_Humidity_PMS','multi-view_model_without_Humidity_BC','multi-view_model_only_Temperature_NO2','multi-view_model_without_Humidity_NO2','multi-view_model_only_Temperature_BC','multi-view_model_only_Temperature_PMS','multi-view_model_only_Humidity_PMS']
    
    if with_speed==True:
        filename_model=model_path+model_names[0]+suffix+extension
    else:
        filename_model=model_path+model_names[0]+extension
    
    multi_view_model_only_BC_NO2 = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[1]+suffix+extension
    else:
        filename_model=model_path+model_names[1]+extension
    
    multi_view_model_only_BC_NO2_PMS = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[2]+suffix+extension
    else:
        filename_model=model_path+model_names[2]+extension
    
    multi_view_model_only_BC_PMS = pickle.load(open(filename_model, 'rb'))
    
    
    if with_speed==True:
        filename_model=model_path+model_names[3]+suffix+extension
    else:
        filename_model=model_path+model_names[3]+extension
    
    multi_view_model_only_NO2_PMS = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[4]+suffix+extension
    else:
        filename_model=model_path+model_names[4]+extension
    
    multi_view_model_only_PMS = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[5]+suffix+extension
    else:
        filename_model=model_path+model_names[5]+extension
    
    multi_view_model_without_Temperature = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[6]+suffix+extension
    else:
        filename_model=model_path+model_names[6]+extension
    
    multi_view_model_without_Temperature_NO2 = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[7]+suffix+extension
    else:
        filename_model=model_path+model_names[7]+extension
    
    multi_view_model_without_Temperature_BC = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[8]+suffix+extension
    else:
        filename_model=model_path+model_names[8]+extension
    
    multi_view_model_without_Temperature_NO2_BC = pickle.load(open(filename_model, 'rb'))
    
    
    if with_speed==True:
        filename_model=model_path+model_names[9]+suffix+extension
    else:
        filename_model=model_path+model_names[9]+extension
    
    multi_view_model_without_Temperature_PMS = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[10]+suffix+extension
    else:
        filename_model=model_path+model_names[10]+extension
    
    multi_view_model_only_Humidity_NO2 = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[11]+suffix+extension
    else:
        filename_model=model_path+model_names[11]+extension
    
    multi_view_model_only_Humidity_BC = pickle.load(open(filename_model, 'rb'))
    
    
    if with_speed==True:
        filename_model=model_path+model_names[12]+suffix+extension
    else:
        filename_model=model_path+model_names[12]+extension
    
    multi_view_model_without_NO2_PMS = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[13]+suffix+extension
    else:
        filename_model=model_path+model_names[13]+extension
    
    multi_view_model_without_Humidity_PMS = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[14]+suffix+extension
    else:
        filename_model=model_path+model_names[14]+extension
    
    multi_view_model_without_Humidity_BC = pickle.load(open(filename_model, 'rb'))
    
    
    if with_speed==True:
        filename_model=model_path+model_names[15]+suffix+extension
    else:
        filename_model=model_path+model_names[15]+extension
    
    multi_view_model_only_Temperature_NO2 = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[16]+suffix+extension
    else:
        filename_model=model_path+model_names[16]+extension
    
    multi_view_model_without_Humidity_NO2 = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[17]+suffix+extension
    else:
        filename_model=model_path+model_names[17]+extension
    
    multi_view_model_only_Temperature_BC = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[18]+suffix+extension
    else:
        filename_model=model_path+model_names[18]+extension
    
    multi_view_model_only_Temperature_PMS = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[19]+suffix+extension
    else:
        filename_model=model_path+model_names[19]+extension
    
    multi_view_model_only_Humidity_PMS = pickle.load(open(filename_model, 'rb'))
    
    if with_speed==True:
        filename_model=model_path+model_names[20]+suffix+extension
        multi_view_model_only_Humidity_Speed = pickle.load(open(filename_model, 'rb'))
        filename_model=model_path+model_names[21]+suffix+extension
        multi_view_model_only_BC_Speed = pickle.load(open(filename_model, 'rb'))
        filename_model=model_path+model_names[22]+suffix+extension
        multi_view_model_only_NO2_Speed = pickle.load(open(filename_model, 'rb'))
        filename_model=model_path+model_names[23]+suffix+extension
        multi_view_model_only_Temperature_Speed = pickle.load(open(filename_model, 'rb'))
        
    if with_speed:
        return multi_view_model_only_BC_NO2,multi_view_model_only_BC_NO2_PMS,multi_view_model_only_BC_PMS,multi_view_model_only_NO2_PMS,multi_view_model_only_PMS, multi_view_model_without_Temperature,multi_view_model_without_Temperature_NO2,multi_view_model_without_Temperature_BC,multi_view_model_without_Temperature_NO2_BC,multi_view_model_without_Temperature_PMS,multi_view_model_only_Humidity_NO2,multi_view_model_only_Humidity_BC,multi_view_model_without_NO2_PMS,multi_view_model_without_Humidity_PMS,multi_view_model_without_Humidity_BC,multi_view_model_only_Temperature_NO2,multi_view_model_without_Humidity_NO2,multi_view_model_only_Temperature_BC,multi_view_model_only_Temperature_PMS,multi_view_model_only_Humidity_PMS,multi_view_model_only_Humidity_Speed,multi_view_model_only_BC_Speed,multi_view_model_only_NO2_Speed,multi_view_model_only_Temperature_Speed
    return multi_view_model_only_BC_NO2,multi_view_model_only_BC_NO2_PMS,multi_view_model_only_BC_PMS,multi_view_model_only_NO2_PMS,multi_view_model_only_PMS, multi_view_model_without_Temperature,multi_view_model_without_Temperature_NO2,multi_view_model_without_Temperature_BC,multi_view_model_without_Temperature_NO2_BC,multi_view_model_without_Temperature_PMS,multi_view_model_only_Humidity_NO2,multi_view_model_only_Humidity_BC,multi_view_model_without_NO2_PMS,multi_view_model_without_Humidity_PMS,multi_view_model_without_Humidity_BC,multi_view_model_only_Temperature_NO2,multi_view_model_without_Humidity_NO2,multi_view_model_only_Temperature_BC,multi_view_model_only_Temperature_PMS,multi_view_model_only_Humidity_PMS


def correct_predictions_based_on_GPS(classes,gps_data,lon_place,lat_place,radius=0.01,place_Name='Domicile'):

    classes_corrected={}
    for key,value in classes.items():
        lon1=gps_data[gps_data['time']==str(key)]['lon'].values
        lat1=gps_data[gps_data['time']==str(key)]['lat'].values
#         print(lon1,lat1)
        if len(lon1)>0 and len(lat1)>0:
            lon1=lon1[0]
            lat1=lat1[0]        
            if haversine(lon1=lon1,lat1=lat1,lon2=lon_place,lat2=lat_place)<radius:
                classes_corrected[key]=place_Name
            else:
                classes_corrected[key]=value
        else:
            classes_corrected[key]=value
        
        values=list(classes_corrected.values())        
        for i in range(2,len(values)-2):
            if values[i]!=values[i+1] and values[i-1]==values[i+1]:
                values[i]=values[i+1]
            if values[i]=='Bureau' and values[i-1]=='Domicile' and values[i-2]=='Domicile' and values[i+2]=='Domicile':
                values[i]='Domicile'
            
            if values[i]=='Domicile' and values[i-1]=='Bureau' and values[i-2]=='Bureau' and values[i+2]=='Bureau':
                values[i]='Bureau'
    
        for key,value in zip(classes_corrected.keys(),values):
            classes_corrected[key]=value
    
    return classes_corrected




def convert_and_save_DF(labels,participant_virtual_id,name='',path='',classification_type="RF"):
    directory=path+'/'+classification_type+'/'+name+"-view/"+str(participant_virtual_id)
    path_to_file=directory+'/'+name+'-'+classification_type+'.csv'    
    if name=="Multi":
        classes=labels
    else:        
        classes={}
        for key,value in labels.items():
            print(key,value)
            classes[key]=get_key(most_frequent(list(value)),dictionary={'Rue': 1, 'Bus': 2, 'Bureau': 3, 'Restaurant': 4, 'Domicile': 5, 'Voiture': 6, 'Magasin': 7, 'Train': 8, 'Voiture_arrêt':6, "Domicile_tiers":5, "Train_arrêt":8, "Bureau_tiers":3,"No Data":-1})
    classes=pd.DataFrame({'participant_virtual_id': participant_virtual_id, 'time': list(classes.keys()), 'activity':list(classes.values())})
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    
    classes.to_csv(path_to_file,index=False)
    




def convert_predictions_to_DF(classes,participant_virtual_id):
    classes=pd.DataFrame({'participant_virtual_id':participant_virtual_id , 'timestamp': list(classes.keys()), 'activity':list(classes.values())})
    return classes




def validate_results_VGP_annotations(df,predicted_labels,correct_annotations=False,activities=['Walk', 'Bus', 'Bureau', 'Restaurant', 'Domicile', 'Vélo','Voiture', 'Magasin', 'Métro','Gare','Motorcycle','Running','Parc']):
    real_activities={}
    score=[]
    false=0
    true=0
    if correct_annotations==True:
        predicted_labels=correct_annotation(predicted_labels)
    for key,value in predicted_labels.items():
        activity=df[df["time"]==str(key)]["activity"].to_list()[0]
        real_activities[key]=activity
        if activity in activities or activity=='Rue':
            if activity==value or (activity=='Rue' and value=='Walk'):
                true+=1
                score.append(1)
            else:
                false+=1
                score.append(0)
    if len(score)>0:
        print("True percentage :",true/len(score))
        print("False percentage :",false/len(score))
    if correct_annotations==True:
        return real_activities,score,predicted_labels
    return real_activities,score





def validate_results_Indoor_Outdoor_Transport(df,predicted_labels,correct_annotations=False,activities=['Walk', 'Bus', 'Bureau', 'Restaurant', 'Domicile', 'Vélo','Voiture', 'Magasin', 'Métro','Gare','Motorcycle','Running','Parc']):
    real_activities={}
    score=[]
    false=0
    true=0
    if correct_annotations==True:
        predicted_labels=correct_annotation(predicted_labels)
    for key,value in predicted_labels.items():
        activity=df[df["time"]==str(key)]["activity"].to_list()[0]
        real_activities[key]=activity
        if activity in activities or activity=='Rue':
            if ((activity=='Walk' or activity=='Rue' or activity=='Running' or activity=='Parc' or activity=='Vélo') and value=='outdoor') or ((activity=='Bus' or activity=='Voiture' or activity=='Motorcycle' or activity=='Métro') and value=='transport') or ((activity=='Bureau' or activity=='Domicile' or activity=='Restaurant' or activity=='Gare' or activity=='Magasin') and value=='indoor'):
                true+=1
                score.append(1)
            else:
                false+=1
                score.append(0)
    if len(score)>0:
        print("True percentage :",true/len(score))
        print("False percentage :",false/len(score))
    if correct_annotations==True:
        return real_activities,score,predicted_labels
    return real_activities,score




def validate_results_Indoor_Outdoor_Transport_DF(df,df_predictions,correct_annotations=False,activities=['Walk', 'Bus', 'Bureau', 'Restaurant', 'Domicile', 'Vélo','Voiture', 'Magasin', 'Métro','Gare','Motorcycle','Running','Parc']):
    real_activities={}
    score=[]
    false=0
    true=0
    if correct_annotations==True:
        predicted_labels=correct_annotation(predicted_labels)
    for i in range(len(df_predictions)):
        activity=df[df["time"]==df_predictions.iloc[i]['timestamp']]["activity"].to_list()[0]
#         real_activities[key]=activity
        value=df_predictions.iloc[i]['detected_label']        
        if activity in activities or activity=='Rue':
            if ((activity=='Walk' or activity=='Rue' or activity=='Running' or activity=='Parc' or activity=='Vélo') and value=='outdoor') or ((activity=='Bus' or activity=='Voiture' or activity=='Motorcycle' or activity=='Métro') and value=='transport') or ((activity=='Bureau' or activity=='Domicile' or activity=='Restaurant' or activity=='Gare' or activity=='Magasin') and value=='indoor'):
                true+=1
                score.append(1)
            else:
                false+=1
                score.append(0)
    if len(score)>0:
        print("True percentage :",true/len(score))
        print("False percentage :",false/len(score))
    if correct_annotations==True:
        return real_activities,score,predicted_labels
    return real_activities,score




def load_models_Two_Step_Classification(model_path):
    
    model_names_Indoor_without_Speed=["multi-view_model_Indoor.sav","multi-view_model_without_BC_Indoor.sav","multi-view_model_without_NO2_Indoor.sav","multi-view_model_without_BC_NO2_Indoor.sav","multi-view_model_only_Temperature_Humidity_Indoor.sav","multi-view_model_without_PMS_Indoor.sav","multi-view_model_without_BC_PMS_Indoor.sav"]
    model_names_Indoor_with_Speed=["multi-view_model_Indoor_with_Speed.sav","multi-view_model_without_BC_Indoor_with_Speed.sav","multi-view_model_without_NO2_Indoor_with_Speed.sav","multi-view_model_without_BC_NO2_Indoor_with_Speed.sav","multi-view_model_only_Temperature_Humidity_Indoor_with_Speed.sav","multi-view_model_without_PMS_Indoor_with_Speed.sav","multi-view_model_without_BC_PMS_Indoor_with_Speed.sav"]
    
    model_names_Outdoor_without_Speed=["multi-view_model_Outdoor.sav","multi-view_model_without_BC_Outdoor.sav","multi-view_model_without_NO2_Outdoor.sav","multi-view_model_without_BC_NO2_Outdoor.sav","multi-view_model_only_Temperature_Humidity_Outdoor.sav","multi-view_model_without_PMS_Outdoor.sav","multi-view_model_without_BC_PMS_Outdoor.sav"]
    model_names_Outdoor_with_Speed=["multi-view_model_Outdoor_with_Speed.sav","multi-view_model_without_BC_Outdoor_with_Speed.sav","multi-view_model_without_NO2_Outdoor_with_Speed.sav","multi-view_model_without_BC_NO2_Outdoor_with_Speed.sav","multi-view_model_only_Temperature_Humidity_Outdoor_with_Speed.sav","multi-view_model_without_PMS_Outdoor_with_Speed.sav","multi-view_model_without_BC_PMS_Outdoor_with_Speed.sav"]
    
    model_names_Transport_without_Speed=["multi-view_model_Transport.sav","multi-view_model_without_BC_Transport.sav","multi-view_model_without_NO2_Transport.sav","multi-view_model_without_BC_NO2_Transport.sav","multi-view_model_only_Temperature_Humidity_Transport.sav","multi-view_model_without_PMS_Transport.sav","multi-view_model_without_BC_PMS_Transport.sav"]
    model_names_Transport_with_Speed=["multi-view_model_Transport_with_Speed.sav","multi-view_model_without_BC_Transport_with_Speed.sav","multi-view_model_without_NO2_Transport_with_Speed.sav","multi-view_model_without_BC_NO2_Transport_with_Speed.sav","multi-view_model_only_Temperature_Humidity_Transport_with_Speed.sav","multi-view_model_without_PMS_Transport_with_Speed.sav","multi-view_model_without_BC_PMS_Transport_with_Speed.sav"]
    
    #Indoor
    #without Speed
    filename_model=model_path+model_names_Indoor_without_Speed[0]
    # load the model from disk
    multi_view_model_Indoor_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Indoor_without_Speed[1]
    # load the model from disk
    multi_view_model_without_BC_Indoor_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Indoor_without_Speed[2]
    # load the model from disk
    multi_view_model_without_NO2_Indoor_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Indoor_without_Speed[3]
    # load the model from disk
    multi_view_model_without_BC_NO2_Indoor_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Indoor_without_Speed[5]
    # load the model from disk
    multi_view_model_without_PMS_Indoor_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Indoor_without_Speed[6]
    # load the model from disk
    multi_view_model_without_BC_PMS_Indoor_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Indoor_without_Speed[4]
    # load the model from disk
    multi_view_model_only_Temperature_Humidity_Indoor_without_Speed = pickle.load(open(filename_model, 'rb'))
    #########################
    
    #with Speed
    filename_model=model_path+model_names_Indoor_with_Speed[0]
    # load the model from disk
    multi_view_model_Indoor_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Indoor_with_Speed[1]
    # load the model from disk
    multi_view_model_without_BC_Indoor_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Indoor_with_Speed[2]
    # load the model from disk
    multi_view_model_without_NO2_Indoor_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Indoor_with_Speed[3]
    # load the model from disk
    multi_view_model_without_BC_NO2_Indoor_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Indoor_with_Speed[5]
    # load the model from disk
    multi_view_model_without_PMS_Indoor_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Indoor_with_Speed[6]
    # load the model from disk
    multi_view_model_without_BC_PMS_Indoor_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Indoor_with_Speed[4]
    # load the model from disk
    multi_view_model_only_Temperature_Humidity_Indoor_with_Speed = pickle.load(open(filename_model, 'rb'))
    #############################
    
    #Outdoor
    #without Speed
    filename_model=model_path+model_names_Outdoor_without_Speed[0]
    # load the model from disk
    multi_view_model_Outdoor_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Outdoor_without_Speed[1]
    # load the model from disk
    multi_view_model_without_BC_Outdoor_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Outdoor_without_Speed[2]
    # load the model from disk
    multi_view_model_without_NO2_Outdoor_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Outdoor_without_Speed[3]
    # load the model from disk
    multi_view_model_without_BC_NO2_Outdoor_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Outdoor_without_Speed[5]
    # load the model from disk
    multi_view_model_without_PMS_Outdoor_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Outdoor_without_Speed[6]
    # load the model from disk
    multi_view_model_without_BC_PMS_Outdoor_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Outdoor_without_Speed[4]
    # load the model from disk
    multi_view_model_only_Temperature_Humidity_Outdoor_without_Speed = pickle.load(open(filename_model, 'rb'))
    ###################
    
    #with Speed
    filename_model=model_path+model_names_Outdoor_with_Speed[0]
    # load the model from disk
    multi_view_model_Outdoor_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Outdoor_with_Speed[1]
    # load the model from disk
    multi_view_model_without_BC_Outdoor_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Outdoor_with_Speed[2]
    # load the model from disk
    multi_view_model_without_NO2_Outdoor_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Outdoor_with_Speed[3]
    # load the model from disk
    multi_view_model_without_BC_NO2_Outdoor_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Outdoor_with_Speed[5]
    # load the model from disk
    multi_view_model_without_PMS_Outdoor_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Outdoor_with_Speed[6]
    # load the model from disk
    multi_view_model_without_BC_PMS_Outdoor_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Outdoor_with_Speed[4]
    # load the model from disk
    multi_view_model_only_Temperature_Humidity_Outdoor_with_Speed = pickle.load(open(filename_model, 'rb'))
    #######################
    
    #Transport
    #without Speed
    filename_model=model_path+model_names_Transport_without_Speed[0]
    # load the model from disk
    multi_view_model_Transport_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Transport_without_Speed[1]
    # load the model from disk
    multi_view_model_without_BC_Transport_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Transport_without_Speed[2]
    # load the model from disk
    multi_view_model_without_NO2_Transport_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Transport_without_Speed[3]
    # load the model from disk
    multi_view_model_without_BC_NO2_Transport_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Transport_without_Speed[5]
    # load the model from disk
    multi_view_model_without_PMS_Transport_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Transport_without_Speed[6]
    # load the model from disk
    multi_view_model_without_BC_PMS_Transport_without_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Transport_without_Speed[4]
    # load the model from disk
    multi_view_model_only_Temperature_Humidity_Transport_without_Speed = pickle.load(open(filename_model, 'rb'))
    ########################
    
    #with Speed
    filename_model=model_path+model_names_Transport_with_Speed[0]
    # load the model from disk
    multi_view_model_Transport_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Transport_with_Speed[1]
    # load the model from disk
    multi_view_model_without_BC_Transport_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Transport_with_Speed[2]
    # load the model from disk
    multi_view_model_without_NO2_Transport_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Transport_with_Speed[3]
    # load the model from disk
    multi_view_model_without_BC_NO2_Transport_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Transport_with_Speed[5]
    # load the model from disk
    multi_view_model_without_PMS_Transport_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Transport_with_Speed[6]
    # load the model from disk
    multi_view_model_without_BC_PMS_Transport_with_Speed = pickle.load(open(filename_model, 'rb'))

    filename_model=model_path+model_names_Transport_with_Speed[4]
    # load the model from disk
    multi_view_model_only_Temperature_Humidity_Transport_with_Speed = pickle.load(open(filename_model, 'rb'))


    
    return multi_view_model_Indoor_without_Speed,multi_view_model_without_BC_Indoor_without_Speed,multi_view_model_without_NO2_Indoor_without_Speed,multi_view_model_without_BC_NO2_Indoor_without_Speed,multi_view_model_without_PMS_Indoor_without_Speed,multi_view_model_without_BC_PMS_Indoor_without_Speed,multi_view_model_only_Temperature_Humidity_Indoor_without_Speed,multi_view_model_Indoor_with_Speed,multi_view_model_without_BC_Indoor_with_Speed,multi_view_model_without_NO2_Indoor_with_Speed,multi_view_model_without_BC_NO2_Indoor_with_Speed,multi_view_model_without_PMS_Indoor_with_Speed,multi_view_model_without_BC_PMS_Indoor_with_Speed,multi_view_model_only_Temperature_Humidity_Indoor_with_Speed,multi_view_model_Outdoor_without_Speed,multi_view_model_without_BC_Outdoor_without_Speed,multi_view_model_without_NO2_Outdoor_without_Speed,multi_view_model_without_BC_NO2_Outdoor_without_Speed,multi_view_model_without_PMS_Outdoor_without_Speed,multi_view_model_without_BC_PMS_Outdoor_without_Speed,multi_view_model_only_Temperature_Humidity_Outdoor_without_Speed,multi_view_model_Outdoor_with_Speed,multi_view_model_without_BC_Outdoor_with_Speed,multi_view_model_without_NO2_Outdoor_with_Speed,multi_view_model_without_BC_NO2_Outdoor_with_Speed,multi_view_model_without_PMS_Outdoor_with_Speed,multi_view_model_without_BC_PMS_Outdoor_with_Speed,multi_view_model_only_Temperature_Humidity_Outdoor_with_Speed,multi_view_model_Transport_without_Speed,multi_view_model_without_BC_Transport_without_Speed,multi_view_model_without_NO2_Transport_without_Speed,multi_view_model_without_BC_NO2_Transport_without_Speed,multi_view_model_without_PMS_Transport_without_Speed,multi_view_model_without_BC_PMS_Transport_without_Speed,multi_view_model_only_Temperature_Humidity_Transport_without_Speed,multi_view_model_Transport_with_Speed,multi_view_model_without_BC_Transport_with_Speed,multi_view_model_without_NO2_Transport_with_Speed,multi_view_model_without_BC_NO2_Transport_with_Speed,multi_view_model_without_PMS_Transport_with_Speed,multi_view_model_without_BC_PMS_Transport_with_Speed,multi_view_model_only_Temperature_Humidity_Transport_with_Speed




def predict_labels_Indoor_Outdoor_Transport(dfs,classes,model_path='./models/new_models/',classes_removed=False):
    Temperature_model,Humidity_model,NO2_model,BC_model,PM1_model,PM25_model,PM10_model=load_models(model_path=model_path,basic_models=True)
    multi_view_model_Indoor_without_Speed,multi_view_model_without_BC_Indoor_without_Speed,multi_view_model_without_NO2_Indoor_without_Speed,multi_view_model_without_BC_NO2_Indoor_without_Speed,multi_view_model_without_PMS_Indoor_without_Speed,multi_view_model_without_BC_PMS_Indoor_without_Speed,multi_view_model_only_Temperature_Humidity_Indoor_without_Speed,multi_view_model_Indoor_with_Speed,multi_view_model_without_BC_Indoor_with_Speed,multi_view_model_without_NO2_Indoor_with_Speed,multi_view_model_without_BC_NO2_Indoor_with_Speed,multi_view_model_without_PMS_Indoor_with_Speed,multi_view_model_without_BC_PMS_Indoor_with_Speed,multi_view_model_only_Temperature_Humidity_Indoor_with_Speed,multi_view_model_Outdoor_without_Speed,multi_view_model_without_BC_Outdoor_without_Speed,multi_view_model_without_NO2_Outdoor_without_Speed,multi_view_model_without_BC_NO2_Outdoor_without_Speed,multi_view_model_without_PMS_Outdoor_without_Speed,multi_view_model_without_BC_PMS_Outdoor_without_Speed,multi_view_model_only_Temperature_Humidity_Outdoor_without_Speed,multi_view_model_Outdoor_with_Speed,multi_view_model_without_BC_Outdoor_with_Speed,multi_view_model_without_NO2_Outdoor_with_Speed,multi_view_model_without_BC_NO2_Outdoor_with_Speed,multi_view_model_without_PMS_Outdoor_with_Speed,multi_view_model_without_BC_PMS_Outdoor_with_Speed,multi_view_model_only_Temperature_Humidity_Outdoor_with_Speed,multi_view_model_Transport_without_Speed,multi_view_model_without_BC_Transport_without_Speed,multi_view_model_without_NO2_Transport_without_Speed,multi_view_model_without_BC_NO2_Transport_without_Speed,multi_view_model_without_PMS_Transport_without_Speed,multi_view_model_without_BC_PMS_Transport_without_Speed,multi_view_model_only_Temperature_Humidity_Transport_without_Speed,multi_view_model_Transport_with_Speed,multi_view_model_without_BC_Transport_with_Speed,multi_view_model_without_NO2_Transport_with_Speed,multi_view_model_without_BC_NO2_Transport_with_Speed,multi_view_model_without_PMS_Transport_with_Speed,multi_view_model_without_BC_PMS_Transport_with_Speed,multi_view_model_only_Temperature_Humidity_Transport_with_Speed=load_models_Two_Step_Classification(model_path=model_path)
    labels_dictionay={}
    labels_Temperature={}
    labels_Humidity={}
    labels_NO2={}
    labels_BC={}
    labels_PM1={}
    labels_PM25={}
    labels_PM10={}
    labels_Speed={}
    z=0
    i=0
    
    for df_test in dfs:        
        detected_activity=classes.iloc[i]['activity']
        print(df_test.iloc[0]['time'],classes.iloc[i]['timestamp'])
        i+=1
        
        if len(df_test.dropna(subset=["Speed"]))>1:
            
            Speed_model,multi_view_model,multi_view_model_without_BC,multi_view_model_without_NO2,multi_view_model_without_BC_NO2,multi_view_model_without_PMS,multi_view_model_without_BC_PMS,multi_view_model_only_Temperature_Humidity=load_models(model_path=model_path,with_speed=True)
            
            if len(df_test)>0:
                Temperature_data,Humidity_data,NO2_data,BC_data,PM1_data,PM25_data,PM10_data,Speed_data=prepare_set_with_speed(df_test)
            if len(df_test)>0:

#                 predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,calculate_mean_std(Temperature_data))
                predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,Temperature_data)
                print("Temperature")

#                 predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,calculate_mean_std(Humidity_data))
                predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,Humidity_data)
                print("Humidity")

#                 predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,calculate_mean_std(NO2_data))
                predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,NO2_data)
                print("NO2")

#                 predicted_BC,predicted_proba_BC=predict_view(BC_model,calculate_mean_std(BC_data))
                predicted_BC,predicted_proba_BC=predict_view(BC_model,BC_data)
                print("BC")

#                 predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,calculate_mean_std(PM1_data))
                predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,PM1_data)
                print("PM1.0")

#                 predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,calculate_mean_std(PM25_data))
                predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,PM25_data)
                print("PM2.5")

#                 predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,calculate_mean_std(PM10_data))
                predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,PM10_data)
                print("PM10")

#                 predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,calculate_mean_std(Speed_data))
                predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,Speed_data)
                print("Speed")

                print("counter",z)
                print("===============================================================================")
                z+=1

                new_data = prepare_new_dataset_with_speed(predicted_Temperature,predicted_proba_Temperature,predicted_Humidity,predicted_proba_Humidity,predicted_NO2,predicted_proba_NO2,predicted_BC,predicted_proba_BC,predicted_PM1,predicted_proba_PM1,predicted_PM25,predicted_proba_PM25,predicted_PM10,predicted_proba_PM10,predicted_Speed,predicted_proba_Speed,classes_removed=classes_removed)
                print(new_data)
                
                if detected_activity=='indoor':

                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                        labels=multi_view_model_Indoor_with_Speed.predict(new_data)
                    else:
                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                            labels=multi_view_model_without_BC_Indoor_with_Speed.predict(new_data)
                        else:
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                labels=multi_view_model_without_NO2_Indoor_with_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                    labels=multi_view_model_without_BC_NO2_Indoor_with_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 and len(predicted_Speed)>0:
                                        labels=multi_view_model_without_PMS_Indoor_with_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2) and len(predicted_Speed)>0:
                                            print(multi_view_model_without_BC_PMS)
                                            labels=multi_view_model_without_BC_PMS_Indoor_with_Speed.predict(new_data)
                                        else:                                
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_Speed)>0:
                                                labels=multi_view_model_only_Temperature_Humidity_Indoor_with_Speed.predict(new_data)
                else:
                    if detected_activity=='outdoor':
                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                            labels=multi_view_model_Outdoor_with_Speed.predict(new_data)
                        else:
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                labels=multi_view_model_without_BC_Outdoor_with_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:                                        
                                    labels=multi_view_model_without_NO2_Outdoor_with_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                        labels=multi_view_model_without_BC_NO2_Outdoor_with_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 and len(predicted_Speed)>0:
                                            labels=multi_view_model_without_PMS_Outdoor_with_Speed.predict(new_data)
                                        else:
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2) and len(predicted_Speed)>0:
                                                print(multi_view_model_without_BC_PMS)
                                                labels=multi_view_model_without_BC_PMS_Outdoor_with_Speed.predict(new_data)
                                            else:                                
                                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_Speed)>0:
                                                    labels=multi_view_model_only_Temperature_Humidity_Outdoor_with_Speed.predict(new_data)
                    else:
                        if detected_activity=='transport':
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                labels=multi_view_model_Transport_with_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                    labels=multi_view_model_without_BC_Transport_with_Speed.predict(new_data)
                                else:                                        
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                        labels=multi_view_model_without_NO2_Transport_with_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                            labels=multi_view_model_without_BC_NO2_Transport_with_Speed.predict(new_data)
                                        else:
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 and len(predicted_Speed)>0:
                                                labels=multi_view_model_without_PMS_Transport_with_Speed.predict(new_data)
                                            else:
                                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2) and len(predicted_Speed)>0:
                                                    print(multi_view_model_without_BC_PMS)
                                                    labels=multi_view_model_without_BC_PMS_Transport_with_Speed.predict(new_data)
                                                else:                                
                                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_Speed)>0:
                                                        labels=multi_view_model_only_Temperature_Humidity_Transport_with_Speed.predict(new_data)


#                 labels_dictionay[str(df_test["time"].iloc[0])]=labels.tolist()
                time=str(df_test["time"].iloc[0])    
                labels_dictionay[time]=labels.tolist()
                
                if len(predicted_Temperature)>0:
                    labels_Temperature[time]=predicted_Temperature[0].tolist()
                else:
                    labels_Temperature[time]=[-1]
                
                if len(predicted_Humidity)>0:
                    labels_Humidity[time]=predicted_Humidity[0].tolist()
                else:
                    labels_Humidity[time]=[-1]
                    
                if len(predicted_NO2)>0:    
                    labels_NO2[time]=predicted_NO2[0].tolist()
                else:
                    labels_NO2[time]=[-1]
                    
                if len(predicted_BC)>0:    
                    labels_BC[time]=predicted_BC[0].tolist()
                else:
                    labels_BC[time]=[-1]
                
                if len(predicted_PM1)>0:    
                    labels_PM1[time]=predicted_PM1[0].tolist()
                else:
                    labels_PM1[time]=[-1]
                
                if len(predicted_PM25)>0:    
                    labels_PM25[time]=predicted_PM25[0].tolist()
                else:
                    labels_PM25[time]=[-1]
                
                if len(predicted_PM10)>0:    
                    labels_PM10[time]=predicted_PM10[0].tolist()
                else:
                    labels_PM10[time]=[-1]
                    
                if len(predicted_Speed)>0:    
                    labels_Speed[time]=predicted_Speed[0].tolist()
                else:
                    labels_Speed[time]=[-1]
#                 labels_dictionay[str(df_test["time"].iloc[0])]=[most_frequent(new_data[0])]
        else:            
            multi_view_model,multi_view_model_without_BC,multi_view_model_without_NO2,multi_view_model_without_BC_NO2,multi_view_model_without_PMS,multi_view_model_without_BC_PMS,multi_view_model_only_Temperature_Humidity=load_models(model_path=model_path,with_speed=False,basic_models=False)
            if len(df_test)>1:
                Temperature_data,Humidity_data,NO2_data,BC_data,PM1_data,PM25_data,PM10_data=prepare_set(df_test)
            if len(df_test)>1:
#                 print("T",calculate_mean_std(Temperature_data))
#                 predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,calculate_mean_std(Temperature_data))
                predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,Temperature_data)
                print("Temperature")
        
#                 predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,calculate_mean_std(Humidity_data))
                predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,Humidity_data)
                print("Humidity")

#                 predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,calculate_mean_std(NO2_data))
                predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,NO2_data)
                print("NO2")

#                 predicted_BC,predicted_proba_BC=predict_view(BC_model,calculate_mean_std(BC_data))
                predicted_BC,predicted_proba_BC=predict_view(BC_model,BC_data)
                print("BC")
                

#                 predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,calculate_mean_std(PM1_data))
                predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,PM1_data)
                print("PM1.0")

#                 predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,calculate_mean_std(PM25_data))
                predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,PM25_data)
                print("PM2.5")

#                 predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,calculate_mean_std(PM10_data))
                predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,PM10_data)
                print("PM10")

#                 predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,Speed_data)
#                 print("Speed")

                print("counter",z)
                print("===============================================================================")
                z+=1

                new_data = prepare_new_dataset(predicted_Temperature,predicted_proba_Temperature,predicted_Humidity,predicted_proba_Humidity,predicted_NO2,predicted_proba_NO2,predicted_BC,predicted_proba_BC,predicted_PM1,predicted_proba_PM1,predicted_PM25,predicted_proba_PM25,predicted_PM10,predicted_proba_PM10,classes_removed=classes_removed)
                print(new_data)
                
                if detected_activity=='indoor':

                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                        labels=multi_view_model_Indoor_without_Speed.predict(new_data)
                    else:
                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                            labels=multi_view_model_without_BC_Indoor_without_Speed.predict(new_data)
                        else:
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                labels=multi_view_model_without_NO2_Indoor_without_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    labels=multi_view_model_without_BC_NO2_Indoor_without_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 :
                                        labels=multi_view_model_without_PMS_Indoor_without_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2):
                                            labels=multi_view_model_without_BC_PMS_Indoor_without_Speed.predict(new_data)
                                        else:                                
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0:
                                                labels=multi_view_model_only_Temperature_Humidity_Indoor_without_Speed.predict(new_data)
                else:
                    if detected_activity=='outdoor':

                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                            labels=multi_view_model_Outdoor_without_Speed.predict(new_data)
                        else:
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                labels=multi_view_model_without_BC_Outdoor_without_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    labels=multi_view_model_without_NO2_Outdoor_without_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                        labels=multi_view_model_without_BC_NO2_Outdoor_without_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 :
                                            labels=multi_view_model_without_PMS_Outdoor_without_Speed.predict(new_data)
                                        else:
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2):
                                                labels=multi_view_model_without_BC_PMS_Outdoor_without_Speed.predict(new_data)
                                            else:                                
                                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0:
                                                    labels=multi_view_model_only_Temperature_Humidity_Outdoor_without_Speed.predict(new_data)
                    else:
                        if detected_activity=='transport':

                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                labels=multi_view_model_Transport_without_Speed.predict(new_data)
                                print(labels)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    labels=multi_view_model_without_BC_Transport_without_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                        labels=multi_view_model_without_NO2_Transport_without_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                            labels=multi_view_model_without_BC_NO2_Transport_without_Speed.predict(new_data)
                                        else:
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 :
                                                labels=multi_view_model_without_PMS_Transport_without_Speed.predict(new_data)
                                            else:
                                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2):
                                                    labels=multi_view_model_without_BC_PMS_Transport_without_Speed.predict(new_data)
                                                else:                                
                                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0:
                                                        labels=multi_view_model_only_Temperature_Humidity_Transport_without_Speed.predict(new_data)

                time=str(df_test["time"].iloc[0])    
                labels_dictionay[time]=labels.tolist()
                if len(predicted_Temperature)>0:
                    labels_Temperature[time]=predicted_Temperature[0].tolist()
                else:
                    labels_Temperature[time]=[-1]
                
                if len(predicted_Humidity)>0:
                    labels_Humidity[time]=predicted_Humidity[0].tolist()
                else:
                    labels_Humidity[time]=[-1]
                    
                if len(predicted_NO2)>0:    
                    labels_NO2[time]=predicted_NO2[0].tolist()
                else:
                    labels_NO2[time]=[-1]
                    
                if len(predicted_BC)>0:    
                    labels_BC[time]=predicted_BC[0].tolist()
                else:
                    labels_BC[time]=[-1]
                
                if len(predicted_PM1)>0:    
                    labels_PM1[time]=predicted_PM1[0].tolist()
                else:
                    labels_PM1[time]=[-1]
                
                if len(predicted_PM25)>0:    
                    labels_PM25[time]=predicted_PM25[0].tolist()
                else:
                    labels_PM25[time]=[-1]
                
                if len(predicted_PM10)>0:    
                    labels_PM10[time]=predicted_PM10[0].tolist()
                else:
                    labels_PM10[time]=[-1]
                
                labels_Speed[time]=[-1]
                
                
                    
                
#                 labels_dictionay[str(df_test["time"].iloc[0])]=[most_frequent(new_data[0])]
            
    
    return labels_dictionay,labels_Temperature,labels_Humidity,labels_NO2,labels_BC,labels_PM1,labels_PM25,labels_PM10,labels_Speed




def map_prediction_detected_label(prediction):    
    if prediction=='indoor':
        return 0
    if prediction=='outdoor':
        return 2
    if prediction=='transport':
        return 1




def map_activity(activity):
    if activity=='Domicile' or activity=='Bureau' or activity=='Magasin' or activity=='Restaurant' or activity=='Gare' or activity=="Indoor" or activity=='Cinéma':
        return 0
    if activity=='Walk' or activity=='Running' or activity=='Vélo' or activity=='Parc' :
        return 2
    if activity=='Motorcycle' or activity=='Voiture' or activity=='Bus' or activity=='Métro' or activity=='Train':
        return 1




def predict_labels_Indoor_Outdoor_Transport_Ensemble(dfs,classes,ensemble_model,model_path='./models/new_models/',classes_removed=False):
    Temperature_model,Humidity_model,NO2_model,BC_model,PM1_model,PM25_model,PM10_model=load_models(model_path=model_path,basic_models=True)
    multi_view_model_Indoor_without_Speed,multi_view_model_without_BC_Indoor_without_Speed,multi_view_model_without_NO2_Indoor_without_Speed,multi_view_model_without_BC_NO2_Indoor_without_Speed,multi_view_model_without_PMS_Indoor_without_Speed,multi_view_model_without_BC_PMS_Indoor_without_Speed,multi_view_model_only_Temperature_Humidity_Indoor_without_Speed,multi_view_model_Indoor_with_Speed,multi_view_model_without_BC_Indoor_with_Speed,multi_view_model_without_NO2_Indoor_with_Speed,multi_view_model_without_BC_NO2_Indoor_with_Speed,multi_view_model_without_PMS_Indoor_with_Speed,multi_view_model_without_BC_PMS_Indoor_with_Speed,multi_view_model_only_Temperature_Humidity_Indoor_with_Speed,multi_view_model_Outdoor_without_Speed,multi_view_model_without_BC_Outdoor_without_Speed,multi_view_model_without_NO2_Outdoor_without_Speed,multi_view_model_without_BC_NO2_Outdoor_without_Speed,multi_view_model_without_PMS_Outdoor_without_Speed,multi_view_model_without_BC_PMS_Outdoor_without_Speed,multi_view_model_only_Temperature_Humidity_Outdoor_without_Speed,multi_view_model_Outdoor_with_Speed,multi_view_model_without_BC_Outdoor_with_Speed,multi_view_model_without_NO2_Outdoor_with_Speed,multi_view_model_without_BC_NO2_Outdoor_with_Speed,multi_view_model_without_PMS_Outdoor_with_Speed,multi_view_model_without_BC_PMS_Outdoor_with_Speed,multi_view_model_only_Temperature_Humidity_Outdoor_with_Speed,multi_view_model_Transport_without_Speed,multi_view_model_without_BC_Transport_without_Speed,multi_view_model_without_NO2_Transport_without_Speed,multi_view_model_without_BC_NO2_Transport_without_Speed,multi_view_model_without_PMS_Transport_without_Speed,multi_view_model_without_BC_PMS_Transport_without_Speed,multi_view_model_only_Temperature_Humidity_Transport_without_Speed,multi_view_model_Transport_with_Speed,multi_view_model_without_BC_Transport_with_Speed,multi_view_model_without_NO2_Transport_with_Speed,multi_view_model_without_BC_NO2_Transport_with_Speed,multi_view_model_without_PMS_Transport_with_Speed,multi_view_model_without_BC_PMS_Transport_with_Speed,multi_view_model_only_Temperature_Humidity_Transport_with_Speed=load_models_Two_Step_Classification(model_path=model_path)
    labels_dictionay={}
    labels_Temperature={}
    labels_Humidity={}
    labels_NO2={}
    labels_BC={}
    labels_PM1={}
    labels_PM25={}
    labels_PM10={}
    labels_Speed={}
    z=0
    i=0
    
    for df_test in dfs:        
        if str(classes.iloc[i]['detected_label'])=='nan':
            detected_activity=classes.iloc[i]['prediction']
        else:
            print("In ensemble phase")
            ensemble_prediction=ensemble_model.predict(np.array([[map_prediction_detected_label(classes.iloc[i]['prediction']),map_prediction_detected_label(classes.iloc[i]['detected_label'])]]))[0]
            if ensemble_prediction==0:
                detected_activity='indoor'
            else:
                if ensemble_prediction==1:
                    detected_activity='transport'
                else:
                    detected_activity='outdoor'
                    
        print(df_test.iloc[0]['time'],classes.iloc[i]['time'])
        i+=1
        
        if len(df_test.dropna(subset=["Speed"]))>1:
            
            Speed_model,multi_view_model,multi_view_model_without_BC,multi_view_model_without_NO2,multi_view_model_without_BC_NO2,multi_view_model_without_PMS,multi_view_model_without_BC_PMS,multi_view_model_only_Temperature_Humidity=load_models(model_path=model_path,with_speed=True)
            
            if len(df_test)>0:
                Temperature_data,Humidity_data,NO2_data,BC_data,PM1_data,PM25_data,PM10_data,Speed_data=prepare_set_with_speed(df_test)
            if len(df_test)>0:

#                 predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,calculate_mean_std(Temperature_data))
                predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,Temperature_data)
                print("Temperature")

#                 predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,calculate_mean_std(Humidity_data))
                predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,Humidity_data)
                print("Humidity")

#                 predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,calculate_mean_std(NO2_data))
                predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,NO2_data)
                print("NO2")

#                 predicted_BC,predicted_proba_BC=predict_view(BC_model,calculate_mean_std(BC_data))
                predicted_BC,predicted_proba_BC=predict_view(BC_model,BC_data)
                print("BC")

#                 predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,calculate_mean_std(PM1_data))
                predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,PM1_data)
                print("PM1.0")

#                 predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,calculate_mean_std(PM25_data))
                predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,PM25_data)
                print("PM2.5")

#                 predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,calculate_mean_std(PM10_data))
                predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,PM10_data)
                print("PM10")

#                 predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,calculate_mean_std(Speed_data))
                predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,Speed_data)
                print("Speed")

                print("counter",z)
                print("===============================================================================")
                z+=1

                new_data = prepare_new_dataset_with_speed(predicted_Temperature,predicted_proba_Temperature,predicted_Humidity,predicted_proba_Humidity,predicted_NO2,predicted_proba_NO2,predicted_BC,predicted_proba_BC,predicted_PM1,predicted_proba_PM1,predicted_PM25,predicted_proba_PM25,predicted_PM10,predicted_proba_PM10,predicted_Speed,predicted_proba_Speed,classes_removed=classes_removed)
                print(new_data)
                
                if detected_activity=='indoor':

                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                        labels=multi_view_model_Indoor_with_Speed.predict(new_data)
                    else:
                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                            labels=multi_view_model_without_BC_Indoor_with_Speed.predict(new_data)
                        else:
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                labels=multi_view_model_without_NO2_Indoor_with_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                    labels=multi_view_model_without_BC_NO2_Indoor_with_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 and len(predicted_Speed)>0:
                                        labels=multi_view_model_without_PMS_Indoor_with_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2) and len(predicted_Speed)>0:
                                            print(multi_view_model_without_BC_PMS)
                                            labels=multi_view_model_without_BC_PMS_Indoor_with_Speed.predict(new_data)
                                        else:                                
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_Speed)>0:
                                                labels=multi_view_model_only_Temperature_Humidity_Indoor_with_Speed.predict(new_data)
                else:
                    if detected_activity=='outdoor':
                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                            labels=multi_view_model_Outdoor_with_Speed.predict(new_data)
                        else:
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                labels=multi_view_model_without_BC_Outdoor_with_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:                                        
                                    labels=multi_view_model_without_NO2_Outdoor_with_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                        labels=multi_view_model_without_BC_NO2_Outdoor_with_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 and len(predicted_Speed)>0:
                                            labels=multi_view_model_without_PMS_Outdoor_with_Speed.predict(new_data)
                                        else:
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2) and len(predicted_Speed)>0:
                                                print(multi_view_model_without_BC_PMS)
                                                labels=multi_view_model_without_BC_PMS_Outdoor_with_Speed.predict(new_data)
                                            else:                                
                                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_Speed)>0:
                                                    labels=multi_view_model_only_Temperature_Humidity_Outdoor_with_Speed.predict(new_data)
                    else:
                        if detected_activity=='transport':
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                labels=multi_view_model_Transport_with_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                    labels=multi_view_model_without_BC_Transport_with_Speed.predict(new_data)
                                else:                                        
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                        labels=multi_view_model_without_NO2_Transport_with_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                            labels=multi_view_model_without_BC_NO2_Transport_with_Speed.predict(new_data)
                                        else:
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 and len(predicted_Speed)>0:
                                                labels=multi_view_model_without_PMS_Transport_with_Speed.predict(new_data)
                                            else:
                                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2) and len(predicted_Speed)>0:
                                                    print(multi_view_model_without_BC_PMS)
                                                    labels=multi_view_model_without_BC_PMS_Transport_with_Speed.predict(new_data)
                                                else:                                
                                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_Speed)>0:
                                                        labels=multi_view_model_only_Temperature_Humidity_Transport_with_Speed.predict(new_data)


#                 labels_dictionay[str(df_test["time"].iloc[0])]=labels.tolist()
                time=str(df_test["time"].iloc[0])    
                labels_dictionay[time]=labels.tolist()
                
                if len(predicted_Temperature)>0:
                    labels_Temperature[time]=predicted_Temperature[0].tolist()
                else:
                    labels_Temperature[time]=[-1]
                
                if len(predicted_Humidity)>0:
                    labels_Humidity[time]=predicted_Humidity[0].tolist()
                else:
                    labels_Humidity[time]=[-1]
                    
                if len(predicted_NO2)>0:    
                    labels_NO2[time]=predicted_NO2[0].tolist()
                else:
                    labels_NO2[time]=[-1]
                    
                if len(predicted_BC)>0:    
                    labels_BC[time]=predicted_BC[0].tolist()
                else:
                    labels_BC[time]=[-1]
                
                if len(predicted_PM1)>0:    
                    labels_PM1[time]=predicted_PM1[0].tolist()
                else:
                    labels_PM1[time]=[-1]
                
                if len(predicted_PM25)>0:    
                    labels_PM25[time]=predicted_PM25[0].tolist()
                else:
                    labels_PM25[time]=[-1]
                
                if len(predicted_PM10)>0:    
                    labels_PM10[time]=predicted_PM10[0].tolist()
                else:
                    labels_PM10[time]=[-1]
                    
                if len(predicted_Speed)>0:    
                    labels_Speed[time]=predicted_Speed[0].tolist()
                else:
                    labels_Speed[time]=[-1]
#                 labels_dictionay[str(df_test["time"].iloc[0])]=[most_frequent(new_data[0])]
        else:            
            multi_view_model,multi_view_model_without_BC,multi_view_model_without_NO2,multi_view_model_without_BC_NO2,multi_view_model_without_PMS,multi_view_model_without_BC_PMS,multi_view_model_only_Temperature_Humidity=load_models(model_path=model_path,with_speed=False,basic_models=False)
            if len(df_test)>1:
                Temperature_data,Humidity_data,NO2_data,BC_data,PM1_data,PM25_data,PM10_data=prepare_set(df_test)
            if len(df_test)>1:
#                 print("T",calculate_mean_std(Temperature_data))
#                 predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,calculate_mean_std(Temperature_data))
                predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,Temperature_data)
                print("Temperature")
        
#                 predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,calculate_mean_std(Humidity_data))
                predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,Humidity_data)
                print("Humidity")

#                 predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,calculate_mean_std(NO2_data))
                predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,NO2_data)
                print("NO2")

#                 predicted_BC,predicted_proba_BC=predict_view(BC_model,calculate_mean_std(BC_data))
                predicted_BC,predicted_proba_BC=predict_view(BC_model,BC_data)
                print("BC")
                

#                 predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,calculate_mean_std(PM1_data))
                predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,PM1_data)
                print("PM1.0")

#                 predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,calculate_mean_std(PM25_data))
                predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,PM25_data)
                print("PM2.5")

#                 predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,calculate_mean_std(PM10_data))
                predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,PM10_data)
                print("PM10")

#                 predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,Speed_data)
#                 print("Speed")

                print("counter",z)
                print("===============================================================================")
                z+=1

                new_data = prepare_new_dataset(predicted_Temperature,predicted_proba_Temperature,predicted_Humidity,predicted_proba_Humidity,predicted_NO2,predicted_proba_NO2,predicted_BC,predicted_proba_BC,predicted_PM1,predicted_proba_PM1,predicted_PM25,predicted_proba_PM25,predicted_PM10,predicted_proba_PM10,classes_removed=classes_removed)
                print(new_data)
                
                if detected_activity=='indoor':

                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                        labels=multi_view_model_Indoor_without_Speed.predict(new_data)
                    else:
                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                            labels=multi_view_model_without_BC_Indoor_without_Speed.predict(new_data)
                        else:
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                labels=multi_view_model_without_NO2_Indoor_without_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    labels=multi_view_model_without_BC_NO2_Indoor_without_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 :
                                        labels=multi_view_model_without_PMS_Indoor_without_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2):
                                            labels=multi_view_model_without_BC_PMS_Indoor_without_Speed.predict(new_data)
                                        else:                                
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0:
                                                labels=multi_view_model_only_Temperature_Humidity_Indoor_without_Speed.predict(new_data)
                else:
                    if detected_activity=='outdoor':

                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                            labels=multi_view_model_Outdoor_without_Speed.predict(new_data)
                        else:
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                labels=multi_view_model_without_BC_Outdoor_without_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    labels=multi_view_model_without_NO2_Outdoor_without_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                        labels=multi_view_model_without_BC_NO2_Outdoor_without_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 :
                                            labels=multi_view_model_without_PMS_Outdoor_without_Speed.predict(new_data)
                                        else:
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2):
                                                labels=multi_view_model_without_BC_PMS_Outdoor_without_Speed.predict(new_data)
                                            else:                                
                                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0:
                                                    labels=multi_view_model_only_Temperature_Humidity_Outdoor_without_Speed.predict(new_data)
                    else:
                        if detected_activity=='transport':

                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                labels=multi_view_model_Transport_without_Speed.predict(new_data)
                                print(labels)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    labels=multi_view_model_without_BC_Transport_without_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                        labels=multi_view_model_without_NO2_Transport_without_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                            labels=multi_view_model_without_BC_NO2_Transport_without_Speed.predict(new_data)
                                        else:
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 :
                                                labels=multi_view_model_without_PMS_Transport_without_Speed.predict(new_data)
                                            else:
                                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2):
                                                    labels=multi_view_model_without_BC_PMS_Transport_without_Speed.predict(new_data)
                                                else:                                
                                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0:
                                                        labels=multi_view_model_only_Temperature_Humidity_Transport_without_Speed.predict(new_data)

                time=str(df_test["time"].iloc[0])    
                labels_dictionay[time]=labels.tolist()
                if len(predicted_Temperature)>0:
                    labels_Temperature[time]=predicted_Temperature[0].tolist()
                else:
                    labels_Temperature[time]=[-1]
                
                if len(predicted_Humidity)>0:
                    labels_Humidity[time]=predicted_Humidity[0].tolist()
                else:
                    labels_Humidity[time]=[-1]
                    
                if len(predicted_NO2)>0:    
                    labels_NO2[time]=predicted_NO2[0].tolist()
                else:
                    labels_NO2[time]=[-1]
                    
                if len(predicted_BC)>0:    
                    labels_BC[time]=predicted_BC[0].tolist()
                else:
                    labels_BC[time]=[-1]
                
                if len(predicted_PM1)>0:    
                    labels_PM1[time]=predicted_PM1[0].tolist()
                else:
                    labels_PM1[time]=[-1]
                
                if len(predicted_PM25)>0:    
                    labels_PM25[time]=predicted_PM25[0].tolist()
                else:
                    labels_PM25[time]=[-1]
                
                if len(predicted_PM10)>0:    
                    labels_PM10[time]=predicted_PM10[0].tolist()
                else:
                    labels_PM10[time]=[-1]
                
                labels_Speed[time]=[-1]
                
                
                    
                
#                 labels_dictionay[str(df_test["time"].iloc[0])]=[most_frequent(new_data[0])]
            
    
    return labels_dictionay,labels_Temperature,labels_Humidity,labels_NO2,labels_BC,labels_PM1,labels_PM25,labels_PM10,labels_Speed




def predict_labels_Indoor_Outdoor_Transport_Hilbert(dfs,classes,model_path='./models/new_models/',classes_removed=False):
    Temperature_model,Humidity_model,NO2_model,BC_model,PM1_model,PM25_model,PM10_model=load_models(model_path=model_path,basic_models=True)
    multi_view_model_Indoor_without_Speed,multi_view_model_without_BC_Indoor_without_Speed,multi_view_model_without_NO2_Indoor_without_Speed,multi_view_model_without_BC_NO2_Indoor_without_Speed,multi_view_model_without_PMS_Indoor_without_Speed,multi_view_model_without_BC_PMS_Indoor_without_Speed,multi_view_model_only_Temperature_Humidity_Indoor_without_Speed,multi_view_model_Indoor_with_Speed,multi_view_model_without_BC_Indoor_with_Speed,multi_view_model_without_NO2_Indoor_with_Speed,multi_view_model_without_BC_NO2_Indoor_with_Speed,multi_view_model_without_PMS_Indoor_with_Speed,multi_view_model_without_BC_PMS_Indoor_with_Speed,multi_view_model_only_Temperature_Humidity_Indoor_with_Speed,multi_view_model_Outdoor_without_Speed,multi_view_model_without_BC_Outdoor_without_Speed,multi_view_model_without_NO2_Outdoor_without_Speed,multi_view_model_without_BC_NO2_Outdoor_without_Speed,multi_view_model_without_PMS_Outdoor_without_Speed,multi_view_model_without_BC_PMS_Outdoor_without_Speed,multi_view_model_only_Temperature_Humidity_Outdoor_without_Speed,multi_view_model_Outdoor_with_Speed,multi_view_model_without_BC_Outdoor_with_Speed,multi_view_model_without_NO2_Outdoor_with_Speed,multi_view_model_without_BC_NO2_Outdoor_with_Speed,multi_view_model_without_PMS_Outdoor_with_Speed,multi_view_model_without_BC_PMS_Outdoor_with_Speed,multi_view_model_only_Temperature_Humidity_Outdoor_with_Speed,multi_view_model_Transport_without_Speed,multi_view_model_without_BC_Transport_without_Speed,multi_view_model_without_NO2_Transport_without_Speed,multi_view_model_without_BC_NO2_Transport_without_Speed,multi_view_model_without_PMS_Transport_without_Speed,multi_view_model_without_BC_PMS_Transport_without_Speed,multi_view_model_only_Temperature_Humidity_Transport_without_Speed,multi_view_model_Transport_with_Speed,multi_view_model_without_BC_Transport_with_Speed,multi_view_model_without_NO2_Transport_with_Speed,multi_view_model_without_BC_NO2_Transport_with_Speed,multi_view_model_without_PMS_Transport_with_Speed,multi_view_model_without_BC_PMS_Transport_with_Speed,multi_view_model_only_Temperature_Humidity_Transport_with_Speed=load_models_Two_Step_Classification(model_path=model_path)
    labels_dictionay={}
    labels_Temperature={}
    labels_Humidity={}
    labels_NO2={}
    labels_BC={}
    labels_PM1={}
    labels_PM25={}
    labels_PM10={}
    labels_Speed={}
    z=0
    i=0
    
    for df_test in dfs:        
        if str(classes.iloc[i]['detected_label'])=='nan':
            detected_activity=classes.iloc[i]['prediction']
        else:
            detected_activity=classes.iloc[i]['detected_label']
                    
        print(df_test.iloc[0]['time'],classes.iloc[i]['time'])
        i+=1
        
        if len(df_test.dropna(subset=["Speed"]))>1:
            
            Speed_model,multi_view_model,multi_view_model_without_BC,multi_view_model_without_NO2,multi_view_model_without_BC_NO2,multi_view_model_without_PMS,multi_view_model_without_BC_PMS,multi_view_model_only_Temperature_Humidity=load_models(model_path=model_path,with_speed=True)
            
            if len(df_test)>0:
                Temperature_data,Humidity_data,NO2_data,BC_data,PM1_data,PM25_data,PM10_data,Speed_data=prepare_set_with_speed(df_test)
            if len(df_test)>0:

#                 predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,calculate_mean_std(Temperature_data))
                predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,Temperature_data)
                print("Temperature")

#                 predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,calculate_mean_std(Humidity_data))
                predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,Humidity_data)
                print("Humidity")

#                 predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,calculate_mean_std(NO2_data))
                predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,NO2_data)
                print("NO2")

#                 predicted_BC,predicted_proba_BC=predict_view(BC_model,calculate_mean_std(BC_data))
                predicted_BC,predicted_proba_BC=predict_view(BC_model,BC_data)
                print("BC")

#                 predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,calculate_mean_std(PM1_data))
                predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,PM1_data)
                print("PM1.0")

#                 predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,calculate_mean_std(PM25_data))
                predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,PM25_data)
                print("PM2.5")

#                 predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,calculate_mean_std(PM10_data))
                predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,PM10_data)
                print("PM10")

#                 predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,calculate_mean_std(Speed_data))
                predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,Speed_data)
                print("Speed")

                print("counter",z)
                print("===============================================================================")
                z+=1

                new_data = prepare_new_dataset_with_speed(predicted_Temperature,predicted_proba_Temperature,predicted_Humidity,predicted_proba_Humidity,predicted_NO2,predicted_proba_NO2,predicted_BC,predicted_proba_BC,predicted_PM1,predicted_proba_PM1,predicted_PM25,predicted_proba_PM25,predicted_PM10,predicted_proba_PM10,predicted_Speed,predicted_proba_Speed,classes_removed=classes_removed)
                print(new_data)
                
                if detected_activity=='indoor':

                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                        labels=multi_view_model_Indoor_with_Speed.predict(new_data)
                    else:
                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                            labels=multi_view_model_without_BC_Indoor_with_Speed.predict(new_data)
                        else:
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                labels=multi_view_model_without_NO2_Indoor_with_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                    labels=multi_view_model_without_BC_NO2_Indoor_with_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 and len(predicted_Speed)>0:
                                        labels=multi_view_model_without_PMS_Indoor_with_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2) and len(predicted_Speed)>0:
                                            print(multi_view_model_without_BC_PMS)
                                            labels=multi_view_model_without_BC_PMS_Indoor_with_Speed.predict(new_data)
                                        else:                                
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_Speed)>0:
                                                labels=multi_view_model_only_Temperature_Humidity_Indoor_with_Speed.predict(new_data)
                else:
                    if detected_activity=='outdoor':
                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                            labels=multi_view_model_Outdoor_with_Speed.predict(new_data)
                        else:
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                labels=multi_view_model_without_BC_Outdoor_with_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:                                        
                                    labels=multi_view_model_without_NO2_Outdoor_with_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                        labels=multi_view_model_without_BC_NO2_Outdoor_with_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 and len(predicted_Speed)>0:
                                            labels=multi_view_model_without_PMS_Outdoor_with_Speed.predict(new_data)
                                        else:
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2) and len(predicted_Speed)>0:
                                                print(multi_view_model_without_BC_PMS)
                                                labels=multi_view_model_without_BC_PMS_Outdoor_with_Speed.predict(new_data)
                                            else:                                
                                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_Speed)>0:
                                                    labels=multi_view_model_only_Temperature_Humidity_Outdoor_with_Speed.predict(new_data)
                    else:
                        if detected_activity=='transport':
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                labels=multi_view_model_Transport_with_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                    labels=multi_view_model_without_BC_Transport_with_Speed.predict(new_data)
                                else:                                        
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                        labels=multi_view_model_without_NO2_Transport_with_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0 and len(predicted_Speed)>0:
                                            labels=multi_view_model_without_BC_NO2_Transport_with_Speed.predict(new_data)
                                        else:
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 and len(predicted_Speed)>0:
                                                labels=multi_view_model_without_PMS_Transport_with_Speed.predict(new_data)
                                            else:
                                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2) and len(predicted_Speed)>0:
                                                    print(multi_view_model_without_BC_PMS)
                                                    labels=multi_view_model_without_BC_PMS_Transport_with_Speed.predict(new_data)
                                                else:                                
                                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_Speed)>0:
                                                        labels=multi_view_model_only_Temperature_Humidity_Transport_with_Speed.predict(new_data)


#                 labels_dictionay[str(df_test["time"].iloc[0])]=labels.tolist()
                time=str(df_test["time"].iloc[0])    
                labels_dictionay[time]=labels.tolist()
                
                if len(predicted_Temperature)>0:
                    labels_Temperature[time]=predicted_Temperature[0].tolist()
                else:
                    labels_Temperature[time]=[-1]
                
                if len(predicted_Humidity)>0:
                    labels_Humidity[time]=predicted_Humidity[0].tolist()
                else:
                    labels_Humidity[time]=[-1]
                    
                if len(predicted_NO2)>0:    
                    labels_NO2[time]=predicted_NO2[0].tolist()
                else:
                    labels_NO2[time]=[-1]
                    
                if len(predicted_BC)>0:    
                    labels_BC[time]=predicted_BC[0].tolist()
                else:
                    labels_BC[time]=[-1]
                
                if len(predicted_PM1)>0:    
                    labels_PM1[time]=predicted_PM1[0].tolist()
                else:
                    labels_PM1[time]=[-1]
                
                if len(predicted_PM25)>0:    
                    labels_PM25[time]=predicted_PM25[0].tolist()
                else:
                    labels_PM25[time]=[-1]
                
                if len(predicted_PM10)>0:    
                    labels_PM10[time]=predicted_PM10[0].tolist()
                else:
                    labels_PM10[time]=[-1]
                    
                if len(predicted_Speed)>0:    
                    labels_Speed[time]=predicted_Speed[0].tolist()
                else:
                    labels_Speed[time]=[-1]
#                 labels_dictionay[str(df_test["time"].iloc[0])]=[most_frequent(new_data[0])]
        else:            
            multi_view_model,multi_view_model_without_BC,multi_view_model_without_NO2,multi_view_model_without_BC_NO2,multi_view_model_without_PMS,multi_view_model_without_BC_PMS,multi_view_model_only_Temperature_Humidity=load_models(model_path=model_path,with_speed=False,basic_models=False)
            if len(df_test)>1:
                Temperature_data,Humidity_data,NO2_data,BC_data,PM1_data,PM25_data,PM10_data=prepare_set(df_test)
            if len(df_test)>1:
#                 print("T",calculate_mean_std(Temperature_data))
#                 predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,calculate_mean_std(Temperature_data))
                predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,Temperature_data)
                print("Temperature")
        
#                 predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,calculate_mean_std(Humidity_data))
                predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,Humidity_data)
                print("Humidity")

#                 predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,calculate_mean_std(NO2_data))
                predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,NO2_data)
                print("NO2")

#                 predicted_BC,predicted_proba_BC=predict_view(BC_model,calculate_mean_std(BC_data))
                predicted_BC,predicted_proba_BC=predict_view(BC_model,BC_data)
                print("BC")
                

#                 predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,calculate_mean_std(PM1_data))
                predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,PM1_data)
                print("PM1.0")

#                 predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,calculate_mean_std(PM25_data))
                predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,PM25_data)
                print("PM2.5")

#                 predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,calculate_mean_std(PM10_data))
                predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,PM10_data)
                print("PM10")

#                 predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,Speed_data)
#                 print("Speed")

                print("counter",z)
                print("===============================================================================")
                z+=1

                new_data = prepare_new_dataset(predicted_Temperature,predicted_proba_Temperature,predicted_Humidity,predicted_proba_Humidity,predicted_NO2,predicted_proba_NO2,predicted_BC,predicted_proba_BC,predicted_PM1,predicted_proba_PM1,predicted_PM25,predicted_proba_PM25,predicted_PM10,predicted_proba_PM10,classes_removed=classes_removed)
                print(new_data)
                
                if detected_activity=='indoor':

                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                        labels=multi_view_model_Indoor_without_Speed.predict(new_data)
                    else:
                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                            labels=multi_view_model_without_BC_Indoor_without_Speed.predict(new_data)
                        else:
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                labels=multi_view_model_without_NO2_Indoor_without_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    labels=multi_view_model_without_BC_NO2_Indoor_without_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 :
                                        labels=multi_view_model_without_PMS_Indoor_without_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2):
                                            labels=multi_view_model_without_BC_PMS_Indoor_without_Speed.predict(new_data)
                                        else:                                
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0:
                                                labels=multi_view_model_only_Temperature_Humidity_Indoor_without_Speed.predict(new_data)
                else:
                    if detected_activity=='outdoor':

                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                            labels=multi_view_model_Outdoor_without_Speed.predict(new_data)
                        else:
                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                labels=multi_view_model_without_BC_Outdoor_without_Speed.predict(new_data)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    labels=multi_view_model_without_NO2_Outdoor_without_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                        labels=multi_view_model_without_BC_NO2_Outdoor_without_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 :
                                            labels=multi_view_model_without_PMS_Outdoor_without_Speed.predict(new_data)
                                        else:
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2):
                                                labels=multi_view_model_without_BC_PMS_Outdoor_without_Speed.predict(new_data)
                                            else:                                
                                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0:
                                                    labels=multi_view_model_only_Temperature_Humidity_Outdoor_without_Speed.predict(new_data)
                    else:
                        if detected_activity=='transport':

                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                labels=multi_view_model_Transport_without_Speed.predict(new_data)
                                print(labels)
                            else:
                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    labels=multi_view_model_without_BC_Transport_without_Speed.predict(new_data)
                                else:
                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                        labels=multi_view_model_without_NO2_Transport_without_Speed.predict(new_data)
                                    else:
                                        if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                            labels=multi_view_model_without_BC_NO2_Transport_without_Speed.predict(new_data)
                                        else:
                                            if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_BC)>0 and len(predicted_NO2)>0 :
                                                labels=multi_view_model_without_PMS_Transport_without_Speed.predict(new_data)
                                            else:
                                                if len(predicted_Temperature)>0 and len(predicted_Humidity)>0 and len(predicted_NO2):
                                                    labels=multi_view_model_without_BC_PMS_Transport_without_Speed.predict(new_data)
                                                else:                                
                                                    if len(predicted_Temperature)>0 and len(predicted_Humidity)>0:
                                                        labels=multi_view_model_only_Temperature_Humidity_Transport_without_Speed.predict(new_data)

                time=str(df_test["time"].iloc[0])    
                labels_dictionay[time]=labels.tolist()
                if len(predicted_Temperature)>0:
                    labels_Temperature[time]=predicted_Temperature[0].tolist()
                else:
                    labels_Temperature[time]=[-1]
                
                if len(predicted_Humidity)>0:
                    labels_Humidity[time]=predicted_Humidity[0].tolist()
                else:
                    labels_Humidity[time]=[-1]
                    
                if len(predicted_NO2)>0:    
                    labels_NO2[time]=predicted_NO2[0].tolist()
                else:
                    labels_NO2[time]=[-1]
                    
                if len(predicted_BC)>0:    
                    labels_BC[time]=predicted_BC[0].tolist()
                else:
                    labels_BC[time]=[-1]
                
                if len(predicted_PM1)>0:    
                    labels_PM1[time]=predicted_PM1[0].tolist()
                else:
                    labels_PM1[time]=[-1]
                
                if len(predicted_PM25)>0:    
                    labels_PM25[time]=predicted_PM25[0].tolist()
                else:
                    labels_PM25[time]=[-1]
                
                if len(predicted_PM10)>0:    
                    labels_PM10[time]=predicted_PM10[0].tolist()
                else:
                    labels_PM10[time]=[-1]
                
                labels_Speed[time]=[-1]
                
                
                    
                
#                 labels_dictionay[str(df_test["time"].iloc[0])]=[most_frequent(new_data[0])]
            
    
    return labels_dictionay,labels_Temperature,labels_Humidity,labels_NO2,labels_BC,labels_PM1,labels_PM25,labels_PM10,labels_Speed




def df_to_dict(classes):
    d={}
    for i in range(len(classes)):
        if "timestamp" in classes.columns:
            d[str(classes.iloc[i]['timestamp'])]= classes.iloc[i]['activity']
        else:
            d[str(classes.iloc[i]['time'])]= classes.iloc[i]['activity']
    return d




def correct_predictions_based_on_GPS_(classes,gps_data,lon_place,lat_place,radius=0.001,place_Name='Domicile'):

    classes_corrected={}
    for key,value in classes.items():
        lon1=gps_data[gps_data['time']==str(key)]['lon'].values
        lat1=gps_data[gps_data['time']==str(key)]['lat'].values
#         print(lon1,lat1)
        if len(lon1)>0 and len(lat1)>0:
            lon1=lon1[0]
            lat1=lat1[0]        
            if haversine(lon1=lon1,lat1=lat1,lon2=lon_place,lat2=lat_place)<radius:
                if (value=='Dmoicile' and place_Name=='Bureau') or (value=='Bureau' and place_Name=='Domicile'):
                    classes_corrected[key]=place_Name
            else:
                classes_corrected[key]=value
        else:
            classes_corrected[key]=value
        
        values=list(classes_corrected.values())        
        for i in range(2,len(values)-2):
            if values[i]!=values[i+1] and values[i-1]==values[i+1]:
                values[i]=values[i+1]
            if values[i]=='Bureau' and values[i-1]=='Domicile' and values[i-2]=='Domicile' and values[i+2]=='Domicile':
                values[i]='Domicile'
            
            if values[i]=='Domicile' and values[i-1]=='Bureau' and values[i-2]=='Bureau' and values[i+2]=='Bureau':
                values[i]='Bureau'
    
        for key,value in zip(classes_corrected.keys(),values):
            classes_corrected[key]=value
    
    return classes_corrected




def accuracy_calculation_Indoor_Outdoor_Transport(df,predicted_labels,correct_annotations=False,activities=['Walk', 'Bus', 'Bureau', 'Restaurant', 'Domicile', 'Vélo','Voiture', 'Magasin', 'Métro','Gare','Motorcycle','Running','Parc']):
    real_activities={}
    score=[]
    false=0
    true=0
    if correct_annotations==True:
        predicted_labels=correct_annotation(predicted_labels)
    for key,value in predicted_labels.items():
        activity=df[df["time"]==str(key)]["activity"].to_list()[0]
        real_activities[key]=activity
        if activity in activities or activity=='Rue':
            if ((activity=='Walk' or activity=='Rue' or activity=='Running' or activity=='Parc' or activity=='Vélo') and value=='outdoor') or ((activity=='Bus' or activity=='Voiture' or activity=='Motorcycle' or activity=='Métro') and value=='transport') or ((activity=='Bureau' or activity=='Domicile' or activity=='Restaurant' or activity=='Gare' or activity=='Magasin') and value=='indoor'):
                true+=1
                score.append(1)
            else:
                false+=1
                score.append(0)
    if len(score)>0:
        return np.round(true/len(score),3) *100
        




def accuracy_calculation(df,predicted_labels,correct_annotations=False,activities=['Walk', 'Bus', 'Bureau', 'Restaurant', 'Domicile', 'Vélo','Voiture', 'Magasin', 'Métro','Gare','Motorcycle','Running','Parc']):
    real_activities={}
    score=[]
    false=0
    true=0
    if correct_annotations==True:
        predicted_labels=correct_annotation(predicted_labels)
    for key,value in predicted_labels.items():
        
        activity=df[df["time"]==str(key)]["activity"].to_list()[0]
        real_activities[key]=activity
        if activity in activities:
            if activity==value:
                true+=1
                score.append(1)
            else:
                false+=1
                score.append(0)
    if len(score)>0:
        return np.round(true/len(score),3) * 100
        




def validation_per_participant(participant_virtual_id):
    validation=[]
    df=get_preprocessed_data(participant_virtual_id=participant_virtual_id)
    
    #MVB
    if os.path.exists('./RECORD_Two_Predictions/Original_Predictions/'+str(participant_virtual_id)+'.csv')==True:
        classes_Original=pd.read_csv('./RECORD_Two_Predictions/Original_Predictions/'+str(participant_virtual_id)+'.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_Original)))
    else:
        validation.append(np.nan)
    #MVC
    if os.path.exists('./RECORD_Two_Predictions/Category_Predictions/'+str(participant_virtual_id)+'.csv')==True:    
        classes_Category=pd.read_csv('./RECORD_Two_Predictions/Category_Predictions/'+str(participant_virtual_id)+'.csv')
        validation.append(accuracy_calculation_Indoor_Outdoor_Transport(df,df_to_dict(classes_Category)))
    else:
        validation.append(np.nan)
    #MV2steps
    if os.path.exists('./RECORD_Two_Predictions/TWO_STEP_Predictions/'+str(participant_virtual_id)+'.csv')==True:
        classes_2_steps=pd.read_csv('./RECORD_Two_Predictions/TWO_STEP_Predictions/'+str(participant_virtual_id)+'.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_2_steps)))
    else:
        validation.append(np.nan)
    #PensembleMV2steps
    if os.path.exists('./RECORD_Two_Predictions/TWO_STEP_Predictions_Ensemble/'+str(participant_virtual_id)+'.csv')==True:
        classes_2_steps_ensemble=pd.read_csv('./RECORD_Two_Predictions/TWO_STEP_Predictions_Ensemble/'+str(participant_virtual_id)+'.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_2_steps_ensemble)))
    else:
        validation.append(np.nan)
    #PMV
    if os.path.exists('./RECORD_Two_Predictions/TWO_STEP_Predictions_Hilbert/'+str(participant_virtual_id)+'.csv')==True:
        classes_2_steps_hilbert=pd.read_csv('./RECORD_Two_Predictions/TWO_STEP_Predictions_Hilbert/'+str(participant_virtual_id)+'.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_2_steps_hilbert)))
    else:
        validation.append(np.nan)
    
    #MVB+LO
    if os.path.exists('./RECORD_Two_Predictions/Bureau_Correcred/'+str(participant_virtual_id)+'.csv')==True:
        classes_Bureau_Corrected=pd.read_csv('./RECORD_Two_Predictions/Bureau_Correcred/'+str(participant_virtual_id)+'.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_Bureau_Corrected)))
    else:
        validation.append(np.nan)
    
    #MVB+LH
    if os.path.exists('./RECORD_Two_Predictions/Domicile_Corrected/'+str(participant_virtual_id)+'.csv')==True:
        classes_Domicile_Corrected=pd.read_csv('./RECORD_Two_Predictions/Domicile_Corrected/'+str(participant_virtual_id)+'.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_Domicile_Corrected)))
    else:
        validation.append(np.nan)
    
    #MVP+LO
    if os.path.exists('./RECORD_Two_Predictions/Bureau_Corrected_Hilbert/'+str(participant_virtual_id)+'.csv')==True:
        classes_Bureau_Corrected_Hilbert=pd.read_csv('./RECORD_Two_Predictions/Bureau_Corrected_Hilbert/'+str(participant_virtual_id)+'.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_Bureau_Corrected_Hilbert)))
    else:
        validation.append(np.nan)
        
    #MVP+LH
    if os.path.exists('./RECORD_Two_Predictions/Domicile_Corrected_Hilbert/'+str(participant_virtual_id)+'.csv')==True:
        classes_Domicile_Corrected_Hilbert=pd.read_csv('./RECORD_Two_Predictions/Domicile_Corrected_Hilbert/'+str(participant_virtual_id)+'.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_Domicile_Corrected_Hilbert)))
    else:
        validation.append(np.nan)
    
    #MVP
    if os.path.exists('./RECORD_Two_Predictions/Hilbert_Correction/'+str(participant_virtual_id)+'.csv')==True:
        classes_Corrected_Hilbert=pd.read_csv('./RECORD_Two_Predictions/Hilbert_Correction/'+str(participant_virtual_id)+'.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_Corrected_Hilbert)))
    else:
        validation.append(np.nan)
    
    #MLSTMB
    if os.path.exists('./RECORD_Two_Predictions/MLSTM/'+str(participant_virtual_id)+'/MLSTM.csv')==True:
        classes_MLSTM=pd.read_csv('./RECORD_Two_Predictions/MLSTM/'+str(participant_virtual_id)+'/MLSTM.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_MLSTM)))
    else:
        validation.append(np.nan)
        
    #MLSTMP
    if os.path.exists('./RECORD_Two_Predictions/MLSTM/'+str(participant_virtual_id)+'/MLSTM-corrected.csv')==True:
        classes_MLSTM=pd.read_csv('./RECORD_Two_Predictions/MLSTM/'+str(participant_virtual_id)+'/MLSTM-corrected.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_MLSTM)))
    else:
        validation.append(np.nan)
    
    #MLSTMB+LH
    if os.path.exists('./RECORD_Two_Predictions/MLSTM/'+str(participant_virtual_id)+'/MLSTM-Domicile-Corrected.csv')==True:
        classes_MLSTM_Corrected=pd.read_csv('./RECORD_Two_Predictions/MLSTM/'+str(participant_virtual_id)+'/MLSTM-Domicile-Corrected.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_MLSTM_Corrected)))
    else:
        validation.append(np.nan)
        
    #MLSTMB+LO
    if os.path.exists('./RECORD_Two_Predictions/MLSTM/'+str(participant_virtual_id)+'/MLSTM-Bureau-Corrected.csv')==True:
        classes_MLSTM_B_Corrected=pd.read_csv('./RECORD_Two_Predictions/MLSTM/'+str(participant_virtual_id)+'/MLSTM-Bureau-Corrected.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_MLSTM_B_Corrected)))
    else:
        validation.append(np.nan)
    
        
    #PMV2stepsP
    if os.path.exists('./RECORD_Two_Predictions/Hilbert_Correction_Two_Step_Classification_Based_on_Hilbert/'+str(participant_virtual_id)+'.csv')==True:
        classes_Corrected_Hilbert_H=pd.read_csv('./RECORD_Two_Predictions/Hilbert_Correction_Two_Step_Classification_Based_on_Hilbert/'+str(participant_virtual_id)+'.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_Corrected_Hilbert_H)))
    else:
        validation.append(np.nan)
        
    #PensembleMV2stepsP
    if os.path.exists('./RECORD_Two_Predictions/Hilbert_Correction_Two_Step_Classification_Ensemble/'+str(participant_virtual_id)+'.csv')==True:
        classes_Corrected_Hilbert_E=pd.read_csv('./RECORD_Two_Predictions/Hilbert_Correction_Two_Step_Classification_Ensemble/'+str(participant_virtual_id)+'.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_Corrected_Hilbert_E)))
    else:
        validation.append(np.nan)
    
    #MV2stepsP
    if os.path.exists('./RECORD_Two_Predictions/Hilbert_Correction_Two_Step_Classification/'+str(participant_virtual_id)+'.csv')==True:
        classes_Corrected_Hilbert_E=pd.read_csv('./RECORD_Two_Predictions/Hilbert_Correction_Two_Step_Classification/'+str(participant_virtual_id)+'.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_Corrected_Hilbert_E)))
    else:
        validation.append(np.nan)
        
    #KNN-DTWB
    if os.path.exists('../All/'+str(participant_virtual_id)+'/All.csv')==True:
        classes_KNN=pd.read_csv('../All/'+str(participant_virtual_id)+'/All.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_KNN)))
    else:
        validation.append(np.nan)
        
    #KNN-DTWP
    if os.path.exists('../All/'+str(participant_virtual_id)+'/All-corrected.csv')==True:
        classes_KNN_corrected=pd.read_csv('../All/'+str(participant_virtual_id)+'/All-corrected.csv')
        validation.append(accuracy_calculation(df,df_to_dict(classes_KNN_corrected)))
    else:
        validation.append(np.nan)
        
#     ['MVB','MVCategory','MV2steps','PensembleMV2steps','PMV','MVB+LO','MVB+LH','MVP+LO','MVP+LH','MVP','MLSTMB','MLSTMP','MLSTMB+LH','MLSTMB+LO','PMVP','PensembleMV2stepsP','PMV2stepsP','KNN-DTWB','KNN-DTWP']
    #add KNN-DTW
    

    return validation




def validate_all_participants(participant_virtual_ids=[]):
    d={}
    for participant_virtual_id in participant_virtual_ids:
        d[participant_virtual_id]=validation_per_participant(participant_virtual_id)
        
    return d





def post_processing_hilbert(df):

    classes_corrected={}
    current_stop=df.iloc[0]['detected_label']
    current_prediction=df.iloc[0]['prediction']
    i=0
    while i<len(df):
        prediction=df.iloc[i]['prediction']
        hilbert=df.iloc[i]['detected_label']
        time=str(df.iloc[i]['time'])
        if (hilbert=='Bus' or hilbert=='Motorcycle' or hilbert=='Métro' or hilbert=='Parc' or hilbert=='Train' or hilbert=='Vélo'):# and (prediction=='Domicile' or prediction=='Bureau'):
            classes_corrected[time]=hilbert
            i+=1
        else:
#             classes_corrected[time]=prediction
#             i+=1
            if hilbert=='indoor' or str(hilbert)=='nan' :
                classes_corrected[time]=prediction
                i+=1
            else:
                current_stop=df.iloc[i]['detected_label']
                activities=[]
                t_time=[]
                c=i
                for j in range(c,len(df)):
#                     print(df.iloc[j]['detected_label'])
                    if df.iloc[j]['detected_label']==current_stop:
                        activities.append(df.iloc[i]['prediction'])
                        t_time.append(str(df.iloc[j]['time']))
                        i+=1
                    else:
                        classes_corrected[time]=prediction
                        i+=1
                        current_stop=df.iloc[j]['detected_label']                        
                        break
#                 print("--------------------------------")
#                 print(activities)
                act=most_frequent(activities)
#                 print("most_frequent",act,len(t_time))
                for t,a in zip(t_time,activities):
                    if act != 'No Data':
                        classes_corrected[t]=act
#                 print("--------------------------------")
    
    
    values=list(classes_corrected.values())        
    for i in range(2,len(values)-2):
        if values[i]!=values[i+1] and values[i-1]==values[i+1]:
            values[i]=values[i+1]
        if values[i]=='Bureau' and values[i-1]=='Domicile' and values[i-2]=='Domicile' and values[i+2]=='Domicile':
            values[i]='Domicile'
            
        if values[i]=='Domicile' and values[i-1]=='Bureau' and values[i-2]=='Bureau' and values[i+2]=='Bureau':
            values[i]='Bureau'
    
    for key,value in zip(classes_corrected.keys(),values):
        classes_corrected[key]=value

    
    return classes_corrected




def post_processing_only_hilbert(df):

    classes_corrected={}
    current_stop=df.iloc[0]['detected_label']
    current_prediction=df.iloc[0]['prediction']
    i=0
    while i<len(df):
        prediction=df.iloc[i]['prediction']
        hilbert=df.iloc[i]['detected_label']
        time=str(df.iloc[i]['time'])
        if (hilbert=='Bus' or hilbert=='Motorcycle' or hilbert=='Métro' or hilbert=='Parc' or hilbert=='Train' or hilbert=='Vélo'):# and (prediction=='Domicile' or prediction=='Bureau'):
            classes_corrected[time]=hilbert
            i+=1
        else:
#             classes_corrected[time]=prediction
#             i+=1
            if hilbert=='indoor' or str(hilbert)=='nan' :
                classes_corrected[time]=prediction
                i+=1
            else:
                current_stop=df.iloc[i]['detected_label']
                activities=[]
                t_time=[]
                c=i
                for j in range(c,len(df)):
#                     print(df.iloc[j]['detected_label'])
                    if df.iloc[j]['detected_label']==current_stop:
                        activities.append(df.iloc[i]['prediction'])
                        t_time.append(str(df.iloc[j]['time']))
                        i+=1
                    else:
                        classes_corrected[time]=prediction
                        i+=1
                        current_stop=df.iloc[j]['detected_label']                        
                        break
#                 print("--------------------------------")
#                 print(activities)
                act=most_frequent(activities)
#                 print("most_frequent",act,len(t_time))
                for t,a in zip(t_time,activities):
                    classes_corrected[t]=act
#                 print("--------------------------------")
    
    values=list(classes_corrected.values())        
    for i in range(2,len(values)-2):
        if values[i]!=values[i+1] and values[i-1]==values[i+1]:
            values[i]=values[i+1]
        if values[i]=='Bureau' and values[i-1]=='Domicile' and values[i-2]=='Domicile' and values[i+2]=='Domicile':
            values[i]='Domicile'
            
        if values[i]=='Domicile' and values[i-1]=='Bureau' and values[i-2]=='Bureau' and values[i+2]=='Bureau':
            values[i]='Bureau'
            
            
        if (values[i] in ['Voiture','Métro','Train','Motorcycle','Bus'] and values[i-1] in ['Voiture','Métro','Train','Motorcycle','Bus']) or (values[i] in ['Voiture','Métro','Train','Motorcycle','Bus'] and values[i+1] in ['Voiture','Métro','Train','Motorcycle','Bus']):
            values[i]='Voiture'
    
        if (values[i] in ['Vélo','Walk','Running','Parc','Rue'] and values[i-1] in ['Vélo','Walk','Running','Parc','Rue']) or (values[i] in ['Vélo','Walk','Running','Parc','Rue'] and values[i+1] in ['Vélo','Walk','Running','Parc','Rue']):
            values[i]='Walk'
    
    for key,value in zip(classes_corrected.keys(),values):
        classes_corrected[key]=value

    return classes_corrected




def most_frequent(List):
    counter = 0
    num = List[0]
      
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
  
    return num




def prepare_data_confusion_matrix(dfs,predicted_labels_list,dictionary={'Walk': 1, 'Bus': 2, 'Bureau': 3, 'Restaurant': 4, 'Domicile': 5, 'Vélo': 6, 'Voiture': 7, 'Magasin': 8, 'Métro': 9, 'Gare': 10, 'Motorcycle': 11, 'Running': 12, 'Parc': 13},activities=['Walk', 'Bus', 'Bureau', 'Restaurant', 'Domicile', 'Vélo','Voiture', 'Magasin', 'Métro','Gare','Motorcycle','Running','Parc']):
    
    y_true=[]
    y_predict=[]
    
    false=0
    true=0
    
    for df,predicted_labels in zip(dfs,predicted_labels_list):
        for key,value in predicted_labels.items():
            activity=df[df["time"]==str(key)]["activity"].to_list()[0]
#             real_activities[key]=activity
            if activity in activities and value in activities:
#                 print(activity,value)
                y_true.append(dictionary[activity])
                y_predict.append(dictionary[value])
                
    return y_true,y_predict
                




def get_new_fig(fn, figsize=[20,10]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1
#

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []; text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)

    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        
        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #text to ADD
        font_prop = fm.FontProperties(weight='bold', size=16)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        #print '\n'

        #set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if(per > 0):
            txt = '%s\n%.2f%%' %(cell_val, per)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        #main diagonal
        if(col == lin):
            #set color of the textin the diagonal to white
            oText.set_color('black')
            # set background color in the diagonal to blue
#             facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('black')

    return text_add, text_del
#

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    #print ('\ndf_cm:\n', df_cm, '\n\b\n')
#

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Blues", fmt='.2f', fz=25,
      lw=2, cbar=False, figsize=[20,10], show_null_values=0, pred_val_axis='x'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    pred_val_axis='x'
    if(pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    #thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w',linewidths=lw)

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 20)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 20)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array( df_cm.to_records(index=False).tolist() )
    text_add = []; text_del = [];
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) - [0.5,0.5]
        lin = int(pos[1]); col = int(pos[0]);
        posi += 1
        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    ax.set_title('Confusion matrix \n Multi-view Predictions\n',fontsize=35)
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  #set layout slim
    plt.show()
#

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Blues",
      fmt='.2f', fz=25, lw=12, cbar=False, figsize=[20,10], show_null_values=0, pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    #data
    if(not columns):
        #labels axis integer:
        ##columns = range(1, len(np.unique(y_test))+1)
        #labels axis string:
        from string import ascii_uppercase
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

    confm = confusion_matrix(y_test, predictions)
    cmap = 'Blues';
    fz = 16;
    figsize=[25,15];
    show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=plt.cm.Blues, figsize=figsize, show_null_values=show_null_values, pred_val_axis=pred_val_axis)
#



#
#TEST functions
#
def _test_cm():
    #test function with confusion matrix done
    array = np.array( [[13,  0,  1,  0,  2,  0],
                       [ 0, 50,  2,  0, 10,  0],
                       [ 0, 13, 16,  0,  0,  3],
                       [ 0,  0,  0, 13,  1,  0],
                       [ 0, 40,  0,  1, 15,  0],
                       [ 0,  0,  0,  0,  0, 20]])
    #get pandas dataframe
    df_cm = DataFrame(array, index=range(1,7), columns=range(1,7))
    #colormap: see this and choose your more dear
    cmap = 'PuRd'
    pretty_plot_confusion_matrix(df_cm, cmap=cmap)
#

def _test_data_class():
    """ test function with y_test (actual values) and predictions (predic) """
    #data
    y_test = np.array([1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5])
    predic = np.array([1,2,4,3,5, 1,2,4,3,5, 1,2,3,4,4, 1,4,3,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,3,3,5, 1,2,3,3,5, 1,2,3,4,4, 1,2,3,4,1, 1,2,3,4,1, 1,2,3,4,1, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5])
    """
      Examples to validate output (confusion matrix plot)
        actual: 5 and prediction 1   >>  3
        actual: 2 and prediction 4   >>  1
        actual: 3 and prediction 4   >>  10
    """
    columns = []
    annot = True;
    cmap = 'Oranges';
    fmt = '.2f'
    lw = 0.5
    cbar = False
    show_null_values = 2
    pred_val_axis = 'y'
    #size::
    fz = 12;
    figsize = [9,9];
    if(len(y_test) > 10):
        fz=9; figsize=[14,14];
    plot_confusion_matrix_from_data(y_test, predic, columns,
      annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)




#####



def add_Speed_column(df):
    Speed=[]
    for i in range(len(df)):
#         print((df.shift(-1).iloc[i]["time"]-df.iloc[i]["time"]).seconds)
        if (df.shift(-1).iloc[i]["time"]-df.iloc[i]["time"]).seconds==60:
            if "datetime" in df.columns:
#                 Speed.append(getSpeedFromLonLat_KM_per_H(df.iloc[i]["lng"],df.iloc[i]["lat"],df.shift(-1).iloc[i]["lng"],df.shift(-1).iloc[i]["lat"]))
                Speed.append(getSpeedFromLonLat_KM_per_H(df.shift(-1).iloc[i]["lng"],df.shift(-1).iloc[i]["lat"],df.iloc[i]["lng"],df.iloc[i]["lat"]))
            else:
#                 Speed.append(getSpeedFromLonLat_KM_per_H(df.iloc[i]["lon"],df.iloc[i]["lat"],df.shift(-1).iloc[i]["lon"],df.shift(-1).iloc[i]["lat"]))
                Speed.append(getSpeedFromLonLat_KM_per_H(df.shift(-1).iloc[i]["lon"],df.shift(-1).iloc[i]["lat"],df.iloc[i]["lon"],df.iloc[i]["lat"]))
        else:
            Speed.append(np.nan)
    df["Speed"]=Speed
    return df  


COLOR = {
     'Indoor': '#00CF83',
    'Office': '#20B2AA',
    'Transport': '#FFC0CB',
    'Home': '#8FBC8F',
    'Outdoor': '#D3D3D3',
    'Unkown':'yellow',
    'Unavailable Data': '#F5F5F5'
}


def get_color(k, color_dict=COLOR):
    """
    Return a color (random if "k" is negative)
    """
    if str(k)=='None' or str(k)=='nan' or k=='pdp' or k=='No Data' :
        return color_dict['Unavailable Data']
    else:
        if k=='Motorcycle' or k=='Bus' or k=='Métro' or k=='Voiture' or k=='Train' or k=='Tramway':
            return color_dict['Transport']
        else:
            if k=='Restaurant' or k=='Gare' or k=='Magasin' or k=='Indoor' or k=='Cinéma':
                return color_dict['Indoor']
            else:
                if k=='Parc' or k=='Walk' or k=='Running' or k=='Vélo' or k=='Rue':
                    return color_dict['Outdoor']
                else:
                    if k=='Bureau':
                        return color_dict['Office']
                    else:
                        if k == 'Domicile':
                            return color_dict['Home']
                        else:
                            if k=='Inconnu' or k=='Inconnue':
                                return color_dict['Unkown']
    return color_dict[k]

def plot_journal_VGP(data,predictions, start_datetime=None, end_datetime=None,colors=COLOR,path='Context_VGP/'):
    df = predictions
    df2=df
    participant_virtual_id=data.participant_virtual_id.iloc[0]
    path=path+str(participant_virtual_id)+'-CONTEXT.png'
    print(path)
    if len(data) >0:

        data = data.drop_duplicates('time')
        data = data.set_index('time')
        data = data.resample('60S').asfreq()


    if (len(df) >0) & (len(data) > 0) :
        if len(data.dropna(subset=['BC']))>0 and len(data.dropna(subset=['NO2']))>0 and len(data.dropna(subset=['PM10']))>0:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [3,3,3,1]})

            # TODO: add warning if days between start_datetime and end_datetime do not overlap with df
            df['timestamp']=pd.to_datetime(df['timestamp'])
            if start_datetime is None:
                start_datetime = df[:-1]['timestamp'].min()
            if end_datetime is None:
                end_datetime = df[:-1]['timestamp'].max()

            current_labels = []

            for i in range(len(df)):
                
                t0 = df.iloc[i]['timestamp']
                t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                cl = df2.iloc[i]['activity']

                color = get_color(cl)
                if start_datetime <= t0 <= end_datetime:
                    if cl in current_labels:
                        ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                    else:
                        current_labels += [cl]
                        ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)
            plt.xlim(start_datetime, end_datetime)

            ax1.plot(data.PM10, 'k-')

            ax1.set_ylabel('PM10 (µg/m3)')

            ##################


            current_labels = []

            for i in range(len(df)):
                
                t0 = df.iloc[i]['timestamp']
                t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                cl = df2.iloc[i]['activity']

                color = get_color(cl)
                if start_datetime <= t0 <= end_datetime:
                    if cl in current_labels:
                        ax2.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                    else:
                        current_labels += [cl]
                        ax2.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)


            ax2.plot(data.NO2, 'b-')

            ax2.set_ylabel('NO2 (µg/m3)')
            ############################


            current_labels = []

            for i in range(len(df)):
                
                t0 = df.iloc[i]['timestamp']
                t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                cl = df2.iloc[i]['activity']

                color = get_color(cl)
                if start_datetime <= t0 <= end_datetime:
                    if cl in current_labels:
                        ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                    else:
                        current_labels += [cl]
                        ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)


            ax3.plot(data.BC, 'm-')

            ax3.set_ylabel('BC (ng/m3)')
            ###########################

            for idx, row in data[:-1].iterrows():

                t0 = idx
                t1 = idx+pd.Timedelta(minutes=5)
                cl = row['activity']

                color = get_color(cl)
                if start_datetime <= t0 <= end_datetime:
                    if cl in current_labels:
                        ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                    else:
                        current_labels += [cl]
                        ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

            plt.xlim(start_datetime, end_datetime)

            ax4.set_ylabel('Declared')
            ax4.set_yticklabels([])

            ###########################

#             handles, labels_str = ax3.get_legend_handles_labels()
#             labels = list(labels_str)
#             # sort them by labels
#             import operator
#             hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
#             handles2, labels2 = zip(*hl)
            hl=[]
            for key,value in colors.items():  
                if key!='Cinéma' and key!='Rue' and key!='Données indisponibles':
                    hl.append((mpatches.Patch(color=get_color(key)),key))

            handles2, labels2 = zip(*hl)
            ax4.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0, prop={'size': 16})
            user = data.participant_virtual_id.iloc[0]

            ax1.set_title('Participant %s' % participant_virtual_id)

            plt.savefig(path, bbox_inches = 'tight', pad_inches=.25)

            #*****************************************************************************************
        else:
            if len(data.dropna(subset=['BC']))>0 and len(data.dropna(subset=['NO2']))>0 : #PM not found
                fig, ( ax2, ax3, ax4) = plt.subplots(3, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [3,3,1]})

                # TODO: add warning if days between start_datetime and end_datetime do not overlap with df
                df['timestamp']=pd.to_datetime(df['timestamp'])
                if start_datetime is None:
                    start_datetime = df[:-1]['timestamp'].min()
                if end_datetime is None:
                    end_datetime = df[:-1]['timestamp'].max()

#                 current_labels = []

#                 for i in range(len(df)):

#                     t0 = df.iloc[i]['timestamp']
#                     t1 = df.shift(-1).iloc[i]['timestamp']
#                     cl = df2.iloc[i]['activity']

#                     color = get_color(cl)
#                     if start_datetime <= t0 <= end_datetime:
#                         if cl in current_labels:
#                             ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
#                         else:
#                             current_labels += [cl]
#                             ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)
#                 plt.xlim(start_datetime, end_datetime)

#                 ax1.plot(data.PM10, 'k-')

#                 ax1.set_ylabel('PM10 (µg/m3)')

                ##################


                current_labels = []

                for i in range(len(df)):

                    t0 = df.iloc[i]['timestamp']
                    t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                    cl = df.iloc[i]['activity']

                    color = get_color(cl)
                    if start_datetime <= t0 <= end_datetime:
                        if cl in current_labels:
                            ax2.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                        else:
                            current_labels += [cl]
                            ax2.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)


                ax2.plot(data.NO2, 'b-')

                ax2.set_ylabel('NO2 (µg/m3)')
                ############################


                current_labels = []

                for i in range(len(df)):

                    t0 = df.iloc[i]['timestamp']
                    t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                    cl = df.iloc[i]['activity']

                    color = get_color(cl)
                    if start_datetime <= t0 <= end_datetime:
                        if cl in current_labels:
                            ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                        else:
                            current_labels += [cl]
                            ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)


                ax3.plot(data.BC, 'm-')

                ax3.set_ylabel('BC (ng/m3)')
                ###########################

                for idx, row in data[:-1].iterrows():

                    t0 = idx
                    t1 = idx+pd.Timedelta(minutes=5)
                    cl = row['activity']

                    color = get_color(cl)
                    if start_datetime <= t0 <= end_datetime:
                        if cl in current_labels:
                            ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                        else:
                            current_labels += [cl]
                            ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

                plt.xlim(start_datetime, end_datetime)

                ax4.set_ylabel('Declared')
                ax4.set_yticklabels([])

                ###########################

                hl=[]
                for key,value in colors.items():  
                    if key!='Cinéma' and key!='Rue' and key!='Données indisponibles':
                        hl.append((mpatches.Patch(color=get_color(key)),key))

                handles2, labels2 = zip(*hl)
                ax4.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0, prop={'size': 16})
                user = data.participant_virtual_id.iloc[0]

                ax1.set_title('Participant %s' % participant_virtual_id)

                plt.savefig(path, bbox_inches = 'tight', pad_inches=.25)

                #*****************************************************************************************
            else:
                if len(data.dropna(subset=['BC']))>0 and len(data.dropna(subset=['PM10']))>0 : #NO2 not found
                    fig, ( ax1, ax3, ax4) = plt.subplots(3, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [3,3,1]})

                    # TODO: add warning if days between start_datetime and end_datetime do not overlap with df
                    df['timestamp']=pd.to_datetime(df['timestamp'])
                    if start_datetime is None:
                        start_datetime = df[:-1]['timestamp'].min()
                    if end_datetime is None:
                        end_datetime = df[:-1]['timestamp'].max()

                    current_labels = []

                    for i in range(len(df)):

                        t0 = df.iloc[i]['timestamp']
                        t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                        cl = df.iloc[i]['activity']

                        color = get_color(cl)
                        if start_datetime <= t0 <= end_datetime:
                            if cl in current_labels:
                                ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                            else:
                                current_labels += [cl]
                                ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)
                    plt.xlim(start_datetime, end_datetime)

                    ax1.plot(data.PM10, 'k-')

                    ax1.set_ylabel('PM10 (µg/m3)')

                    ############################


                    current_labels = []

                    for i in range(len(df)):

                        t0 = df.iloc[i]['timestamp']
                        t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                        cl = df.iloc[i]['activity']

                        color = get_color(cl)
                        if start_datetime <= t0 <= end_datetime:
                            if cl in current_labels:
                                ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                            else:
                                current_labels += [cl]
                                ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)


                    ax3.plot(data.BC, 'm-')

                    ax3.set_ylabel('BC (ng/m3)')
                    ###########################

                    for idx, row in data[:-1].iterrows():

                        t0 = idx
                        t1 = idx+pd.Timedelta(minutes=5)
                        cl = row['activity']

                        color = get_color(cl)
                        if start_datetime <= t0 <= end_datetime:
                            if cl in current_labels:
                                ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                            else:
                                current_labels += [cl]
                                ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

                    plt.xlim(start_datetime, end_datetime)

                    ax4.set_ylabel('Declared')
                    ax4.set_yticklabels([])

                    ###########################

                    hl=[]
                    for key,value in colors.items():  
                        if key!='Cinéma' and key!='Rue' and key!='Données indisponibles':
                            hl.append((mpatches.Patch(color=get_color(key)),key))

                    handles2, labels2 = zip(*hl)
                    ax4.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0, prop={'size': 16})
                    user = data.participant_virtual_id.iloc[0]

                    ax1.set_title('Participant %s' % participant_virtual_id)

                    plt.savefig(path, bbox_inches = 'tight', pad_inches=.25)

                    #*****************************************************************************************
                else:
                    if len(data.dropna(subset=['NO2']))>0 and len(data.dropna(subset=['PM10']))>0 : #NO2 not found
                        fig, ( ax1, ax3, ax4) = plt.subplots(3, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [3,3,1]})

                        # TODO: add warning if days between start_datetime and end_datetime do not overlap with df
                        df['timestamp']=pd.to_datetime(df['timestamp'])
                        if start_datetime is None:
                            start_datetime = df[:-1]['timestamp'].min()
                        if end_datetime is None:
                            end_datetime = df[:-1]['timestamp'].max()

                        current_labels = []

                        for i in range(len(df)):

                            t0 = df.iloc[i]['timestamp']
                            t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                            cl = df.iloc[i]['activity']

                            color = get_color(cl)
                            if start_datetime <= t0 <= end_datetime:
                                if cl in current_labels:
                                    ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                else:
                                    current_labels += [cl]
                                    ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)
                        plt.xlim(start_datetime, end_datetime)

                        ax1.plot(data.PM10, 'k-')

                        ax1.set_ylabel('PM10 (µg/m3)')

                        ############################


                        current_labels = []

                        for i in range(len(df)):

                            t0 = df.iloc[i]['timestamp']
                            t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                            cl = df.iloc[i]['activity']

                            color = get_color(cl)
                            if start_datetime <= t0 <= end_datetime:
                                if cl in current_labels:
                                    ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                else:
                                    current_labels += [cl]
                                    ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)


                        ax3.plot(data.NO2, 'm-')

                        ax3.set_ylabel('NO2 (ng/m3)')
                        ###########################

                        for idx, row in data[:-1].iterrows():

                            t0 = idx
                            t1 = idx+pd.Timedelta(minutes=5)
                            cl = row['activity']

                            color = get_color(cl)
                            if start_datetime <= t0 <= end_datetime:
                                if cl in current_labels:
                                    ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                else:
                                    current_labels += [cl]
                                    ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

                        plt.xlim(start_datetime, end_datetime)

                        ax4.set_ylabel('Declared')
                        ax4.set_yticklabels([])

                        ###########################

                        hl=[]
                        for key,value in colors.items():  
                            if key!='Cinéma' and key!='Rue' and key!='Données indisponibles':
                                hl.append((mpatches.Patch(color=get_color(key)),key))

                        handles2, labels2 = zip(*hl)
                        ax4.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0, prop={'size': 16})
                        user = data.participant_virtual_id.iloc[0]

                        ax1.set_title('Participant %s' % participant_virtual_id)

                        plt.savefig(path, bbox_inches = 'tight', pad_inches=.25)


                        #*****************************************************************************************
                    else:
                        if len(data.dropna(subset=['PM10']))>0 : #NO2 and BC not found
                            fig, ( ax1, ax4) = plt.subplots(2, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [3,1]})

                            # TODO: add warning if days between start_datetime and end_datetime do not overlap with df
                            df['timestamp']=pd.to_datetime(df['timestamp'])
                            if start_datetime is None:
                                start_datetime = df[:-1]['timestamp'].min()
                            if end_datetime is None:
                                end_datetime = df[:-1]['timestamp'].max()

                            current_labels = []

                            for i in range(len(df)):

                                t0 = df.iloc[i]['timestamp']
                                t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                                cl = df.iloc[i]['activity']

                                color = get_color(cl)
                                if start_datetime <= t0 <= end_datetime:
                                    if cl in current_labels:
                                        ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                    else:
                                        current_labels += [cl]
                                        ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)
                            plt.xlim(start_datetime, end_datetime)

                            ax1.plot(data.PM10, 'k-')

                            ax1.set_ylabel('PM10 (µg/m3)')

                            ############################


                            for idx, row in data[:-1].iterrows():

                                t0 = idx
                                t1 = idx+pd.Timedelta(minutes=5)
                                cl = row['activity']

                                color = get_color(cl)
                                if start_datetime <= t0 <= end_datetime:
                                    if cl in current_labels:
                                        ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                    else:
                                        current_labels += [cl]
                                        ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

                            plt.xlim(start_datetime, end_datetime)

                            ax4.set_ylabel('Declared')
                            ax4.set_yticklabels([])

                            ###########################

                            hl=[]
                            for key,value in colors.items():  
                                if  key!='Cinéma' and key!='Rue' and key!='Données indisponibles':
                                    hl.append((mpatches.Patch(color=get_color(key)),key))

                            handles2, labels2 = zip(*hl)
                            ax4.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0, prop={'size': 16})
                            user = data.participant_virtual_id.iloc[0]

                            ax1.set_title('Participant %s' % participant_virtual_id)

                            plt.savefig(path, bbox_inches = 'tight', pad_inches=.25)


                            #*****************************************************************************************
                        else:
                            if len(data.dropna(subset=['BC']))>0 : #NO2 and PM10 not found
                                fig, ( ax1, ax4) = plt.subplots(2, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [3,1]})

                                # TODO: add warning if days between start_datetime and end_datetime do not overlap with df
                                df['timestamp']=pd.to_datetime(df['timestamp'])
                                if start_datetime is None:
                                    start_datetime = df[:-1]['timestamp'].min()
                                if end_datetime is None:
                                    end_datetime = df[:-1]['timestamp'].max()

                                current_labels = []

                                for i in range(len(df)):

                                    t0 = df.iloc[i]['timestamp']
                                    t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                                    cl = df.iloc[i]['activity']

                                    color = get_color(cl)
                                    if start_datetime <= t0 <= end_datetime:
                                        if cl in current_labels:
                                            ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                        else:
                                            current_labels += [cl]
                                            ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)
                                plt.xlim(start_datetime, end_datetime)

                                ax1.plot(data.BC, 'k-')

                                ax1.set_ylabel('BC (ng/m3)')

                                ############################


                                for idx, row in data[:-1].iterrows():

                                    t0 = idx
                                    t1 = idx+pd.Timedelta(minutes=5)
                                    cl = row['activity']

                                    color = get_color(cl)
                                    if start_datetime <= t0 <= end_datetime:
                                        if cl in current_labels:
                                            ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                        else:
                                            current_labels += [cl]
                                            ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

                                plt.xlim(start_datetime, end_datetime)

                                ax4.set_ylabel('Declared')
                                ax4.set_yticklabels([])

                                ###########################

                                hl=[]
                                for key,value in colors.items():  
                                    if  key!='Cinéma' and key!='Rue' and key!='Données indisponibles':
                                        hl.append((mpatches.Patch(color=get_color(key)),key))

                                handles2, labels2 = zip(*hl)
                                ax4.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0, prop={'size': 16})
                                user = data.participant_virtual_id.iloc[0]

                                ax1.set_title('Participant %s' % participant_virtual_id)

                                plt.savefig(path, bbox_inches = 'tight', pad_inches=.25)

                                #*****************************************************************************************
                            else:
                                if len(data.dropna(subset=['NO2']))>0 : #BC and PM10 not found
                                    fig, ( ax1, ax4) = plt.subplots(2, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [3,1]})

                                    # TODO: add warning if days between start_datetime and end_datetime do not overlap with df
                                    df['timestamp']=pd.to_datetime(df['timestamp'])
                                    if start_datetime is None:
                                        start_datetime = df[:-1]['timestamp'].min()
                                    if end_datetime is None:
                                        end_datetime = df[:-1]['timestamp'].max()

                                    current_labels = []

                                    for i in range(len(df)):

                                        t0 = df.iloc[i]['timestamp']
                                        t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                                        cl = df.iloc[i]['activity']

                                        color = get_color(cl)
                                        if start_datetime <= t0 <= end_datetime:
                                            if cl in current_labels:
                                                ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                            else:
                                                current_labels += [cl]
                                                ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)
                                    plt.xlim(start_datetime, end_datetime)

                                    ax1.plot(data.BC, 'k-')

                                    ax1.set_ylabel('NO2 (µg/m3)')

                                    ############################


                                    for idx, row in data[:-1].iterrows():

                                        t0 = idx
                                        t1 = idx+pd.Timedelta(minutes=5)
                                        cl = row['activity']

                                        color = get_color(cl)
                                        if start_datetime <= t0 <= end_datetime:
                                            if cl in current_labels:
                                                ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                            else:
                                                current_labels += [cl]
                                                ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

                                    plt.xlim(start_datetime, end_datetime)

                                    ax4.set_ylabel('Declared')
                                    ax4.set_yticklabels([])

                                    ###########################

                                    hl=[]
                                    for key,value in colors.items():  
                                        if  key!='Cinéma' and key!='Rue' and key!='Données indisponibles':
                                            hl.append((mpatches.Patch(color=get_color(key)),key))

                                    handles2, labels2 = zip(*hl)
                                    ax4.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0, prop={'size': 16})
                                    user = data.participant_virtual_id.iloc[0]

                                    ax1.set_title('Participant %s' % participant_virtual_id)

                                    plt.savefig(path, bbox_inches = 'tight', pad_inches=.25)

                                    #*****************************************************************************************
                                    

                                    

def plot_journal_RECORD(data,predictions, start_datetime=None, end_datetime=None,colors=COLOR):
    df = predictions
    df2=df
    participant_virtual_id=data.participant_virtual_id.iloc[0]

    if len(data) >0:

        data = data.drop_duplicates('time')
        data = data.set_index('time')
        data = data.resample('60S').asfreq()


    if (len(df) >0) & (len(data) > 0) :
        if len(data.dropna(subset=['BC']))>0 and len(data.dropna(subset=['NO2']))>0 and len(data.dropna(subset=['PM10']))>0:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [3,3,3,1]})

            # TODO: add warning if days between start_datetime and end_datetime do not overlap with df
            df['timestamp']=pd.to_datetime(df['timestamp'])
            if start_datetime is None:
                start_datetime = df[:-1]['timestamp'].min()
            if end_datetime is None:
                end_datetime = df[:-1]['timestamp'].max()

            current_labels = []

            for i in range(len(df)):
                
                t0 = df.iloc[i]['timestamp']
                t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                cl = df2.iloc[i]['activity']

                color = get_color(cl)
                if start_datetime <= t0 <= end_datetime:
                    if cl in current_labels:
                        ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                    else:
                        current_labels += [cl]
                        ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)
            plt.xlim(start_datetime, end_datetime)

            ax1.plot(data.PM10, 'k-')

            ax1.set_ylabel('PM10 (µg/m3)')

            ##################


            current_labels = []

            for i in range(len(df)):
                
                t0 = df.iloc[i]['timestamp']
                t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                cl = df2.iloc[i]['activity']

                color = get_color(cl)
                if start_datetime <= t0 <= end_datetime:
                    if cl in current_labels:
                        ax2.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                    else:
                        current_labels += [cl]
                        ax2.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)


            ax2.plot(data.NO2, 'b-')

            ax2.set_ylabel('NO2 (µg/m3)')
            ############################


            current_labels = []

            for i in range(len(df)):
                
                t0 = df.iloc[i]['timestamp']
                t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                cl = df2.iloc[i]['activity']

                color = get_color(cl)
                if start_datetime <= t0 <= end_datetime:
                    if cl in current_labels:
                        ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                    else:
                        current_labels += [cl]
                        ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)


            ax3.plot(data.BC, 'm-')

            ax3.set_ylabel('BC (ng/m3)')
            ###########################

            for idx, row in data[:-1].iterrows():

                t0 = idx
                t1 = idx+pd.Timedelta(minutes=5)
                cl = row['activity']

                color = get_color(cl)
                if start_datetime <= t0 <= end_datetime:
                    if cl in current_labels:
                        ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                    else:
                        current_labels += [cl]
                        ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

            plt.xlim(start_datetime, end_datetime)

            ax4.set_ylabel('Declared')
            ax4.set_yticklabels([])

            ###########################

#             handles, labels_str = ax3.get_legend_handles_labels()
#             labels = list(labels_str)
#             # sort them by labels
#             import operator
#             hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
#             handles2, labels2 = zip(*hl)
            hl=[]
            for key,value in colors.items():  
                if key!='Indoor' and key!='Cinéma' and key!='Rue' and key!='Données indisponibles':
                    hl.append((mpatches.Patch(color=get_color(key)),key))

            handles2, labels2 = zip(*hl)
            ax4.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0)
            user = data.participant_virtual_id.iloc[0]

            ax1.set_title('Participant %s' % participant_virtual_id)

            plt.savefig('Context_RECORD/'+str(participant_virtual_id)+'-CONTEXT.png', bbox_inches = 'tight', pad_inches=.25)

            #*****************************************************************************************
        else:
            if len(data.dropna(subset=['BC']))>0 and len(data.dropna(subset=['NO2']))>0 : #PM not found
                fig, ( ax2, ax3, ax4) = plt.subplots(3, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [3,3,1]})

                # TODO: add warning if days between start_datetime and end_datetime do not overlap with df
                df['timestamp']=pd.to_datetime(df['timestamp'])
                if start_datetime is None:
                    start_datetime = df[:-1]['timestamp'].min()
                if end_datetime is None:
                    end_datetime = df[:-1]['timestamp'].max()

#                 current_labels = []

#                 for i in range(len(df)):

#                     t0 = df.iloc[i]['timestamp']
#                     t1 = df.shift(-1).iloc[i]['timestamp']
#                     cl = df2.iloc[i]['activity']

#                     color = get_color(cl)
#                     if start_datetime <= t0 <= end_datetime:
#                         if cl in current_labels:
#                             ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
#                         else:
#                             current_labels += [cl]
#                             ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)
#                 plt.xlim(start_datetime, end_datetime)

#                 ax1.plot(data.PM10, 'k-')

#                 ax1.set_ylabel('PM10 (µg/m3)')

                ##################


                current_labels = []

                for i in range(len(df)):

                    t0 = df.iloc[i]['timestamp']
                    t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                    cl = df.iloc[i]['activity']

                    color = get_color(cl)
                    if start_datetime <= t0 <= end_datetime:
                        if cl in current_labels:
                            ax2.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                        else:
                            current_labels += [cl]
                            ax2.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)


                ax2.plot(data.NO2, 'b-')

                ax2.set_ylabel('NO2 (µg/m3)')
                ############################


                current_labels = []

                for i in range(len(df)):

                    t0 = df.iloc[i]['timestamp']
                    t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                    cl = df.iloc[i]['activity']

                    color = get_color(cl)
                    if start_datetime <= t0 <= end_datetime:
                        if cl in current_labels:
                            ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                        else:
                            current_labels += [cl]
                            ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)


                ax3.plot(data.BC, 'm-')

                ax3.set_ylabel('BC (ng/m3)')
                ###########################

                for idx, row in data[:-1].iterrows():

                    t0 = idx
                    t1 = idx+pd.Timedelta(minutes=5)
                    cl = row['activity']

                    color = get_color(cl)
                    if start_datetime <= t0 <= end_datetime:
                        if cl in current_labels:
                            ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                        else:
                            current_labels += [cl]
                            ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

                plt.xlim(start_datetime, end_datetime)

                ax4.set_ylabel('Declared')
                ax4.set_yticklabels([])

                ###########################

                hl=[]
                for key,value in colors.items():  
                    if key!='Indoor' and key!='Cinéma' and key!='Rue' and key!='Données indisponibles':
                        hl.append((mpatches.Patch(color=get_color(key)),key))

                handles2, labels2 = zip(*hl)
                ax4.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0)
                user = data.participant_virtual_id.iloc[0]

                ax1.set_title('Participant %s' % participant_virtual_id)

                plt.savefig('Context_RECORD/'+str(participant_virtual_id)+'-CONTEXT.png', bbox_inches = 'tight', pad_inches=.25)

                #*****************************************************************************************
            else:
                if len(data.dropna(subset=['BC']))>0 and len(data.dropna(subset=['PM10']))>0 : #NO2 not found
                    fig, ( ax1, ax3, ax4) = plt.subplots(3, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [3,3,1]})

                    # TODO: add warning if days between start_datetime and end_datetime do not overlap with df
                    df['timestamp']=pd.to_datetime(df['timestamp'])
                    if start_datetime is None:
                        start_datetime = df[:-1]['timestamp'].min()
                    if end_datetime is None:
                        end_datetime = df[:-1]['timestamp'].max()

                    current_labels = []

                    for i in range(len(df)):

                        t0 = df.iloc[i]['timestamp']
                        t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                        cl = df.iloc[i]['activity']

                        color = get_color(cl)
                        if start_datetime <= t0 <= end_datetime:
                            if cl in current_labels:
                                ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                            else:
                                current_labels += [cl]
                                ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)
                    plt.xlim(start_datetime, end_datetime)

                    ax1.plot(data.PM10, 'k-')

                    ax1.set_ylabel('PM10 (µg/m3)')

                    ############################


                    current_labels = []

                    for i in range(len(df)):

                        t0 = df.iloc[i]['timestamp']
                        t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                        cl = df.iloc[i]['activity']

                        color = get_color(cl)
                        if start_datetime <= t0 <= end_datetime:
                            if cl in current_labels:
                                ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                            else:
                                current_labels += [cl]
                                ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)


                    ax3.plot(data.BC, 'm-')

                    ax3.set_ylabel('BC (ng/m3)')
                    ###########################

                    for idx, row in data[:-1].iterrows():

                        t0 = idx
                        t1 = idx+pd.Timedelta(minutes=5)
                        cl = row['activity']

                        color = get_color(cl)
                        if start_datetime <= t0 <= end_datetime:
                            if cl in current_labels:
                                ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                            else:
                                current_labels += [cl]
                                ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

                    plt.xlim(start_datetime, end_datetime)

                    ax4.set_ylabel('Declared')
                    ax4.set_yticklabels([])

                    ###########################

                    hl=[]
                    for key,value in colors.items():  
                        if key!='Indoor' and key!='Cinéma' and key!='Rue' and key!='Données indisponibles':
                            hl.append((mpatches.Patch(color=get_color(key)),key))

                    handles2, labels2 = zip(*hl)
                    ax4.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0)
                    user = data.participant_virtual_id.iloc[0]

                    ax1.set_title('Participant %s' % participant_virtual_id)

                    plt.savefig('Context_RECORD/'+str(participant_virtual_id)+'-CONTEXT.png', bbox_inches = 'tight', pad_inches=.25)

                    #*****************************************************************************************
                else:
                    if len(data.dropna(subset=['NO2']))>0 and len(data.dropna(subset=['PM10']))>0 : #NO2 not found
                        fig, ( ax1, ax3, ax4) = plt.subplots(3, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [3,3,1]})

                        # TODO: add warning if days between start_datetime and end_datetime do not overlap with df
                        df['timestamp']=pd.to_datetime(df['timestamp'])
                        if start_datetime is None:
                            start_datetime = df[:-1]['timestamp'].min()
                        if end_datetime is None:
                            end_datetime = df[:-1]['timestamp'].max()

                        current_labels = []

                        for i in range(len(df)):

                            t0 = df.iloc[i]['timestamp']
                            t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                            cl = df.iloc[i]['activity']

                            color = get_color(cl)
                            if start_datetime <= t0 <= end_datetime:
                                if cl in current_labels:
                                    ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                else:
                                    current_labels += [cl]
                                    ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)
                        plt.xlim(start_datetime, end_datetime)

                        ax1.plot(data.PM10, 'k-')

                        ax1.set_ylabel('PM10 (µg/m3)')

                        ############################


                        current_labels = []

                        for i in range(len(df)):

                            t0 = df.iloc[i]['timestamp']
                            t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                            cl = df.iloc[i]['activity']

                            color = get_color(cl)
                            if start_datetime <= t0 <= end_datetime:
                                if cl in current_labels:
                                    ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                else:
                                    current_labels += [cl]
                                    ax3.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)


                        ax3.plot(data.NO2, 'm-')

                        ax3.set_ylabel('NO2 (ng/m3)')
                        ###########################

                        for idx, row in data[:-1].iterrows():

                            t0 = idx
                            t1 = idx+pd.Timedelta(minutes=5)
                            cl = row['activity']

                            color = get_color(cl)
                            if start_datetime <= t0 <= end_datetime:
                                if cl in current_labels:
                                    ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                else:
                                    current_labels += [cl]
                                    ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

                        plt.xlim(start_datetime, end_datetime)

                        ax4.set_ylabel('Declared')
                        ax4.set_yticklabels([])

                        ###########################

                        hl=[]
                        for key,value in colors.items():  
                            if key!='Indoor' and key!='Cinéma' and key!='Rue' and key!='Données indisponibles':
                                hl.append((mpatches.Patch(color=get_color(key)),key))

                        handles2, labels2 = zip(*hl)
                        ax4.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0)
                        user = data.participant_virtual_id.iloc[0]

                        ax1.set_title('Participant %s' % participant_virtual_id)

                        plt.savefig('Context_RECORD/'+str(participant_virtual_id)+'-CONTEXT.png', bbox_inches = 'tight', pad_inches=.25)


                        #*****************************************************************************************
                    else:
                        if len(data.dropna(subset=['PM10']))>0 : #NO2 and BC not found
                            fig, ( ax1, ax4) = plt.subplots(2, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [3,1]})

                            # TODO: add warning if days between start_datetime and end_datetime do not overlap with df
                            df['timestamp']=pd.to_datetime(df['timestamp'])
                            if start_datetime is None:
                                start_datetime = df[:-1]['timestamp'].min()
                            if end_datetime is None:
                                end_datetime = df[:-1]['timestamp'].max()

                            current_labels = []

                            for i in range(len(df)):

                                t0 = df.iloc[i]['timestamp']
                                t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                                cl = df.iloc[i]['activity']

                                color = get_color(cl)
                                if start_datetime <= t0 <= end_datetime:
                                    if cl in current_labels:
                                        ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                    else:
                                        current_labels += [cl]
                                        ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)
                            plt.xlim(start_datetime, end_datetime)

                            ax1.plot(data.PM10, 'k-')

                            ax1.set_ylabel('PM10 (µg/m3)')

                            ############################


                            for idx, row in data[:-1].iterrows():

                                t0 = idx
                                t1 = idx+pd.Timedelta(minutes=5)
                                cl = row['activity']

                                color = get_color(cl)
                                if start_datetime <= t0 <= end_datetime:
                                    if cl in current_labels:
                                        ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                    else:
                                        current_labels += [cl]
                                        ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

                            plt.xlim(start_datetime, end_datetime)

                            ax4.set_ylabel('Declared')
                            ax4.set_yticklabels([])

                            ###########################

                            hl=[]
                            for key,value in colors.items():  
                                if key!='Indoor' and key!='Cinéma' and key!='Rue' and key!='Données indisponibles':
                                    hl.append((mpatches.Patch(color=get_color(key)),key))

                            handles2, labels2 = zip(*hl)
                            ax4.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0)
                            user = data.participant_virtual_id.iloc[0]

                            ax1.set_title('Participant %s' % participant_virtual_id)

                            plt.savefig('Context_RECORD/'+str(participant_virtual_id)+'-CONTEXT.png', bbox_inches = 'tight', pad_inches=.25)


                            #*****************************************************************************************
                        else:
                            if len(data.dropna(subset=['BC']))>0 : #NO2 and PM10 not found
                                fig, ( ax1, ax4) = plt.subplots(2, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [3,1]})

                                # TODO: add warning if days between start_datetime and end_datetime do not overlap with df
                                df['timestamp']=pd.to_datetime(df['timestamp'])
                                if start_datetime is None:
                                    start_datetime = df[:-1]['timestamp'].min()
                                if end_datetime is None:
                                    end_datetime = df[:-1]['timestamp'].max()

                                current_labels = []

                                for i in range(len(df)):

                                    t0 = df.iloc[i]['timestamp']
                                    t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                                    cl = df.iloc[i]['activity']

                                    color = get_color(cl)
                                    if start_datetime <= t0 <= end_datetime:
                                        if cl in current_labels:
                                            ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                        else:
                                            current_labels += [cl]
                                            ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)
                                plt.xlim(start_datetime, end_datetime)

                                ax1.plot(data.BC, 'k-')

                                ax1.set_ylabel('BC (ng/m3)')

                                ############################


                                for idx, row in data[:-1].iterrows():

                                    t0 = idx
                                    t1 = idx+pd.Timedelta(minutes=5)
                                    cl = row['activity']

                                    color = get_color(cl)
                                    if start_datetime <= t0 <= end_datetime:
                                        if cl in current_labels:
                                            ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                        else:
                                            current_labels += [cl]
                                            ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

                                plt.xlim(start_datetime, end_datetime)

                                ax4.set_ylabel('Declared')
                                ax4.set_yticklabels([])

                                ###########################

                                hl=[]
                                for key,value in colors.items():  
                                    if key!='Indoor' and key!='Cinéma' and key!='Rue' and key!='Données indisponibles':
                                        hl.append((mpatches.Patch(color=get_color(key)),key))

                                handles2, labels2 = zip(*hl)
                                ax4.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0)
                                user = data.participant_virtual_id.iloc[0]

                                ax1.set_title('Participant %s' % participant_virtual_id)

                                plt.savefig('Context_RECORD/'+str(participant_virtual_id)+'-CONTEXT.png', bbox_inches = 'tight', pad_inches=.25)

                                #*****************************************************************************************
                            else:
                                if len(data.dropna(subset=['NO2']))>0 : #BC and PM10 not found
                                    fig, ( ax1, ax4) = plt.subplots(2, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [3,1]})

                                    # TODO: add warning if days between start_datetime and end_datetime do not overlap with df
                                    df['timestamp']=pd.to_datetime(df['timestamp'])
                                    if start_datetime is None:
                                        start_datetime = df[:-1]['timestamp'].min()
                                    if end_datetime is None:
                                        end_datetime = df[:-1]['timestamp'].max()

                                    current_labels = []

                                    for i in range(len(df)):

                                        t0 = df.iloc[i]['timestamp']
                                        t1 = df.iloc[i]['timestamp']+pd.Timedelta(minutes=5)
                                        cl = df.iloc[i]['activity']

                                        color = get_color(cl)
                                        if start_datetime <= t0 <= end_datetime:
                                            if cl in current_labels:
                                                ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                            else:
                                                current_labels += [cl]
                                                ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)
                                    plt.xlim(start_datetime, end_datetime)

                                    ax1.plot(data.BC, 'k-')

                                    ax1.set_ylabel('NO2 (µg/m3)')

                                    ############################


                                    for idx, row in data[:-1].iterrows():

                                        t0 = idx
                                        t1 = idx+pd.Timedelta(minutes=5)
                                        cl = row['activity']

                                        color = get_color(cl)
                                        if start_datetime <= t0 <= end_datetime:
                                            if cl in current_labels:
                                                ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
                                            else:
                                                current_labels += [cl]
                                                ax4.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

                                    plt.xlim(start_datetime, end_datetime)

                                    ax4.set_ylabel('Déclaratif')
                                    ax4.set_yticklabels([])

                                    ###########################

                                    hl=[]
                                    for key,value in colors.items():  
                                        if key!='Indoor' and key!='Cinéma' and key!='Rue' and key!='Données indisponibles':
                                            hl.append((mpatches.Patch(color=get_color(key)),key))

                                    handles2, labels2 = zip(*hl)
                                    ax4.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0)
                                    user = data.participant_virtual_id.iloc[0]

                                    ax1.set_title('Participant %s' % participant_virtual_id)

                                    plt.savefig('Context_RECORD/'+str(participant_virtual_id)+'-CONTEXT.png', bbox_inches = 'tight', pad_inches=.25)

                                    #*****************************************************************************************

                                    
def drop_one_row(df,df1,columns_to_drop=["Temperature","Humidity","NO2","BC","PM1.0","PM2.5","PM10"]):   
    if len(df1)!=len(df):
        df1=df1.drop(df.index[0])
    df=df1.dropna(subset=columns_to_drop)    
    
    return df

def remove_column(df,columns_to_drop=["Temperature","Humidity","NO2","BC","PM1.0","PM2.5","PM10","Speed"]):
    for c in columns_to_drop:
        if len(df.dropna(subset=[c]))==1:
            print(c)
            return [s for s in columns_to_drop if s != c]
    return columns_to_drop

def predict_view(model,data):
    predicted=[]
    predicted_prob=[]    
    if len(data)==0:
        return predicted,predicted_prob
    for i in range(len(data)):
        print(i,data[i])
        p=model.predict([data[i]])
        p_prob=model.predict_proba(np.asarray([data[i]]))
        predicted.append(p)
        predicted_prob.append(p_prob)
    return predicted,predicted_prob

from datetime import datetime
def correct_annotation(dictionary): ###Classes as dictionary
    new_dictionary={}
#     prev_key=None
    
    for key,value in dictionary.items():
        if value == 'Domicile' or value=='Domicile_tiers':
            dateTime=datetime.strptime(key,'%Y-%m-%d %H:%M:%S')
            weekday=dateTime.weekday()
#             print(weekday)
            if weekday<5:
                time_in_day=dateTime.time()
                if datetime.strptime('08:30:00','%H:%M:%S').time()<=time_in_day and time_in_day<=datetime.strptime('17:30:00','%H:%M:%S').time():
                    new_dictionary[key]='Bureau'
                else:
                    new_dictionary[key]='Domicile'
            else:
                new_dictionary[key]='Domicile'
        else:            
            if value== 'Bureau' or value== 'Bureau_tiers':
                dateTime=datetime.strptime(key,'%Y-%m-%d %H:%M:%S')
                weekday=dateTime.weekday()
                if weekday<5:
                    time_in_day=dateTime.time()
                    if datetime.strptime('08:00:00','%H:%M:%S').time()>=time_in_day or time_in_day>=datetime.strptime('17:30:00','%H:%M:%S').time():
                        new_dictionary[key]='Domicile'
                    else:
                        new_dictionary[key]='Bureau'
                else:
                    new_dictionary[key]='Domicile'
            else:
                if value == "Voiture_arrêt":
                    new_dictionary[key]="Voiture"
                else:
                    if value == "Train_arrêt":
                        new_dictionary[key]="Train"
                    else:                        
                        new_dictionary[key]=value
        
        if value!='Domicile':
            dateTime=datetime.strptime(key,'%Y-%m-%d %H:%M:%S')
            time_in_day=dateTime.time()
            if time_in_day>=datetime.strptime('01:00:00','%H:%M:%S').time() and time_in_day<=datetime.strptime('05:00:00','%H:%M:%S').time():
                new_dictionary[key]='Domicile'
            else:
                new_dictionary[key]=value
    return new_dictionary


def predict_labels_raw_data_RF(dfs,model_path='./models/new_models/',classes_removed=False):
    Temperature_model,Humidity_model,NO2_model,BC_model,PM1_model,PM25_model,PM10_model=load_models(model_path=model_path,basic_models=True)
    labels_dictionay={}
    labels_Temperature={}
    labels_Humidity={}
    labels_NO2={}
    labels_BC={}
    labels_PM1={}
    labels_PM25={}
    labels_PM10={}
    labels_Speed={}
    z=0

    for df_test in dfs:        
        if len(df_test.dropna(subset=["Speed"]))>1:
            
            Speed_model,multi_view_model,multi_view_model_without_BC,multi_view_model_without_NO2,multi_view_model_without_BC_NO2,multi_view_model_without_PMS,multi_view_model_without_BC_PMS,multi_view_model_only_Temperature_Humidity=load_models(model_path=model_path,with_speed=True)
            multi_view_model_only_BC_NO2,multi_view_model_only_BC_NO2_PMS,multi_view_model_only_BC_PMS,multi_view_model_only_NO2_PMS,multi_view_model_only_PMS, multi_view_model_without_Temperature,multi_view_model_without_Temperature_NO2,multi_view_model_without_Temperature_BC,multi_view_model_without_Temperature_NO2_BC,multi_view_model_without_Temperature_PMS,multi_view_model_only_Humidity_NO2,multi_view_model_only_Humidity_BC,multi_view_model_without_NO2_PMS,multi_view_model_without_Humidity_PMS,multi_view_model_without_Humidity_BC,multi_view_model_only_Temperature_NO2,multi_view_model_without_Humidity_NO2,multi_view_model_only_Temperature_BC,multi_view_model_only_Temperature_PMS,multi_view_model_only_Humidity_PMS,multi_view_model_only_Humidity_Speed,multi_view_model_only_BC_Speed,multi_view_model_only_NO2_Speed,multi_view_model_only_Temperature_Speed=load_new_models(model_path=model_path,with_speed=True)
            
            if len(df_test)>0:
                Temperature_data,Humidity_data,NO2_data,BC_data,PM1_data,PM25_data,PM10_data,Speed_data=prepare_set_with_speed(df_test)
            if len(df_test)>0:

#                 predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,calculate_mean_std(Temperature_data))
                predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,Temperature_data)
                print("Temperature")

#                 predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,calculate_mean_std(Humidity_data))
                predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,Humidity_data)
                print("Humidity")

#                 predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,calculate_mean_std(NO2_data))
                predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,NO2_data)
                print("NO2")

#                 predicted_BC,predicted_proba_BC=predict_view(BC_model,calculate_mean_std(BC_data))
                predicted_BC,predicted_proba_BC=predict_view(BC_model,BC_data)
                print("BC")

#                 predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,calculate_mean_std(PM1_data))
                predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,PM1_data)
                print("PM1.0")

#                 predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,calculate_mean_std(PM25_data))
                predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,PM25_data)
                print("PM2.5")

#                 predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,calculate_mean_std(PM10_data))
                predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,PM10_data)
                print("PM10")

#                 predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,calculate_mean_std(Speed_data))
                predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,Speed_data)
                print("Speed")

                print("counter",z)
                print("===============================================================================")
                z+=1

                new_data = prepare_new_dataset_with_speed(predicted_Temperature,predicted_proba_Temperature,predicted_Humidity,predicted_proba_Humidity,predicted_NO2,predicted_proba_NO2,predicted_BC,predicted_proba_BC,predicted_PM1,predicted_proba_PM1,predicted_PM25,predicted_proba_PM25,predicted_PM10,predicted_proba_PM10,predicted_Speed,predicted_proba_Speed,classes_removed=classes_removed)
                print(new_data)
                
                if len(predicted_Temperature)>0:
                    #Temperature is found
                    if len(predicted_Humidity)>0:
                        #Temperature and Humidity are found
                        if len(predicted_NO2)>0:
                            #Temperature, Humidity, NO2 are found
                            if len(predicted_BC)>0:
                                #Temperature, Humidity, NO2, BC are found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, Humidity, NO2, BC, PMS are found
                                    if len(predicted_Speed)>0:
                                        #Temperature, Humidity, NO2, BC, PMS and Speed are found
                                        labels=multi_view_model.predict(new_data)
                                    else:
                                        labels=[-1]
                                else:
                                    #Temperature, Humidity, NO2, BC, are found but PMS not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_without_PMS.predict(new_data)
                            else:
                                #Temperature, Humidity, NO2 are found but BC Not found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, Humidity, NO2, PMS are found but BC Not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_without_BC.predict(new_data)
                                    else:
                                        labels=[-1]
                                else:
                                    #Temperature, Humidity, NO2 are found but BC, PMS Not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_without_BC_PMS.predict(new_data)
                                    else:
                                        labels=[-1]
                        else:
                            #Temperature and Humidity are found but NO2 not found
                            if len(predicted_BC)>0:
                                #Temperature, Humidity, BC are found but NO2 not found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, Humidity, BC, PMS are found but NO2 not found
                                    if len(predicted_Speed)>0:
                                        labels= multi_view_model_without_NO2.predict(new_data)
                                    else:
                                        labels=[-1]
                                else:
                                    #Temperature, Humidity, BC are found but NO2, PMS not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_without_NO2_PMS.predict(new_data)
                                    else:
                                        labels=[-1]
                            else:
                                #Temperature, Humidity are found but NO2 and BC not found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, Humidity, PMS are found but NO2 and BC not found
                                    if len(predicted_Speed):
                                        labels=multi_view_model_without_BC_NO2.predict(new_data)
                                    else:
                                        labels=[-1]
                                else:
                                    #Temperature, Humidityare found but NO2 and BC, PMS not found
                                    if len(predicted_Speed):
                                        labels=multi_view_model_only_Temperature_Humidity.predict(new_data)
                                    else:
                                        labels=[-1]
                                    
                    else:
                        #Temperature is found but Humidity not found
                        if len(predicted_NO2)>0:
                            #Temperature, NO2 is found but Humidity not found
                            if len(predicted_BC)>0:
                                #Temperature, NO2, BC is found but Humidity not found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, NO2, BC, PMS is found but Humidity not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_without_Humidity.predict(new_data)
                                    else:
                                        labels=[-1]
                                else:
                                    #Temperature, NO2, BC is found but Humidity, PMS not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_without_Humidity_PMS.predict(new_data) #to be trained1
                                    else:
                                        labels=[-1]
                                    
                            else:
                                #Temperature, NO2 is found but Humidity, BC not found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, NO2, PMS is found but Humidity, BC not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_without_Humidity_BC.predict(new_data) #to be trained1
                                    else:
                                        labels=[-1]
                                else:
                                    #Temperature, NO2 is found but Humidity, BC, PMS not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_Temperature_NO2.predict(new_data) #to be trained1
                                    else:
                                        labels=[-1]                                    
                        else:
                            #Temperature is found but Humidity, NO2 not found
                            if len(predicted_BC)>0:
                                #Temperature, BC is found but Humidity, NO2 not found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, BC, PMS is found but Humidity, NO2 not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_without_Humidity_NO2.predict(new_data) #to be trained1
                                    else:
                                        labels=[-1]
                                else:
                                    #Temperature, BC is found but Humidity, NO2, PMS not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_Temperature_BC.predict(new_data) #to be trained1
                                    else:
                                        labels=[-1]
                            else:
                                #Temperature is found but Humidity, NO2, BC not found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, PMS is found but Humidity, NO2, BC not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_Temperature_PMS.predict(new_data)#to be trained1
                                    else:
                                        labels=[-1]
                                else:
                                    #Temperature is found but Humidity, NO2, BC, PMS not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_Temperature_Speed.predict(new_data) #to be trained1
                                    else:
                                        labels=[-1]
                            
                else:
                    #Temperature is not found
                    if len(predicted_Humidity)>0:
                        #Humidity found and Temperature not found
                        if len(predicted_NO2)>0:
                            #Humidity, NO2 found and Temperature not found
                            if len(predicted_BC)>0:
                                #Humidity, NO2, BC found and Temperature not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #Humidity, NO2, BC, PMS found and Temperature not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_without_Temperature.predict(new_data)
                                    else:
                                        labels=[-1]
                                else:
                                    #Humidity, NO2, BC found and Temperature, PMS not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_without_Temperature_PMS.predict(new_data)
                                    else:
                                        labels=[-1]
                            else:
                                #Humidity, NO2 found and Temperature, BC not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #Humidity, NO2, PMS found and Temperature, BC not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_without_Temperature_BC.predict(new_data)
                                    else:
                                        labels=[-1]
                                else:
                                    #Humidity, NO2 found and Temperature, BC, PMS not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_Humidity_NO2.predict(new_data)
                                    else:
                                        labels=[-1]
                        else:
                            #Humidity found but Temperature and NO2 not found
                            if len(predicted_BC)>0:
                                #Humidity, BC found but Temperature and NO2 not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #Humidity, BC, PMS found but Temperature and NO2 not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_without_Temperature_NO2.predict(new_data)
                                    else:
                                        labels=[-1]
                                else:
                                    #Humidity, BC found but Temperature and NO2, PMS not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_Humidity_BC.predict(new_data)
                                    else:
                                        labels=[-1]
                            else:
                                #Humidity found but Temperature, BC, NO2 not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #Humidity, PMS found but Temperature, BC, NO2 not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_Humidity_PMS.predict(new_data)# to be trained1
                                    else:
                                        labels=[-1]
                                else:
                                    #Humidity found but Temperature, BC, NO2, PMS not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_Humidity_Speed.predict(new_data) #to be trained1
                                    else:
                                        labels=[-1]

                    else:
                        #Temperature and Humidity not found
                        if len(predicted_NO2)>0:
                            #NO2 found but Temperature, Humidity not found
                            if len(predicted_BC)>0:
                                #NO2, BC found but Temperature, Humidity not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #NO2, BC, PMS found but Temperature, Humidity not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_BC_NO2_PMS.predict(new_data)
                                    else:
                                        labels=[-1]
                                else:
                                    #NO2, BC found but Temperature, Humidity,PMS not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_BC_NO2.predict(new_data)
                                    else:
                                        labels=[-1]
                            else:
                                #NO2 found but Temperature, Humidity, BC not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #NO2,PMS found but Temperature, Humidity, BC not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_NO2_PMS.predict(new_data)
                                    else:
                                        labels=[-1]
                                else:
                                    #NO2 found but Temperature, Humidity, BC, PMS not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_NO2_Speed.predict(new_data) #to be trained1
                                    else:
                                        labels=[-1]
                        else:
                            #Temperature, Humidity, NO2 not found
                            if len(predicted_BC)>0:
                                #BC found but Temperature, Humidity, NO2 not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #BC, PMS found but Temperature, Humidity, NO2 not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_BC_PMS.predict(new_data)
                                    else:
                                        labels=[-1]
                                else:
                                    #BC found but Temperature, Humidity, NO2, PMS not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_BC_Speed.predict(new_data) #to be trained1
                                    else:
                                        labels=[-1]
                            else:
                                #Temperature, Humidity, NO2, BC not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #PMS found but Temperature, Humidity, NO2, BC not found
                                    if len(predicted_Speed)>0:
                                        labels=multi_view_model_only_PMS.predict(new_data)
                                    else:
                                        labels=[-1]
                                else:
                                    #Temperature, Humidity, NO2, BC, PMS not found
                                    if len(predicted_Speed)>0:
                                        labels=predicted_Speed
                                    else:
                                        labels=[-1]
                                                                     
                                                                     



#                 labels_dictionay[str(df_test["time"].iloc[0])]=labels.tolist()
                if len(predicted_Temperature)==0 and len(predicted_Humidity)==0 and len(predicted_BC)==0 and len(predicted_NO2)==0 and len(predicted_PM1)==0 and len(predicted_PM25)==0 and len(predicted_PM10)==0 and len(predicted_Speed)==0:
                    labels=[-1]
                    
                time=str(df_test["time"].iloc[0])    
                
                if type(labels)=='numpy.ndarray':
                    labels_dictionay[time]=labels.tolist()
                else:
                    labels_dictionay[time]=labels
                
                if len(predicted_Temperature)>0:
                    labels_Temperature[time]=predicted_Temperature[0].tolist()
                else:
                    labels_Temperature[time]=[-1]
                
                if len(predicted_Humidity)>0:
                    labels_Humidity[time]=predicted_Humidity[0].tolist()
                else:
                    labels_Humidity[time]=[-1]
                    
                if len(predicted_NO2)>0:    
                    labels_NO2[time]=predicted_NO2[0].tolist()
                else:
                    labels_NO2[time]=[-1]
                    
                if len(predicted_BC)>0:    
                    labels_BC[time]=predicted_BC[0].tolist()
                else:
                    labels_BC[time]=[-1]
                
                if len(predicted_PM1)>0:    
                    labels_PM1[time]=predicted_PM1[0].tolist()
                else:
                    labels_PM1[time]=[-1]
                
                if len(predicted_PM25)>0:    
                    labels_PM25[time]=predicted_PM25[0].tolist()
                else:
                    labels_PM25[time]=[-1]
                
                if len(predicted_PM10)>0:    
                    labels_PM10[time]=predicted_PM10[0].tolist()
                else:
                    labels_PM10[time]=[-1]
                    
                if len(predicted_Speed)>0:    
                    labels_Speed[time]=predicted_Speed[0].tolist()
                else:
                    labels_Speed[time]=[-1]
#                 labels_dictionay[str(df_test["time"].iloc[0])]=[most_frequent(new_data[0])]
        else:            
            multi_view_model,multi_view_model_without_BC,multi_view_model_without_NO2,multi_view_model_without_BC_NO2,multi_view_model_without_PMS,multi_view_model_without_BC_PMS,multi_view_model_only_Temperature_Humidity=load_models(model_path=model_path,with_speed=False,basic_models=False)
            multi_view_model_only_BC_NO2,multi_view_model_only_BC_NO2_PMS,multi_view_model_only_BC_PMS,multi_view_model_only_NO2_PMS,multi_view_model_only_PMS, multi_view_model_without_Temperature,multi_view_model_without_Temperature_NO2,multi_view_model_without_Temperature_BC,multi_view_model_without_Temperature_NO2_BC,multi_view_model_without_Temperature_PMS,multi_view_model_only_Humidity_NO2,multi_view_model_only_Humidity_BC,multi_view_model_without_NO2_PMS,multi_view_model_without_Humidity_PMS,multi_view_model_without_Humidity_BC,multi_view_model_only_Temperature_NO2,multi_view_model_without_Humidity_NO2,multi_view_model_only_Temperature_BC,multi_view_model_only_Temperature_PMS,multi_view_model_only_Humidity_PMS=load_new_models(model_path,with_speed=False)
            
            if len(df_test)>1:
                Temperature_data,Humidity_data,NO2_data,BC_data,PM1_data,PM25_data,PM10_data=prepare_set(df_test)
            if len(df_test)>1:
#                 print("T",calculate_mean_std(Temperature_data))
#                 predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,calculate_mean_std(Temperature_data))
                predicted_Temperature,predicted_proba_Temperature=predict_view(Temperature_model,Temperature_data)
                print("Temperature")
        
#                 predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,calculate_mean_std(Humidity_data))
                predicted_Humidity,predicted_proba_Humidity=predict_view(Humidity_model,Humidity_data)
                print("Humidity")

#                 predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,calculate_mean_std(NO2_data))
                predicted_NO2,predicted_proba_NO2=predict_view(NO2_model,NO2_data)
                print("NO2")

#                 predicted_BC,predicted_proba_BC=predict_view(BC_model,calculate_mean_std(BC_data))
                predicted_BC,predicted_proba_BC=predict_view(BC_model,BC_data)
                print("BC")
                

#                 predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,calculate_mean_std(PM1_data))
                predicted_PM1,predicted_proba_PM1=predict_view(PM1_model,PM1_data)
                print("PM1.0")

#                 predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,calculate_mean_std(PM25_data))
                predicted_PM25,predicted_proba_PM25=predict_view(PM25_model,PM25_data)
                print("PM2.5")

#                 predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,calculate_mean_std(PM10_data))
                predicted_PM10,predicted_proba_PM10=predict_view(PM10_model,PM10_data)
                print("PM10")

#                 predicted_Speed,predicted_proba_Speed=predict_view(Speed_model,Speed_data)
#                 print("Speed")

                print("counter",z)
                print("===============================================================================")
                z+=1

                new_data = prepare_new_dataset(predicted_Temperature,predicted_proba_Temperature,predicted_Humidity,predicted_proba_Humidity,predicted_NO2,predicted_proba_NO2,predicted_BC,predicted_proba_BC,predicted_PM1,predicted_proba_PM1,predicted_PM25,predicted_proba_PM25,predicted_PM10,predicted_proba_PM10,classes_removed=classes_removed)
                print(new_data)

                
                
                if len(predicted_Temperature)>0:
                    #Temperature is found
                    if len(predicted_Humidity)>0:
                        #Temperature and Humidity are found
                        if len(predicted_NO2)>0:
                            #Temperature, Humidity, NO2 are found
                            if len(predicted_BC)>0:
                                #Temperature, Humidity, NO2, BC are found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, Humidity, NO2, BC, PMS are found
                                    labels=multi_view_model.predict(new_data)
                                else:
                                    #Temperature, Humidity, NO2, BC, are found but PMS not found
                                    labels=multi_view_model_without_PMS.predict(new_data)
                            else:
                                #Temperature, Humidity, NO2 are found but BC Not found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, Humidity, NO2, PMS are found but BC Not found
                                    labels=multi_view_model_without_BC.predict(new_data)                                    
                                else:
                                    #Temperature, Humidity, NO2 are found but BC, PMS Not found
                                    labels=multi_view_model_without_BC_PMS.predict(new_data)
                        else:
                            #Temperature and Humidity are found but NO2 not found
                            if len(predicted_BC)>0:
                                #Temperature, Humidity, BC are found but NO2 not found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, Humidity, BC, PMS are found but NO2 not found
                                    labels= multi_view_model_without_NO2.predict(new_data)
                                else:
                                    #Temperature, Humidity, BC are found but NO2, PMS not found
                                    labels=multi_view_model_without_NO2_PMS.predict(new_data)
                            else:
                                #Temperature, Humidity are found but NO2 and BC not found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, Humidity, PMS are found but NO2 and BC not found
#                                     print("new_data",new_data)
#                                     print('type',type(multi_view_model_without_BC_NO2))
                                    labels=multi_view_model_without_BC_NO2.predict(new_data)
                                else:
                                    #Temperature, Humidityare found but NO2 and BC, PMS not found
                                    labels=multi_view_model_only_Temperature_Humidity.predict(new_data)
                                    
                    else:
                        #Temperature is found but Humidity not found
                        if len(predicted_NO2)>0:
                            #Temperature, NO2 is found but Humidity not found
                            if len(predicted_BC)>0:
                                #Temperature, NO2, BC is found but Humidity not found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, NO2, BC, PMS is found but Humidity not found
                                    labels=multi_view_model_without_Humidity.predict(new_data)
                                else:
                                    #Temperature, NO2, BC is found but Humidity, PMS not found
                                    labels=multi_view_model_without_Humidity_PMS.predict(new_data) #to be trained1
                                    
                            else:
                                #Temperature, NO2 is found but Humidity, BC not found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, NO2, PMS is found but Humidity, BC not found
                                    labels=multi_view_model_without_Humidity_BC.predict(new_data) #to be trained1
                                else:
                                    #Temperature, NO2 is found but Humidity, BC, PMS not found
                                    labels=multi_view_model_only_Temperature_NO2.predict(new_data) #to be trained1
                                    
                        else:
                            #Temperature is found but Humidity, NO2 not found
                            if len(predicted_BC)>0:
                                #Temperature, BC is found but Humidity, NO2 not found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, BC, PMS is found but Humidity, NO2 not found
                                    labels=multi_view_model_without_Humidity_NO2.predict(new_data) #to be trained1
                                else:
                                    #Temperature, BC is found but Humidity, NO2, PMS not found
                                    labels=multi_view_model_only_Temperature_BC.predict(new_data) #to be trained1
                            else:
                                #Temperature is found but Humidity, NO2, BC not found
                                if len(predicted_PM1)>0 and len(predicted_PM25)>0 and len(predicted_PM10)>0:
                                    #Temperature, PMS is found but Humidity, NO2, BC not found
                                    labels=multi_view_model_only_Temperature_PMS.predict(new_data)#to be trained1
                                else:
                                    #Temperature is found but Humidity, NO2, BC, PMS not found
                                    labels=predicted_Temperature
                            
                else:
                    #Temperature is not found
                    if len(predicted_Humidity)>0:
                        #Humidity found and Temperature not found
                        if len(predicted_NO2)>0:
                            #Humidity, NO2 found and Temperature not found
                            if len(predicted_BC)>0:
                                #Humidity, NO2, BC found and Temperature not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #Humidity, NO2, BC, PMS found and Temperature not found
                                    labels=multi_view_model_without_Temperature.predict(new_data)
                                else:
                                    #Humidity, NO2, BC found and Temperature, PMS not found
                                    labels=multi_view_model_without_Temperature_PMS.predict(new_data)
                            else:
                                #Humidity, NO2 found and Temperature, BC not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #Humidity, NO2, PMS found and Temperature, BC not found
                                    labels=multi_view_model_without_Temperature_BC.predict(new_data)
                                else:
                                    #Humidity, NO2 found and Temperature, BC, PMS not found
                                    labels=multi_view_model_only_Humidity_NO2.predict(new_data)
                        else:
                            #Humidity found but Temperature and NO2 not found
                            if len(predicted_BC)>0:
                                #Humidity, BC found but Temperature and NO2 not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #Humidity, BC, PMS found but Temperature and NO2 not found
                                    labels=multi_view_model_without_Temperature_NO2.predict(new_data)
                                else:
                                    #Humidity, BC found but Temperature and NO2, PMS not found
                                    labels=multi_view_model_only_Humidity_BC.predict(new_data)
                            else:
                                #Humidity found but Temperature, BC, NO2 not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #Humidity, PMS found but Temperature, BC, NO2 not found
                                    labels=multi_view_model_only_Humidity_PMS.predict(new_data)# to be trained1
                                else:
                                    #Humidity found but Temperature, BC, NO2, PMS not found
                                    labels=predicted_Humidity

                    else:
                        #Temperature and Humidity not found
                        if len(predicted_NO2)>0:
                            #NO2 found but Temperature, Humidity not found
                            if len(predicted_BC)>0:
                                #NO2, BC found but Temperature, Humidity not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #NO2, BC, PMS found but Temperature, Humidity not found
                                    labels=multi_view_model_only_BC_NO2_PMS.predict(new_data)
                                else:
                                    #NO2, BC found but Temperature, Humidity,PMS not found
                                    labels=multi_view_model_only_BC_NO2.predict(new_data)
                            else:
                                #NO2 found but Temperature, Humidity, BC not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #NO2,PMS found but Temperature, Humidity, BC not found
                                    labels=multi_view_model_only_NO2_PMS.predict(new_data)
                                else:
                                    #NO2 found but Temperature, Humidity, BC, PMS not found
                                    labels=predicted_NO2
                        else:
                            #Temperature, Humidity, NO2 not found
                            if len(predicted_BC)>0:
                                #BC found but Temperature, Humidity, NO2 not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #BC, PMS found but Temperature, Humidity, NO2 not found
                                    labels=multi_view_model_only_BC_PMS.predict(new_data)
                                else:
                                    #BC found but Temperature, Humidity, NO2, PMS not found
                                    labels=predicted_BC
                            else:
                                #Temperature, Humidity, NO2, BC not found
                                if len(predicted_PM1)>0 and len(predicted_PM10)>0 and len(predicted_PM25)>0:
                                    #PMS found but Temperature, Humidity, NO2, BC not found
                                    labels=multi_view_model_only_PMS.predict(new_data)
#                                 else:
#                                     #Temperature, Humidity, NO2, BC, PMS not found
#                                     labels=predicted_Speed
                ###########


                if len(predicted_Temperature)==0 and len(predicted_Humidity)==0 and len(predicted_BC)==0 and len(predicted_NO2)==0 and len(predicted_PM1)==0 and len(predicted_PM25)==0 and len(predicted_PM10)==0:
                    labels=[-1]
                    

                time=str(df_test["time"].iloc[0])    
                if type(labels)=='numpy.ndarray':
                    labels_dictionay[time]=labels.tolist()
                else:
                    labels_dictionay[time]=labels
                    
                if len(predicted_Temperature)>0:
                    labels_Temperature[time]=predicted_Temperature[0].tolist()
                else:
                    labels_Temperature[time]=[-1]
                
                if len(predicted_Humidity)>0:
                    labels_Humidity[time]=predicted_Humidity[0].tolist()
                else:
                    labels_Humidity[time]=[-1]
                    
                if len(predicted_NO2)>0:    
                    labels_NO2[time]=predicted_NO2[0].tolist()
                else:
                    labels_NO2[time]=[-1]
                    
                if len(predicted_BC)>0:    
                    labels_BC[time]=predicted_BC[0].tolist()
                else:
                    labels_BC[time]=[-1]
                
                if len(predicted_PM1)>0:    
                    labels_PM1[time]=predicted_PM1[0].tolist()
                else:
                    labels_PM1[time]=[-1]
                
                if len(predicted_PM25)>0:    
                    labels_PM25[time]=predicted_PM25[0].tolist()
                else:
                    labels_PM25[time]=[-1]
                
                if len(predicted_PM10)>0:    
                    labels_PM10[time]=predicted_PM10[0].tolist()
                else:
                    labels_PM10[time]=[-1]
                
                labels_Speed[time]=[-1]
                
                
                
                
                    
                
#                 labels_dictionay[str(df_test["time"].iloc[0])]=[most_frequent(new_data[0])]
            
    
    return labels_dictionay,labels_Temperature,labels_Humidity,labels_NO2,labels_BC,labels_PM1,labels_PM25,labels_PM10,labels_Speed

def prepare_set(df1):
    df_Temperature=df1[["Temperature"]]
    df_Humidity=df1[["Humidity"]]
    df_NO2=df1[["NO2"]]
    df_BC=df1[["BC"]]
    df_PM1=df1[["PM1.0"]]
    df_PM25=df1[["PM2.5"]]
    df_PM10=df1[["PM10"]]
    
    if len(df_Temperature.dropna())>2:
        train_Temperature,_,_,_,_,_,_,_=fill_values(df_Temperature.dropna(),columns=["Temperature"])
    else:
        train_Temperature=[]
    
    if len(df_Humidity.dropna())>2:
        _,train_Humidity,_,_,_,_,_,_=fill_values(df_Humidity.dropna(),columns=["Humidity"])
    else:
        train_Humidity=[]
    
    if len(df_NO2.dropna())>2:
        _,_,train_NO2,_,_,_,_,_=fill_values(df_NO2.dropna(),columns=["NO2"])
    else:
        train_NO2=[]
    
    if len(df_BC.dropna())>2:
        _,_,_,train_BC,_,_,_,_=fill_values(df_BC.dropna(),columns=["BC"])
    else:
        train_BC=[]
        
    if len(df_PM1.dropna())>2:
        _,_,_,_,train_PM1,_,_,_=fill_values(df_PM1.dropna(),columns=["PM1.0"])
    else:
        train_PM1=[]
    
         
    if len(df_PM25.dropna())>2:
        _,_,_,_,_,train_PM25,_,_=fill_values(df_PM25.dropna(),columns=["PM2.5"])
    else:
        train_PM25=[]
    
    if len(df_PM10.dropna())>2:
        _,_,_,_,_,_,train_PM10,_=fill_values(df_PM10.dropna(),columns=["PM10"])
    else:
        train_PM10=[]
                                    
    
    return train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10


def prepare_set_with_speed(df1):
    df_Temperature=df1[["Temperature"]]
    df_Humidity=df1[["Humidity"]]
    df_NO2=df1[["NO2"]]
    df_BC=df1[["BC"]]
    df_PM1=df1[["PM1.0"]]
    df_PM25=df1[["PM2.5"]]
    df_PM10=df1[["PM10"]]
    df_Speed=df1[["Speed"]]
    
    
    if len(df_Temperature.dropna())>2:
        train_Temperature,_,_,_,_,_,_,_=fill_values(df_Temperature.dropna(),columns=["Temperature"])
    else:
        train_Temperature=[]
    
    if len(df_Humidity.dropna())>2:
        _,train_Humidity,_,_,_,_,_,_=fill_values(df_Humidity.dropna(),columns=["Humidity"])
    else:
        train_Humidity=[]
    
    if len(df_NO2.dropna())>2:
        _,_,train_NO2,_,_,_,_,_=fill_values(df_NO2.dropna(),columns=["NO2"])
    else:
        train_NO2=[]
    
    if len(df_BC.dropna())>2:
        _,_,_,train_BC,_,_,_,_=fill_values(df_BC.dropna(),columns=["BC"])
    else:
        train_BC=[]
        
    if len(df_PM1.dropna())>2:
        _,_,_,_,train_PM1,_,_,_=fill_values(df_PM1.dropna(),columns=["PM1.0"])
    else:
        train_PM1=[]
    
         
    if len(df_PM25.dropna())>2:
        _,_,_,_,_,train_PM25,_,_=fill_values(df_PM25.dropna(),columns=["PM2.5"])
    else:
        train_PM25=[]
    
    if len(df_PM10.dropna())>2:
        _,_,_,_,_,_,train_PM10,_=fill_values(df_PM10.dropna(),columns=["PM10"])
    else:
        train_PM10=[]
        
    if len(df_Speed.dropna())>2:
        _,_,_,_,_,_,_,train_Speed=fill_values(df_Speed.dropna(),columns=["Speed"])
    else:
        train_Speed=[]
    
    
    return train_Temperature,train_Humidity,train_NO2,train_BC,train_PM1,train_PM25,train_PM10,train_Speed

def correct_annotation__(dictionary): ###Classes as dictionary
    new_dictionary={}
#     prev_key=None
    
    for key,value in dictionary.items():
        if value == 'Domicile' or value=='Domicile_tiers':
            dateTime=datetime.strptime(key,'%Y-%m-%d %H:%M:%S')
            weekday=dateTime.weekday()
#             print(weekday)
            if weekday<5:
                time_in_day=dateTime.time()
                if datetime.strptime('09:00:00','%H:%M:%S').time()<=time_in_day and time_in_day<=datetime.strptime('18:00:00','%H:%M:%S').time():
                    new_dictionary[key]='Bureau'
                else:
                    new_dictionary[key]='Domicile'
            else:
                new_dictionary[key]='Domicile'
        else:            
            if value== 'Bureau' or value== 'Bureau_tiers':
                dateTime=datetime.strptime(key,'%Y-%m-%d %H:%M:%S')
                weekday=dateTime.weekday()
                if weekday<5:
                    time_in_day=dateTime.time()
                    if datetime.strptime('08:30:00','%H:%M:%S').time()>=time_in_day or time_in_day>=datetime.strptime('18:30:00','%H:%M:%S').time():
                        new_dictionary[key]='Domicile'
                    else:
                        new_dictionary[key]='Bureau'
                else:
                    new_dictionary[key]='Domicile'
            else:
                if value == "Voiture_arrêt":
                    new_dictionary[key]="Voiture"
                else:
                    if value == "Train_arrêt":
                        new_dictionary[key]="Train"
                    else:                        
                        new_dictionary[key]=value
        
        if value!='Domicile':
            dateTime=datetime.strptime(key,'%Y-%m-%d %H:%M:%S')
            time_in_day=dateTime.time()
            if time_in_day>=datetime.strptime('01:00:00','%H:%M:%S').time() and time_in_day<=datetime.strptime('05:00:00','%H:%M:%S').time():
                new_dictionary[key]='Domicile'
            else:
                new_dictionary[key]=value
        
    print("................")
    return new_dictionary
