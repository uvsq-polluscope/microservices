#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# In[1]:

from utils import *
from db_connection import get_gps, get_gps_2
from skmob.preprocessing import filtering
from skmob.preprocessing import detection
from skmob.preprocessing import clustering


# In[7]:
def stop_skmob(participant_virtual_id=987014104, stop_radius_factor=0.2, \
            minutes_for_a_stop=10.0,no_data_for_minutes= 600, cluster_radius_km=0.1):
    
    df = get_gps(participant_virtual_id=participant_virtual_id)
        
    tdf = skmob.TrajDataFrame(df,\
    latitude='lat', longitude='lon', datetime='time', user_id='participant_virtual_id')
    
    #df = filtering.filter(tdf, max_speed_kmh=max_speed_kmh)
    #GPS data is already filtered
    
    stdf = detection.stops(tdf, stop_radius_factor=stop_radius_factor, \
            minutes_for_a_stop=minutes_for_a_stop, spatial_radius_km=0.2,no_data_for_minutes= no_data_for_minutes,\
                       leaving_time=True)
    
    cstdf = clustering.cluster(stdf, cluster_radius_km=cluster_radius_km, min_samples=1)
    
    cstdf.sort_values('datetime', inplace=True)
     
    df['stops'] = -1
    
    lendf = len(cstdf)
    
    for i in range(lendf):

        df.loc[(df.time >= cstdf.datetime.iloc[i]) & (df.time <=cstdf.leaving_datetime.iloc[i]),'stops'] = cstdf.cluster.iloc[i]
    
    ### Encode ground truth into 0/1
    ### 1 stands for move and 0 stands for stop:
    df['_activity_'] = 1
    
    for i in range(len(df)):
        
        if df.loc[i].activity in indoor:
            df.loc[i ,'_activity_'] = 0
            
    ### Encode detected stops into 0/1
    ### 1 stands for move and 0 stands for stop:
    df['_stops_'] = np.where(df.stops==-1,1,0)
        
    return df

# In[8]

def stop_skmob_2(participant_virtual_id=987014104, max_speed_kmh=100., stop_radius_factor=0.2, \
            minutes_for_a_stop=10.0,no_data_for_minutes= 600, cluster_radius_km=0.1):
    
    df = get_gps_2(participant_virtual_id=participant_virtual_id)
        
    tdf = skmob.TrajDataFrame(df,\
    latitude='lat', longitude='lon', datetime='time', user_id='participant_virtual_id')
    
    #df = filtering.filter(tdf, max_speed_kmh=max_speed_kmh)
    #GPS data is already filtered
    
    stdf = detection.stops(tdf, stop_radius_factor=stop_radius_factor, \
            minutes_for_a_stop=minutes_for_a_stop, spatial_radius_km=0.2,no_data_for_minutes= no_data_for_minutes,\
                       leaving_time=True)
    
    cstdf = clustering.cluster(stdf, cluster_radius_km=cluster_radius_km, min_samples=1)
    
    cstdf.sort_values('datetime', inplace=True)
     
    df['stops'] = -1
    
    lendf = len(cstdf)
    
    for i in range(lendf):

        df.loc[(df.time >= cstdf.datetime.iloc[i]) & (df.time <=cstdf.leaving_datetime.iloc[i]),'stops'] = cstdf.cluster.iloc[i]
    
    ### Encode ground truth into 0/1
    ### 1 stands for move and 0 stands for stop:
    df['_activity_'] = 1
    
    for i in range(len(df)):
        
        if df.loc[i].activity in indoor:
            df.loc[i ,'_activity_'] = 0
            
    ### Encode detected stops into 0/1
    ### 1 stands for move and 0 stands for stop:
    df['_stops_'] = np.where(df.stops==-1,1,0)
        
    return df