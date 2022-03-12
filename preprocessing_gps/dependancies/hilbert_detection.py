#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# In[1]:

from utils import *
from db_connection import get_gps_hilbert,get_gps_hilbert_vgp


# In[7]:
def stop_hilbert(participant_virtual_id=987014104, density_threshold=100, hilbert_power=3, merge_min=2):
    
    df = get_gps_hilbert(participant_virtual_id=participant_virtual_id)
        
    df['stop'] = -1    
    df_count = df.groupby('hilbert').count().sort_values('participant_virtual_id',  ascending=False)
    df_count = df_count[df_count.participant_virtual_id>=density_threshold]
    
    i=1
    for item in df_count.index:
        if np.ceil(item/4**hilbert_power) in np.ceil(df[df.stop!=-1].hilbert.unique()/4**hilbert_power): 
            if len(list(set(df[(np.ceil(df.hilbert/4**hilbert_power)==np.ceil(item/4**hilbert_power)) & (df.stop!=-1)].stop)))==1:
                df.loc[df.hilbert == item ,'stop'] = list(set(df[(np.ceil(df.hilbert/4**hilbert_power)==np.ceil(item/4**hilbert_power)) & (df.stop!=-1)].stop))[0]
            else:
                print("Need to check the hilbert code: ", item)
        elif np.ceil(item/4**hilbert_power) not in np.ceil(df[df.stop!=-1].hilbert.unique()/4**hilbert_power): 
            if df.loc[df.hilbert == item].stop.unique() == -1:
                df.loc[df.hilbert == item ,'stop'] = i
                i+=1
    
    df_to_array = df.values
    list_df = []
    j=0
    lendf = len(df)-1
    for i in range(lendf):
        if df_to_array[i][6] != df_to_array[i+1][6]:
            list_df.append(df_to_array[j:i+1])
            j=i+1
    list_df.append(df_to_array[j:])

    for i in range(len(list_df)):
        list_df[i] = pd.DataFrame(list_df[i], columns=df.columns)
        for x in list_df[i].columns:
            list_df[i][x]=list_df[i][x].astype(df[x].dtypes.name)
    
    index = 1
    while index < len(list_df)-1:
        if list_df[index].time.max() - list_df[index].time.min() < datetime.timedelta(minutes=merge_min):
            list_df[index-1] = pd.concat([list_df[index-1], list_df[index], list_df[index+1]])
            list_df.pop(index+1)
            list_df.pop(index)
        else:
            index+=1
    
    for df in list_df:
        df['stop']=df.stop.median()
    
    dfs = pd.DataFrame()
    for item in list_df:
        dfs=pd.concat([dfs, item])
    
    dfs.reset_index(drop=True,inplace=True)
    
    dfs = encode_activity_stop(dfs)
    
    return dfs

# In[8]

def stop_hilbert_vgp(participant_virtual_id=9999915, density_threshold=100, hilbert_power=3, merge_min=2):
    
    df = get_gps_hilbert_vgp(participant_virtual_id=participant_virtual_id)
        
    df['stop'] = -1    
    df_count = df.groupby('hilbert').count().sort_values('participant_virtual_id',  ascending=False)
    df_count = df_count[df_count.participant_virtual_id>=density_threshold]
    
    i=1
    for item in df_count.index:
        if np.ceil(item/4**hilbert_power) in np.ceil(df[df.stop!=-1].hilbert.unique()/4**hilbert_power): 
            if len(list(set(df[(np.ceil(df.hilbert/4**hilbert_power)==np.ceil(item/4**hilbert_power)) & (df.stop!=-1)].stop)))==1:
                df.loc[df.hilbert == item ,'stop'] = list(set(df[(np.ceil(df.hilbert/4**hilbert_power)==np.ceil(item/4**hilbert_power)) & (df.stop!=-1)].stop))[0]
            else:
                print("Need to check the hilbert code: ", item)
        elif np.ceil(item/4**hilbert_power) not in np.ceil(df[df.stop!=-1].hilbert.unique()/4**hilbert_power): 
            if df.loc[df.hilbert == item].stop.unique() == -1:
                df.loc[df.hilbert == item ,'stop'] = i
                i+=1
    
    df_to_array = df.values
    list_df = []
    j=0
    lendf = len(df)-1
    for i in range(lendf):
        if df_to_array[i][6] != df_to_array[i+1][6]:
            list_df.append(df_to_array[j:i+1])
            j=i+1
    list_df.append(df_to_array[j:])

    for i in range(len(list_df)):
        list_df[i] = pd.DataFrame(list_df[i], columns=df.columns)
        for x in list_df[i].columns:
            list_df[i][x]=list_df[i][x].astype(df[x].dtypes.name)
    
    index = 1
    while index < len(list_df)-1:
        if list_df[index].time.max() - list_df[index].time.min() < datetime.timedelta(minutes=merge_min):
            list_df[index-1] = pd.concat([list_df[index-1], list_df[index], list_df[index+1]])
            list_df.pop(index+1)
            list_df.pop(index)
        else:
            index+=1
    
    for df in list_df:
        df['stop']=df.stop.median()
    
    dfs = pd.DataFrame()
    for item in list_df:
        dfs=pd.concat([dfs, item])
    
    dfs.reset_index(drop=True,inplace=True)
    
    dfs = encode_activity_stop(dfs)
    
    return dfs