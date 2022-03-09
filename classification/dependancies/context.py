#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from pathlib import Path
from dependancies.db_connection_context import get_pollutants, get_str_of_id
from math import radians, cos, sin, asin, sqrt
#%matplotlib notebook

engine_processed = create_engine('postgresql://postgres:postgres@192.168.33.123:5432/'+"processed_data")
engine_vgp = create_engine('postgresql://postgres:postgres@192.168.33.123:5432/'+"polluscopev5-last-version-2020")
#participants = pd.read_csv('c:/\Polluscope/Cartes_participants/list_participants.csv', infer_datetime_format=True, parse_dates=[3,4])

def get_activities(participant_virtual_id,kit_id):
    
    df_activities = pd.read_sql('''select "participants_vgp"."participant_virtual_id", "tabletActivityApp"."timestamp"::timestamp AS "time",
     "tabletActivityApp"."activity", 
    lead("timestamp"::timestamp) over (order by "tabletActivityApp".id asc) as next_row
    from "tablet","tabletActivityApp","campaignParticipantKit","kit","participant", "participants_vgp"
    where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "participants_vgp"."kit_id" = "kit"."id"
    and "participants_vgp"."participant_id" = "participant"."id"
    and "participants_vgp"."participant_virtual_id" = '''+get_str_of_id(participant_virtual_id)+'''
    and "participants_vgp"."kit_id" = '''+str(kit_id)+'''
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "tabletActivityApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletActivityApp"."tablet_id"="tablet"."id"
    order by 2''',engine_vgp)
    
    end_dates = pd.read_csv('C:/Polluscope/Plots_participants/end_dates.csv',infer_datetime_format=True, parse_dates=[2])
    

    lendf = len(df_activities) - 1
    
    if (len(df_activities)>0) & (len(end_dates[(end_dates.participant_virtual_id==participant_virtual_id) & (end_dates.kit_id==kit_id)])>0):

        df_activities.loc[lendf,'next_row'] = end_dates[(end_dates.participant_virtual_id==participant_virtual_id) & (end_dates.kit_id==kit_id)].end_date.iloc[0]
        
    elif (len(df_activities)>0) & (len(end_dates[(end_dates.participant_virtual_id==participant_virtual_id) & (end_dates.kit_id==kit_id)])==0):
        
        df_activities.loc[lendf,'next_row']  = df_activities.loc[lendf,'time'] + dt.timedelta(minutes=120)

    
    outdoor = ['Rue', 'Parc','Montagne','Vélo']
    indoor = ['Restaurant','Magasin', 'Cinéma', 'Gare','Voiture_arrêt','Train_arrêt','Inconnu','Domicile_tiers', 'Bureau-tiers', 'Bureau_Tiers']
    transport = ['Voiture', 'Métro', 'Bus', 'Moto', 'Tramway', 'Train']
    
    df_activities['activity_'] = np.where(df_activities.activity.isin(outdoor), 'Extérieur', \
                      np.where(df_activities.activity.isin(transport), 'Transport', \
                      np.where(df_activities.activity.isin(indoor), 'Intérieur', df_activities.activity)))
    
#     df_activities['time'] = pd.to_datetime(df_activities['time'], utc=True)
#     df_activities['next_row'] = pd.to_datetime(df_activities['next_row'], utc=True)
#     df_activities['time'] = df_activities['time'].dt.tz_localize(None)
#     df_activities['next_row'] = df_activities['next_row'].dt.tz_localize(None)
    
    #df_activities['time'] = np.where(df_activities['time'] > pd.Timestamp('2019-10-31 23:59:59'), df_activities['time'] - dt.timedelta(minutes=60), df_activities['time'])
    
    #df_activities['next_row'] = np.where(df_activities['next_row'] > pd.Timestamp('2019-10-31 23:59:59'), df_activities['next_row'] - dt.timedelta(minutes=60), df_activities['next_row'])
    
    return df_activities
    
####

def get_processed_data(participant_virtual_id='9999971', kit_id=70):
    
    df = pd.read_sql('''select C.participant_virtual_id, C.time, C."Temperature", C."Humidity", C."PM1.0", C."PM2.5", C."PM10",
C."NO2", C."BC", C."Speed", C.activity
    from data_processed_vgp C, participants_vgp V
    where C.participant_virtual_id='''+get_str_of_id(participant_virtual_id)+'''
    and C.participant_virtual_id = V.participant_virtual_id
    and V.kit_id='''+str(kit_id)+'''
    and C.time between V.start_date and V.end_date
    order by 2''', engine_processed)
    
    return df

#####

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6372.8 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r * 1000 #convert to meter

################

def get_gps(participant_virtual_id='9999971',kit_id=70):
    
    df = pd.read_sql('''select C.participant_virtual_id, C.time, C.lon, C.lat
    from clean_gps C, participants_vgp V
    where C.participant_virtual_id='''+get_str_of_id(participant_virtual_id)+'''
    and C.participant_virtual_id = V.participant_virtual_id
    and V.kit_id='''+str(kit_id)+'''
    and C.time between V.start_date and V.end_date
    order by 2''', engine_processed)
    
    liste = []
    segment_array = df.values
    lendf = len(segment_array)-1
    for j in range(lendf):
        data1 = segment_array[j]
        data2 = segment_array[j+1]
        liste.append(haversine(data1[2], data1[3], data2[2], data2[3])/(data2[1] - data1[1]).total_seconds()) #metre/second
    liste.append(np.nan)

    df['speed'] = liste
    
    segments = splitting_hafsa(df)
    
    for i in range(len(segments)-1):
    
        segments[i]['speed_mean_'] = segments[i]['speed'].mean()
        segments[i]['distance'] = haversine(segments[i].lon.iloc[0], segments[i].lat.iloc[0], segments[i].lon.iloc[-1], segments[i].lat.iloc[-1])
    
    data2 = pd.DataFrame()
    for segment in segments:
        data2 = pd.concat([data2, segment])
    
    return data2

###############################

def get_end_date(participant_virtual_id, kit_id):

    end_date = pd.read_sql('''select "end_date"::timestamp as "end_date
                            from "participants_vgp"
                            where "participant_virtual_id" = '''+get_str_of_id(participant_virtual_id)+'''
                            and kit_id='''+str(kit_id), engine_processed)
    return end_date

# In[1]:
def plot_activities(id=9999920):

    data = pd.read_csv('detected_activities/participant_'+str(id)+'.csv')

    data['timestamp'] = pd.to_datetime(data['timestamp'], infer_datetime_format=True)



    data['prediction'] = np.where(data.activity_stop=='Bureau', 1,\
                             np.where(data.activity_stop=='Domicile', 2,\
                                     np.where(data.activity_stop=='outdoor',3,\
                                             np.where(data.activity_stop=='indoor', 4,5))))
    outdoor = ['Rue', 'Parc', 'Montagne']
    indoor = ['Restaurant','Magasin', 'Cinéma', 'Gare','Voiture_arrêt','Train_arrêt','Inconnu','Domicile_tiers', 'Bureau_Tiers', 'Bureau_tiers']
    transport = ['Voiture', 'Métro', 'Bus', 'Moto', 'Tramway', 'Train', 'Vélo']

    data['truth'] = np.where(data.activity=='Bureau', 1,\
                                 np.where(data.activity=='Domicile', 2,\
                                         np.where(data.activity.isin(outdoor),3,\
                                                 np.where(data.activity.isin(indoor), 4,\
                                                         np.where(data.activity.isin(transport), 5, 0)))))


    data['truth_'] = data['truth'] + .5


    ig, ax1 = plt.subplots(figsize=(20,5))

    #ax2 = ax1.twinx()
    positions = (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5)
    labels = ('Bureau','annotation_Bureau', 'Domicile','annotation_Domicile', \
              'outdoor','annotation_outdoor', 'indoor','annotation_indoor',\
             'transport','annotation_transport')
    plt.yticks(positions, labels)
    ax1.plot(data[data.prediction==5].timestamp, data[data.prediction==5].prediction, 'g-',linestyle="", marker="o", label='Transport')
    ax1.plot(data[data.prediction==4].timestamp, data[data.prediction==4].prediction, 'b-', linestyle="",marker="v", label='Indoor')
    ax1.plot(data[data.prediction==3].timestamp, data[data.prediction==3].prediction, 'r-',linestyle="", marker="h", label='Outdoor')
    ax1.plot(data[data.prediction==2].timestamp, data[data.prediction==2].prediction, 'y-', linestyle="",marker="D", label='Domicile')
    ax1.plot(data[data.prediction==1].timestamp, data[data.prediction==1].prediction, 'c-', linestyle="",marker="*", label='Bureau')

    ax1.plot(data[data.truth_==5.5].timestamp, data[data.truth_==5.5].truth_, 'g-',linestyle="", marker="o", label='Transport annotation')
    ax1.plot(data[data.truth_==4.5].timestamp, data[data.truth_==4.5].truth_, 'b-', linestyle="",marker="v", label='Indoor annotation')
    ax1.plot(data[data.truth_==3.5].timestamp, data[data.truth_==3.5].truth_, 'r-',linestyle="", marker="h", label='Outdoor annotation')
    ax1.plot(data[data.truth_==2.5].timestamp, data[data.truth_==2.5].truth_, 'y-', linestyle="",marker="H", label='Domicile annotation')
    ax1.plot(data[data.truth_==1.5].timestamp, data[data.truth_==1.5].truth_, 'c-', linestyle="",marker="*", label='Bureau annotation')

    legend = ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', shadow=True, fontsize='small')

    Path("detected_activities/"+str(id)).mkdir(parents=True, exist_ok=True)
    plt.savefig('detected_activities/'+str(id)+'/participant_'+str(id)+'.png')

    return print('done for participant %s : '%id)

# In[3]:

COLOR = {
     'Intérieur': '#00CF83',
    'Bureau': '#20B2AA',
    'Transport': '#FFC0CB',
    'Domicile': '#8FBC8F',
    'Extérieur': '#D3D3D3',
    'Données indisponibles': '#F5F5F5'
}


# In[4]:

def get_color(k, color_dict=COLOR):
    """
    Return a color (random if "k" is negative)
    """
    return color_dict[k]


# In[5]:

def plot_journal(participant_virtual_id=9999961, kit_id=55, start_datetime=None, end_datetime=None, VGP_Week='VGP_W1'):

    """
    ----Part of code from scikit-mobility----

        Parameters
        ----------

        start_datetime : datetime.datetime, optional
            only stops made after this date will be plotted. If `None` the datetime of the oldest stop will be selected. The default is `None`.

        end_datetime : datetime.datetime, optional
            only stops made before this date will be plotted. If `None` the datetime of the newest stop will be selected. The default is `None`.

        predictions : optinal, default is True.
                        Plots participants annotations along with the algorithm's predictions.
                        If it is False, pollutants concentrations will be plotted on the declared activities only.


        Returns
        -------
        matplotlib.axes
            the `matplotlib.axes` object of the plotted diary.

    """
    

    df = get_gps(participant_virtual_id=participant_virtual_id, kit_id=kit_id)
    df2 = get_activities(participant_virtual_id=participant_virtual_id, kit_id=kit_id)
    
    start_dates = pd.read_csv('C:/Polluscope/Plots_participants/start_dates.csv',infer_datetime_format=True, parse_dates=[2])

    
    start_datetime = df2['time'].min()
    end_datetime = df2['next_row'].max() #+ dt.timedelta(minutes=120)
#     end_datetime_1 = df['Heure'].max()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 6), gridspec_kw={'height_ratios': [3,3]})

    
#     if start_datetime is None:
#         start_datetime = start_dates[(start_dates.participant_virtual_id==participant_virtual_id) & (start_dates.kit_id==kit_id)].start_date.iloc[0]
#     if end_datetime is None:
#         end_datetime = df2['next_row'].max()  #+ dt.timedelta(minutes=120)

        #end_datetime_1 = df2[:-1]['timestamp'].max() #+ dt.timedelta(minutes=10)
            

    current_labels = []

    for idx, row in df2.iterrows():

        t0 = row['time']
        t1 = row['next_row']
        cl = row['activity_']

        color = get_color(cl)
        if start_datetime <= t0 <= end_datetime:
            if cl in current_labels:
                ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
            else:
                current_labels += [cl]
                ax1.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

    plt.xlim(start_datetime, end_datetime)
    
    ax1.plot(df.set_index('time').distance, 'k-')

#         if pollutant[0] == 'BC':
#             unite = '(ng/m3)'
#         else:
#             unite = '(µg/m3)'

    ax1.set_ylabel('distance')

#     ax1.set_ylabel('Déclaratif')
#     ax1.set_yticklabels([])

    ###########################
    current_labels = []

    for idx, row in df2.iterrows():

        t0 = row['time']
        t1 = row['next_row']
        cl = row['activity_']

        color = get_color(cl)
        if start_datetime <= t0 <= end_datetime:
            if cl in current_labels:
                ax2.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)
            else:
                current_labels += [cl]
                ax2.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color, label=cl)

    plt.xlim(start_datetime, end_datetime)
    
    ax2.plot(df.set_index('time').speed_mean_, 'k-')

    ax2.set_ylabel('speed_mean_')
    #ax2.set_yticklabels([])


    ###########################

    handles, labels_str = ax2.get_legend_handles_labels()
    labels = list(labels_str)
    # sort them by labels
    import operator
    hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
    handles2, labels2 = zip(*hl)

    ax2.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0)
    user = df.participant_virtual_id.iloc[0]

    ax1.set_title('Participant %s' % participant_virtual_id)
    
    #plt.show()

    #plt.savefig('plot_activities_actualisé/'+str(participant_virtual_id)+'-'+str(VGP_Week)+'-CONTEXT.png', bbox_inches = 'tight', pad_inches=.25)

    return fig

def get_pollutants_complete_data(participant_virtual_id, kit_id):

    df_NO2=pd.read_sql('''select res.*
from(select r1.user_id,timestamp, measurement_value*1.912 as "NO2"
from(
select user_id, time_id, measurement_type_id, measurement_value 
from data_mmd
where user_id='''+get_str_of_id(participant_virtual_id)+'''
and measurement_type_id='3'
) as r1
left join
(select row_number() OVER () as "id", "timestamp"
from(
SELECT generate_series(date '2019-10-15 02:00:00', date '2020-01-01 01:00:00', '1 minute')::timestamp as "timestamp"
order by 1) as rr) as r_time2
on r1.time_id::int = r_time2."id"
order by 2) as res, participants_vgp
where res.user_id = participants_vgp.participant_virtual_id
and participants_vgp.kit_id='''+str(kit_id)+'''
and res.timestamp between participants_vgp.start_date and participants_vgp.end_date
and "NO2" is not null''',engine_vgp)
    
    df_BC=pd.read_sql('''select res.*
from(select r1.user_id,timestamp, measurement_value as "BC"
from(
select user_id, time_id, measurement_type_id, measurement_value 
from data_mmd
where user_id='''+get_str_of_id(participant_virtual_id)+'''
and measurement_type_id='4'
) as r1
left join
(select row_number() OVER () as "id", "timestamp"
from(
SELECT generate_series(date '2019-10-15 02:00:00', date '2020-01-01 01:00:00', '1 minute')::timestamp as "timestamp"
order by 1) as rr) as r_time2
on r1.time_id::int = r_time2."id"
order by 2) as res, participants_vgp
where res.user_id = participants_vgp.participant_virtual_id
and participants_vgp.kit_id='''+str(kit_id)+'''
and res.timestamp between participants_vgp.start_date and participants_vgp.end_date
and "BC" is not null''',engine_vgp)
    
    df_PM10=pd.read_sql('''select res.*
from(select r1.user_id,timestamp, measurement_value as "PM10"
from(
select user_id, time_id, measurement_type_id, measurement_value 
from data_mmd
where user_id='''+get_str_of_id(participant_virtual_id)+'''
and measurement_type_id='2'
) as r1
left join
(select row_number() OVER () as "id", "timestamp"
from(
SELECT generate_series(date '2019-10-15 02:00:00', date '2020-01-01 01:00:00', '1 minute')::timestamp as "timestamp"
order by 1) as rr) as r_time2
on r1.time_id::int = r_time2."id"
order by 2) as res, participants_vgp
where res.user_id = participants_vgp.participant_virtual_id
and participants_vgp.kit_id='''+str(kit_id)+'''
and res.timestamp between participants_vgp.start_date and participants_vgp.end_date
and "PM10" is not null''',engine_vgp)
    
    return df_NO2, df_BC, df_PM10

def call_data_for_plot(participant_virtual_id, kit_id):

    df_NO2, df_BC, df_PM10 = get_pollutants_complete_data(participant_virtual_id=participant_virtual_id, kit_id=kit_id)

    ### Ramène le timestamp à UTC

#     df_NO2['timestamp'] = np.where(df_NO2['timestamp'] > pd.Timestamp('2019-10-31 23:59:59'), df_NO2['timestamp'] - dt.timedelta(minutes=60), df_NO2['timestamp']) 
    
#     df_BC['timestamp'] = np.where(df_BC['timestamp'] > pd.Timestamp('2019-10-31 23:59:59'), df_BC['timestamp'] - dt.timedelta(minutes=60), df_BC['timestamp'])
    
#     df_PM10['timestamp'] = np.where(df_PM10['timestamp'] > pd.Timestamp('2019-10-31 23:59:59'), df_PM10['timestamp'] - dt.timedelta(minutes=60), df_PM10['timestamp'])
    
#     df_NO2['timestamp'] =  df_NO2['timestamp'] - dt.timedelta(minutes=60)
#     df_BC['timestamp'] =  df_BC['timestamp'] - dt.timedelta(minutes=60)
#     df_PM10['timestamp'] =  df_PM10['timestamp'] - dt.timedelta(minutes=60)

    #data['time'] = data['time'] - dt.timedelta(minutes=60)

    df= pd.read_sql('''select detected_activities.*
from detected_activities, participants_vgp
where detected_activities.participant_virtual_id='''+get_str_of_id(participant_virtual_id)+'''
and detected_activities.participant_virtual_id = participants_vgp.participant_virtual_id
and participants_vgp.kit_id='''+str(kit_id)+'''
and detected_activities.timestamp between participants_vgp.start_date and participants_vgp.end_date
order by 2,3
    ''',engine_processed)

    df.sort_values('timestamp', inplace=True)
    #df['timestamp'] = np.where(df['timestamp'] > pd.Timestamp('2019-10-31 23:59:59'), df['timestamp'] + dt.timedelta(minutes=120), df['timestamp']+dt.timedelta(minutes=60))
    
    #df['timestamp'] = df['timestamp'] - dt.timedelta(minutes=60)

    df['leaving_time'] = df.shift(-1)['timestamp'] #+ dt.timedelta(minutes=10)

    df['activity_stop'] = np.where(df['activity_stop'] == 'transport', 'Transport',\
                                np.where(df['activity_stop'] == 'indoor', 'Intérieur',\
                                        np.where(df['activity_stop'] == 'outdoor', 'Extérieur', \
                                                 np.where(df['activity_stop'] == 'No Data', 'Données indisponibles', df['activity_stop']) )))
    end_dates = pd.read_csv('C:/Polluscope/Plots_participants/end_dates.csv',infer_datetime_format=True, parse_dates=[2])
    
    if (len(df)>0) & (len(end_dates[(end_dates.participant_virtual_id==participant_virtual_id) & (end_dates.kit_id==kit_id)])>0):
        
        lendf2 = len(df) - 1
        df.loc[lendf2,'leaving_time'] = end_dates[(end_dates.participant_virtual_id==participant_virtual_id) & (end_dates.kit_id==kit_id)].end_date.iloc[0]
        
    elif (len(df)>0) & (len(end_dates[(end_dates.participant_virtual_id==participant_virtual_id) & (end_dates.kit_id==kit_id)])==0):
        lendf2 = len(df) - 1
        df.loc[lendf2,'leaving_time'] = df.loc[lendf2,'leaving_time'] + dt.timedelta(minutes=120)
    
    return df, df_NO2, df_BC, df_PM10



#########################
def splitting_hafsa(df,time=300):
    
    
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