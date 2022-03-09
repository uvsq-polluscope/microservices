#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# In[1]:

import pandas as pd
import numpy as np
import math
from hilbertcurve.hilbertcurve import HilbertCurve
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import math
from pyproj import Proj, transform
import datetime as dt
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None

#in general: engine = create_engine('dialect+driver://username:password@host:port/database'
url_processed_data='postgresql://postgres:postgres@192.168.33.123:5432/processed_data'
url_processed_data_jdbc = 'jdbc:postgresql://postgres:postgres@192.168.33.123:5432/processed_data'

url_oringinal_data_jdbc = 'jdbc:postgresql://postgres:postgres@192.168.33.123:5432/polluscopev5-last-version-2020'
url_original_data='postgresql://postgres:postgres@192.168.33.123:5432/polluscopev5-last-version-2020'

engine = create_engine(url_original_data)

engine_processed = create_engine(url_processed_data)

#participants = pd.read_csv('c:/\Polluscope/Cartes_participants/list_participants.csv', infer_datetime_format=True, parse_dates=[3,4])

# In[5]:

import os
# jardrv = "C:/Polluscope/DB_Connection/postgresql-42.2.18.jar"
jardrv="hdfs:///home/ubuntu/.ivy2/jars/postgresql-42.2.12.jre6.jar"

# create the SparkSession while pointing to the driver
# import findspark 
# findspark.init()
# import pyspark 
# from pyspark.sql import SparkSession
# spark = SparkSession.builder\
#                         .master("yarn") \
#                         .appName("connect-to-db") \
#                         .enableHiveSupport() \
#                         .config("spark.driver.extraClassPath", jardrv) \
#                         .config("spark.jars.packages","org.postgresql:postgresql:42.2.12")\
#                         .getOrCreate()
# spark

def get_str_of_id(id):
    return "'"+str(id)+"'"


# In[12]:


def get_gps(participant_virtual_id):
    df_id_participant_kit_vgp=pd.read_sql('''select T1."time", T1."participant_virtual_id" as "participant_virtual_id",
    T1."lat" as "lat", T1."lon" as "lon", T2."activity"
    from
    (select "tabletPositionApp"."timestamp" as "time", "participants_vgp"."participant_virtual_id" as "participant_virtual_id",
    "tabletPositionApp"."lat" as "lat", "tabletPositionApp"."lon" as "lon", "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "tabletPositionApp","campaignParticipantKit","kit","participant","participants_vgp"
    where 
    "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "participants_vgp"."kit_id" = "kit"."id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "participants_vgp"."participant_id" = "participant"."id"
    and "participants_vgp"."participant_virtual_id" =  '''+get_str_of_id(participant_virtual_id)+'''
    and "tabletPositionApp"."timestamp" 
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date") as T1
    left join
    (
    select "tabletActivityApp"."timestamp" AS "time",
     "tabletActivityApp"."activity", 
    lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row, 
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
    where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "tabletActivityApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletActivityApp"."tablet_id"="tablet"."id"
    ) as T2 
    on date_trunc('minute',T1."time") between T2."time" and T2.next_row - interval '1 sec'
   and T2."kit_id"=T1."kit_id"
    and T2."participant_id"=T1."participant_id"
    order by 1
    ''',engine)
    
    return df_id_participant_kit_vgp

# In[3]:

def add_hilbert_index_modified(tablet_position_df,Long_min, Long_max, Lat_min, Lat_max,nb_c,nb_r,dimensions,iterations):
    if len(tablet_position_df)>0:
        h = Lat_max - Lat_min
        w = Long_max - Long_min
        cw = w/nb_c
        ch= h/nb_r

        #call the class HilbertCurve
        hilbert_curve = HilbertCurve(iterations, dimensions)

        #Add col_num and row_num to the pandas dataframe
        tablet_position_df['col_num'] = tablet_position_df.apply (lambda row: math.floor((row['lon'] - Long_min)/ cw), axis=1)
        tablet_position_df['row_num'] = tablet_position_df.apply (lambda row: math.floor((row['lat'] - Lat_min)/ ch), axis=1)
        tablet_position_df['lon_centre_cell'] = tablet_position_df.apply (lambda row: Long_min+row['col_num']*cw+cw/2, axis=1)
        tablet_position_df['lat_centre_cell'] = tablet_position_df.apply (lambda row: Lat_min+row['row_num']*ch+ch/2, axis=1)
        ## Keep only data inside the grid
        tablet_position_df = tablet_position_df[(tablet_position_df.lon_centre_cell<=Long_max) & (tablet_position_df.lon_centre_cell>=Long_min) & (tablet_position_df.lat_centre_cell<=Lat_max) & (tablet_position_df.lat_centre_cell>=Lat_min)]
        #Add hilbert column to pandas datafram with the hilbert values calculated from the col_num and row_num columns, if the gps is out of ile-de-france put hilbert=-1
        tablet_position_df['hilbert'] = tablet_position_df.apply(lambda row : hilbert_curve.distance_from_coordinates([row['col_num'], row['row_num']])  if row['col_num'] > 0 and row['row_num'] > 0  else -1, axis = 1) 
    
    return tablet_position_df


# In[3]:


#Long_min, Long_max, Lat_min, Lat_max are the bord of the desired grid 
#nb_c,nb_r are the number of verctical and horizontal split of the grid
#dimensions is the dimensions of the grid, dor gps data it should be 2
#iterations: the number of iterations used in constructing the Hilbert curve (must be > 0)
#constraint(should be): nb_c > 2^iterations -1 and nb_r > 2^iterations -1
#h,w means the high and the width of the grid
#cw,ch means the high and the width of each cell
def add_hilbert_index(tablet_position_df,Long_min, Long_max, Lat_min, Lat_max,nb_c,nb_r,dimensions,iterations):
    h = Lat_max - Lat_min
    w = Long_max - Long_min
    cw = w/nb_c
    ch= h/nb_r
    
    #call the class HilbertCurve
    hilbert_curve = HilbertCurve(iterations, dimensions)
    
    #Add col_num and row_num to the pandas dataframe
    tablet_position_df['col_num'] = tablet_position_df.apply (lambda row: math.floor((row['lon'] - Long_min)/ cw), axis=1)
    tablet_position_df['row_num'] = tablet_position_df.apply (lambda row: math.floor((row['lat'] - Lat_min)/ ch), axis=1)
    
    #Add hilbert column to pandas datafram with the hilbert values calculated from the col_num and row_num columns, if the gps is out of ile-de-france put hilbert=-1
    tablet_position_df['hilbert'] = tablet_position_df.apply(lambda row : hilbert_curve.distance_from_coordinates([row['col_num'], row['row_num']])  if row['col_num'] > 0 and row['row_num'] > 0  else -1, axis = 1) 
    
    return tablet_position_df


# In[4]:


#We need this function to transform the lambert to gps because the hilbert funtion work with the gps
def lambert_to_gps(Long_min, Long_max, Lat_min, Lat_max):
    inProj = Proj(init='epsg:27572')
    outProj = Proj(init='epsg:4326')
    Long_min, Lat_min = transform(inProj,outProj,Long_min,Lat_min)
    Long_max, Lat_max = transform(inProj,outProj,Long_max,Lat_max)
     
    return Long_min, Long_max, Lat_min, Lat_max

# In[5]:

'''
Système de coordonnées : Lambert II / EPSG 27572
résolutions 50m :
Ile de france: Xmin = 516000,  Ymin  = 2319000,  Xmax : 695950, Ymax = 2498950, taille = 3600*3600 (taille=nb_c*nb_r)

How to check the resolution?
cell_width_IDF= (Xmax - Xmin)/3600= 50; cell_hight_IDF= (Ymax - Ymin)/3600= 50

For example if we want a resolution 25m we need to modify the taille to 7198*7198
cell_width_IDF= (Xmax - Xmin)/7198= 25; cell_hight_IDF= (Ymax - Ymin)/7198= 25
'''
Lambert_Long_min = 516000
Lambert_Long_max = 695950
Lambert_Lat_min  = 2319000
Lambert_Lat_max = 2498950
Long_min,Long_max,Lat_min,Lat_max= lambert_to_gps(Lambert_Long_min, Lambert_Long_max, Lambert_Lat_min, Lambert_Lat_max)


# In[6]:


nb_c,nb_r=3600,3600
dimensions,iterations=2,12

# In[7]:
def add_hilbert(participant_virtual_id):
    tablet_position_df=add_hilbert_index(get_gps(participant_virtual_id=participant_virtual_id),Long_min, Long_max, Lat_min,
                                     Lat_max,nb_c,nb_r,dimensions,iterations)#apply the hilbert function on this pandas dataframe
    return tablet_position_df

# In[8]:
def add_hilbert_modified(participant_virtual_id):
    tablet_position_df=add_hilbert_index_modified(get_gps(participant_virtual_id=participant_virtual_id),Long_min, Long_max, Lat_min,
                                     Lat_max,nb_c,nb_r,dimensions,iterations)#apply the hilbert function on this pandas dataframe
    return tablet_position_df

# In[9]:
# def get_all_participantIDs_and_kitIDs(url=url_original_data_jdbc,campaign_id=1,properties={'user': 'postgres', 'password': 'postgres'}):
#     '''This function will return the corresponding kit id and participant id as a dataframe. so you can lookup these ids and 
#     use them in the function of getting data fro postgres'''
#     table='''
#     (select distinct(res.*)
#     from(
#     select "participant"."participant_virtual_id","participant"."id" as "participant_id", "kit"."id" as "kit_id","campaignParticipantKit"."start_date", "campaignParticipantKit"."end_date" from "campaignParticipantKit","participant","kit"
#     where "campaignParticipantKit"."participant_id"="participant"."id"
#     and "campaignParticipantKit"."kit_id"="kit"."id" and campaign_id='''+str(campaign_id)+'''
#     )as res)as test'''
#     properties = properties
#     df = spark.read.jdbc(url=url, table=table, properties=properties)
#     df2 = df.toPandas()
#     return df2

# In[10]:

def get_pollutants(participant_virtual_id, kit_id):

    df_pollutants_vgp=pd.read_sql('''select data_processed_vgp."participant_virtual_id", data_processed_vgp."time", data_processed_vgp."NO2", data_processed_vgp."PM10", data_processed_vgp."BC"
        from data_processed_vgp, participants_vgp
        where data_processed_vgp.participant_virtual_id='''+get_str_of_id(participant_virtual_id)+'''
        and data_processed_vgp.participant_virtual_id = participants_vgp.participant_virtual_id
        and participants_vgp.kit_id='''+str(kit_id)+'''
        and data_processed_vgp.time between participants_vgp.start_date and participants_vgp.end_date
        order by 2''',engine_processed)
    
    return df_pollutants_vgp

def get_pollutants_38(participant_virtual_id, kit_id):

    df_pollutants_vgp=pd.read_sql('''select data_processed_vgp."participant_virtual_id", data_processed_vgp."time", data_processed_vgp."NO2", data_processed_vgp."PM10", data_processed_vgp."BC"
        from data_processed_vgp, participants_vgp
        where data_processed_vgp.participant_virtual_id='''+get_str_of_id(participant_virtual_id)+'''
        and data_processed_vgp.participant_virtual_id = participants_vgp.participant_virtual_id
        and participants_vgp.kit_id='''+str(kit_id)+'''
        and data_processed_vgp.time between participants_vgp.start_date and participants_vgp.end_date
        and data_processed_vgp.time not between '2019-11-04 10:14:00' and '2019-11-04 15:39:00' 
        order by 2''',engine_processed)
    
    return df_pollutants_vgp


# In[11]:

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
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "tabletActivityApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletActivityApp"."tablet_id"="tablet"."id"
    order by 2''',engine)
    
    end_date = pd.read_sql('''select "end_date"
                            from "participants_vgp"
                            where "participant_virtual_id" = '''+get_str_of_id(participant_virtual_id)+'''
                            and kit_id='''+str(kit_id)+''' ''', engine)
    

    lendf = len(df_activities) - 1
    
    if len(df_activities) >0:
        
        df_activities.loc[lendf,'next_row'] = end_date.loc[0,'end_date']
    
    outdoor = ['Rue', 'Parc','Montagne','Vélo']
    indoor = ['Restaurant','Magasin', 'Cinéma', 'Gare','Voiture_arrêt','Train_arrêt','Inconnu','Domicile_tiers', 'Bureau-tiers', 'Bureau_Tiers']
    transport = ['Voiture', 'Métro', 'Bus', 'Moto', 'Tramway', 'Train']
    
    df_activities['activity_'] = np.where(df_activities.activity.isin(outdoor), 'Extérieur', \
                      np.where(df_activities.activity.isin(transport), 'Transport', \
                      np.where(df_activities.activity.isin(indoor), 'Intérieur', df_activities.activity)))
    
    df_activities['time'] = pd.to_datetime(df_activities['time'], utc=True)
    df_activities['next_row'] = pd.to_datetime(df_activities['next_row'], utc=True)
    df_activities['time'] = df_activities['time'].dt.tz_localize(None)
    df_activities['next_row'] = df_activities['next_row'].dt.tz_localize(None)
    
    df_activities['time'] = np.where(df_activities['time'] > pd.Timestamp('2019-10-31 23:59:59'), df_activities['time'] - dt.timedelta(minutes=60), df_activities['time'])
    
    df_activities['next_row'] = np.where(df_activities['next_row'] > pd.Timestamp('2019-10-31 23:59:59'), df_activities['next_row'] - dt.timedelta(minutes=60), df_activities['next_row'])
    
    return df_activities
    
####

def get_end_date(participant_virtual_id, kit_id):

    end_date = pd.read_sql('''select "end_date"::timestamp as "end_date
                            from "participants_vgp"
                            where "participant_virtual_id" = '''+get_str_of_id(participant_virtual_id)+'''
                            and kit_id='''+str(kit_id), engine)
    return end_date