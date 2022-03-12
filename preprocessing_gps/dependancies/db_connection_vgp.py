#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import os
jardrv = "C:/Polluscope/DB_Connection/postgresql-42.2.18.jar"

# create the SparkSession while pointing to the driver
import findspark 
findspark.init()
import pyspark 
from pyspark.sql import SparkSession
spark = SparkSession.builder                        .master("local")                         .appName("connect-to-db")                         .enableHiveSupport()                         .config("spark.driver.extraClassPath", jardrv)                         .getOrCreate()
spark


# In[3]:


def get_postgres_data(url='jdbc:postgresql://localhost:5432/VGP',kit_id=57,participant_id=31):    
    '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
    ....) and will return them as a dataframe'''
    
    #table= query
    
    table = '''
    (select distinct(res.*)
    from (
    select r1."participant_virtual_id", r1."time", r1."PM2.5", r2."PM10", r3."PM1.0", r4."Temperature", r5."Humidity", r6."NO2", r7."BC", r11."vitesse(m/s)", r8."activity"
    , r9."event"
    from (
    select  "participant"."participant_virtual_id",
        "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM2.5",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 7
    ) as r1
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM10",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 8
    ) as r2 on date_trunc('minute',r1."time")=date_trunc('minute',r2."time")
    and "r2"."kit_id"=r1."kit_id" and "r2"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM1.0",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 9
    ) as r3 on date_trunc('minute',r1."time")=date_trunc('minute',r3."time")
    and "r3"."kit_id"=r1."kit_id" and "r3"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Temperature",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 4
    ) as r4 on date_trunc('minute',r1."time")=date_trunc('minute',r4."time")
    and "r4"."kit_id"=r1."kit_id" and "r4"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Humidity",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 5
    ) as r5 on date_trunc('minute',r1."time")=date_trunc('minute',r5."time")
    and "r5"."kit_id"=r1."kit_id" and "r5"."participant_id"=r1."participant_id" 
    left join
    (select  "cairsensMeasure"."timestamp" AS "time",
     "cairsensMeasure"."level" AS "NO2",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
    where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "cairsensMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "cairsensMeasure"."cairsens_id"="cairsens"."id"
    )as r6 on date_trunc('minute',r1."time")=date_trunc('minute',r6."time")
    and "r6"."kit_id"=r1."kit_id" and "r6"."participant_id"=r1."participant_id" 
    left join
    (
    select  "ae51Measure"."timestamp" AS "time",
      "ae51Measure"."bc" AS "BC",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
    where
    "ae51Measure"."ae51_id"="kit"."ae51_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "ae51Measure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    )as r7 on date_trunc('minute',r1."time")=date_trunc('minute',r7."time")
    and "r7"."kit_id"=r1."kit_id" and "r7"."participant_id"=r1."participant_id" 

    left join
    (
    select t1."time",
    st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "vitesse(m/s)"
    from(
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t1, (
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t2
    where t2."time"=t1."time"+ interval '1 minutes' 
    ) as r11 on date_trunc('minute',r1."time")=date_trunc('minute',r11."time")

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
    ) as r8 on date_trunc('minute',r1."time") between r8."time" and r8.next_row
    and "r8"."kit_id"=r1."kit_id" and "r8"."participant_id"=r1."participant_id" 
    left join
    (
    select "tabletEventApp"."timestamp" AS "time",
    "tabletEventApp"."event",
    lead("timestamp") over (order by "tabletEventApp".id asc) as next_row,
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
    where "tabletEventApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "tabletEventApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletEventApp"."tablet_id"="tablet"."id"
    ) as r9 on date_trunc('minute',r1."time") between r9."time" and r9.next_row
    and "r9"."kit_id"=r1."kit_id" and "r9"."participant_id"=r1."participant_id" 
    )as res
    order by res."time") as Test

    '''

    properties = {'user': 'postgres', 'password': 'postgres'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2


# In[4]:


def get_participantID_and_kitID(participant_virtual_id,url='jdbc:postgresql://localhost:5432/VGP'):
    '''This function will return the corresponding kit id and participant id as a dataframe. so you can lookup these ids and 
    use them in the function of getting data fro postgres'''
    table='''
    (select distinct(res.*)
    from(
    select "participant"."id" as "participant_id", "kit"."id" as "kit_id" from "campaignParticipantKit","participant","kit"
    where "campaignParticipantKit"."participant_id"="participant"."id"
    and "campaignParticipantKit"."kit_id"="kit"."id"
    and "participant"."participant_virtual_id"='''+get_str_of_id(participant_virtual_id)+''')as res)as test'''
    properties = {'user': 'postgres', 'password': 'postgres'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2


# In[5]:


def get_str_of_id(id):
    return "'"+str(id)+"'"


# In[6]:


def get_BC_data(url='jdbc:postgresql://localhost:5432/VGP',kit_id=57,participant_id=31):    
    '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
    ....) and will return them as a dataframe'''
    
    #table= query
    
    table = '''
    (select distinct(res.*)
    from (
    select r1."participant_virtual_id", r1."time", r1."BC", r4."Temperature", r5."Humidity", r6."NO2", r7."PM2.5", r2."PM10", r3."PM1.0", r11."vitesse(m/s)", r8."activity"
    , r9."event"
    from (
    
    select  "participant"."participant_virtual_id","ae51Measure"."timestamp" AS "time",
      "ae51Measure"."bc" AS "BC",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
    where
    "ae51Measure"."ae51_id"="kit"."ae51_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "ae51Measure"."bc" is not null
    and "ae51Measure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    ) as r1
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM10",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 8
    ) as r2 on date_trunc('minute',r1."time")=date_trunc('minute',r2."time")
    and "r2"."kit_id"=r1."kit_id" and "r2"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM1.0",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 9
    ) as r3 on date_trunc('minute',r1."time")=date_trunc('minute',r3."time")
    and "r3"."kit_id"=r1."kit_id" and "r3"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Temperature",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 4
    ) as r4 on date_trunc('minute',r1."time")=date_trunc('minute',r4."time")
    and "r4"."kit_id"=r1."kit_id" and "r4"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Humidity",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 5
    ) as r5 on date_trunc('minute',r1."time")=date_trunc('minute',r5."time")
    and "r5"."kit_id"=r1."kit_id" and "r5"."participant_id"=r1."participant_id" 
    left join
    (select  "cairsensMeasure"."timestamp" AS "time",
     "cairsensMeasure"."level" AS "NO2",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
    where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "cairsensMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "cairsensMeasure"."cairsens_id"="cairsens"."id"
    )as r6 on date_trunc('minute',r1."time")=date_trunc('minute',r6."time")
    and "r6"."kit_id"=r1."kit_id" and "r6"."participant_id"=r1."participant_id" 
    left join
    (
    select
        "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM2.5",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 7    
    )as r7 on date_trunc('minute',r1."time")=date_trunc('minute',r7."time")
    and "r7"."kit_id"=r1."kit_id" and "r7"."participant_id"=r1."participant_id" 

    left join
    (
    select t1."time",
    st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "vitesse(m/s)"
    from(
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t1, (
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t2
    where t2."time"=t1."time"+ interval '1 minutes' 
    ) as r11 on date_trunc('minute',r1."time")=date_trunc('minute',r11."time")

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
    ) as r8 on date_trunc('minute',r1."time") between r8."time" and r8.next_row
    and "r8"."kit_id"=r1."kit_id" and "r8"."participant_id"=r1."participant_id" 
    left join
    (
    select "tabletEventApp"."timestamp" AS "time",
    "tabletEventApp"."event",
    lead("timestamp") over (order by "tabletEventApp".id asc) as next_row,
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
    where "tabletEventApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "tabletEventApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletEventApp"."tablet_id"="tablet"."id"
    ) as r9 on date_trunc('minute',r1."time") between r9."time" and r9.next_row
    and "r9"."kit_id"=r1."kit_id" and "r9"."participant_id"=r1."participant_id" 
    )as res
    order by res."time") as Test

    '''

    properties = {'user': 'postgres', 'password': 'postgres'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2


# In[7]:


def get_positive_BC_data(url='jdbc:postgresql://localhost:5432/VGP',kit_id=57,participant_id=31):    
    '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
    ....) and will return them as a dataframe'''
    
    #table= query
    
    table = '''
    (select distinct(res.*)
    from (
    select r1."participant_virtual_id", r1."time", r1."BC", r4."Temperature", r5."Humidity", r6."NO2", r7."PM2.5", r2."PM10", r3."PM1.0", r11."vitesse(m/s)", r8."activity"
    , r9."event"
    from (
    
    select  "participant"."participant_virtual_id","ae51Measure"."timestamp" AS "time",
      "ae51Measure"."bc" AS "BC",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
    where
    "ae51Measure"."ae51_id"="kit"."ae51_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "ae51Measure"."bc" is not null
    and "ae51Measure"."bc" >= 0
    and "ae51Measure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    ) as r1
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM10",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 8
    ) as r2 on date_trunc('minute',r1."time")=date_trunc('minute',r2."time")
    and "r2"."kit_id"=r1."kit_id" and "r2"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM1.0",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 9
    ) as r3 on date_trunc('minute',r1."time")=date_trunc('minute',r3."time")
    and "r3"."kit_id"=r1."kit_id" and "r3"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Temperature",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 4
    ) as r4 on date_trunc('minute',r1."time")=date_trunc('minute',r4."time")
    and "r4"."kit_id"=r1."kit_id" and "r4"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Humidity",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 5
    ) as r5 on date_trunc('minute',r1."time")=date_trunc('minute',r5."time")
    and "r5"."kit_id"=r1."kit_id" and "r5"."participant_id"=r1."participant_id" 
    left join
    (select  "cairsensMeasure"."timestamp" AS "time",
     "cairsensMeasure"."level" AS "NO2",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
    where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "cairsensMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "cairsensMeasure"."cairsens_id"="cairsens"."id"
    )as r6 on date_trunc('minute',r1."time")=date_trunc('minute',r6."time")
    and "r6"."kit_id"=r1."kit_id" and "r6"."participant_id"=r1."participant_id" 
    left join
    (
    select
        "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM2.5",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 7    
    )as r7 on date_trunc('minute',r1."time")=date_trunc('minute',r7."time")
    and "r7"."kit_id"=r1."kit_id" and "r7"."participant_id"=r1."participant_id" 

    left join
    (
    select t1."time",
    st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "vitesse(m/s)"
    from(
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t1, (
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t2
    where t2."time"=t1."time"+ interval '1 minutes' 
    ) as r11 on date_trunc('minute',r1."time")=date_trunc('minute',r11."time")

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
    ) as r8 on date_trunc('minute',r1."time") between r8."time" and r8.next_row
    and "r8"."kit_id"=r1."kit_id" and "r8"."participant_id"=r1."participant_id" 
    left join
    (
    select "tabletEventApp"."timestamp" AS "time",
    "tabletEventApp"."event",
    lead("timestamp") over (order by "tabletEventApp".id asc) as next_row,
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
    where "tabletEventApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "tabletEventApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletEventApp"."tablet_id"="tablet"."id"
    ) as r9 on date_trunc('minute',r1."time") between r9."time" and r9.next_row
    and "r9"."kit_id"=r1."kit_id" and "r9"."participant_id"=r1."participant_id" 
    )as res
    order by res."time") as Test

    '''

    properties = {'user': 'postgres', 'password': 'postgres'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2


# In[8]:


def get_activity_data(url='jdbc:postgresql://localhost:5432/VGP',kit_id=57,participant_id=31):    
    '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
    ....) and will return them as a dataframe'''
    
    #table= query
    
    table = '''
    (select Test.participant_virtual_id,Test.time,Test.activity,Test.act_time as actual_activity_time from (
        (select distinct(res.*),lead("time") over (order by res desc) as res_pre_row
    from (
    select r1."participant_virtual_id",r1.time,r1.next_row as t,r1.kit_id,r1.participant_id
    from (
    select  "participant"."participant_virtual_id",
        "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM2.5",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id",
     lead("timestamp") over (order by "canarinMeasure".id asc) as next_row,
     lead("timestamp") over (order by "canarinMeasure".id desc) as pre_row
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "campaignParticipantKit"."participant_id"="participant"."id"    
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 7
    ) as r1    )
    
     as res order by res."time") as r2
     left join
    (select "tabletActivityApp"."timestamp" AS "act_time",
             "tabletActivityApp"."activity", 
            lead("timestamp") over (order by "tabletActivityApp".id asc) as act_next_row,
                "kit"."id" as "kkit_id", "participant"."id" as "pparticipant_id"
        from "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
        where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
        and "kit"."id"="campaignParticipantKit"."kit_id"
        and "campaignParticipantKit"."participant_id"="participant"."id"
        and "tabletActivityApp"."timestamp"
        between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
        and "tabletActivityApp"."tablet_id"="tablet"."id"
        and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''

        UNION

        select date_trunc('minute',"tabletEventApp"."timestamp") AS "act_time",
        "tabletEventApp"."event",
            lead("timestamp") over (order by "tabletEventApp".id asc) as act_next_row,
                "kit"."id" as "kkit_id", "participant"."id" as "pparticipant_id"
        from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
        where "tabletEventApp"."tablet_id"="kit"."tablet_id"
        and "kit"."id"="campaignParticipantKit"."kit_id"
        and "campaignParticipantKit"."participant_id"="participant"."id"
        and "tabletEventApp"."timestamp"
        between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
        and "tabletEventApp"."tablet_id"="tablet"."id"
        and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
        
    ) as r8 on (date_trunc('minute',r2."time") between r8."act_time" and r8.act_next_row
    and (date_trunc('minute',r2.res_pre_row) not between r8."act_time" and r8.act_next_row )
    and "r8"."kkit_id"=r2."kit_id" and "r8"."pparticipant_id"=r2."participant_id")
    or (date_trunc('minute',r2."time") between r8."act_time" and r8.act_next_row
    and (r2.res_pre_row is null)
    and "r8"."kkit_id"=r2."kit_id" and "r8"."pparticipant_id"=r2."participant_id")
    )
    
    as Test
    where Test.activity is not null) as ae 

    '''

    properties = {'user': 'postgres', 'password': 'postgres'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2
    


# In[9]:


def get_all_participantIDs_and_kitIDs(url='jdbc:postgresql://localhost:5432/VGP',campaign_id=1):
    '''This function will return the corresponding kit id and participant id as a dataframe. so you can lookup these ids and 
    use them in the function of getting data fro postgres'''
    table='''
    (select distinct(res.*)
    from(
    select "participant"."participant_virtual_id","participant"."id" as "participant_id", "kit"."id" as "kit_id","campaignParticipantKit"."start_date", "campaignParticipantKit"."end_date" from "campaignParticipantKit","participant","kit"
    where "campaignParticipantKit"."participant_id"="participant"."id"
    and "campaignParticipantKit"."kit_id"="kit"."id" and campaign_id='''+str(campaign_id)+'''
    )as res)as test'''
    properties = {'user': 'postgres', 'password': 'postgres'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2


# In[10]:


def get_statistical_features(start_date,end_date,url='jdbc:postgresql://localhost:5432/VGP',kit_id=57,participant_id=31,columnName="BC"):
    start_date="'"+start_date+"'"
    end_date="'"+end_date+"'"
    if columnName=="BC":        
        table = '''
        (select activity as activity,to_date(to_char(date_trunc('day',r2."time"), 'MMM dd yyyy '), 'MMM dd yyyy ') "jour", 
        min("'''+columnName+'''"),max("'''+columnName+'''"),avg("'''+columnName+'''") as moyen, percentile_disc(0.5) within group (order by "'''+columnName+'''") as median
        from
        (select  "ae51Measure"."timestamp" AS "time",
          "ae51Measure"."bc" AS "'''+columnName+'''"
        from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
        where "ae51Measure"."ae51_id"="kit"."ae51_id"
        and "kit"."id"="campaignParticipantKit"."kit_id"
        and "campaignParticipantKit"."participant_id"="participant"."id"
        and "kit"."id"= '''+str(kit_id)+'''
        and "participant"."id"='''+str(participant_id)+'''
        and "ae51Measure"."timestamp"
        between '''+start_date+''' and '''+end_date+'''
        and "ae51Measure"."ae51_id"="ae51"."id"
        ) as r1,
        (select "tabletActivityApp"."timestamp" AS time,
        lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row,
        "tabletActivityApp"."activity" as activity
        FROM "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
        where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
        and "kit"."id"="campaignParticipantKit"."kit_id"
        and "campaignParticipantKit"."participant_id"="participant"."id"
        and "kit"."id"='''+str(kit_id)+'''
        and "participant"."id"='''+str(participant_id)+'''
        and "tabletActivityApp"."timestamp"
        between '''+start_date+''' and '''+end_date+'''
        and "tabletActivityApp"."tablet_id"="tablet"."id") as r2
        where  date_trunc('minute',r1."time") between date_trunc('minute',r2."time") and date_trunc('minute',r2.next_row)
        group by "jour", activity
        order by 1,2
        ) as Test

            '''
    else:
        if columnName=="NO2":        
            table = '''
            (select activity as activity,to_date(to_char(date_trunc('day',r2."time"), 'MMM dd yyyy '), 'MMM dd yyyy ') "jour", 
            min("'''+columnName+'''"),max("'''+columnName+'''"),avg("'''+columnName+'''") as moyen, percentile_disc(0.5) within group (order by "'''+columnName+'''") as median
            from
            (select  "cairsensMeasure"."timestamp" AS "time",
             "cairsensMeasure"."level" AS "'''+columnName+'''"
            from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
            where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
            and "kit"."id"= '''+str(kit_id)+'''
            and "participant"."id"='''+str(participant_id)+'''
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "cairsensMeasure"."timestamp"
            between '''+start_date+''' and '''+end_date+'''
            and "cairsensMeasure"."cairsens_id"="cairsens"."id"
            ) as r1,
            (select "tabletActivityApp"."timestamp" AS time,
            lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row,
            "tabletActivityApp"."activity" as activity
            FROM "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
            where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "kit"."id"='''+str(kit_id)+'''
            and "participant"."id"='''+str(participant_id)+'''
            and "tabletActivityApp"."timestamp"
            between '''+start_date+''' and '''+end_date+'''
            and "tabletActivityApp"."tablet_id"="tablet"."id") as r2
            where  date_trunc('minute',r1."time") between date_trunc('minute',r2."time") and date_trunc('minute',r2.next_row)
            group by "jour", activity
            order by 1,2
            ) as Test

                '''
        else:
            if columnName=="vitesse(m/s)" or columnName=="Speed":        
                table = '''
                (select activity as activity,to_date(to_char(date_trunc('day',r2."time"), 'MMM dd yyyy '), 'MMM dd yyyy ') "jour", 
                min("'''+columnName+'''"),max("'''+columnName+'''"),avg("'''+columnName+'''") as moyen, percentile_disc(0.5) within group (order by "'''+columnName+'''") as median
                from
                (

                select t1."time",
                st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "'''+columnName+'''"
                from(
                select DISTINCT ON (res1."time")"time","lat","lon"
                from
                (select date_trunc('minute', "timestamp") AS "time",
                  "tabletPositionApp"."lat",
                  "tabletPositionApp"."lon"
                from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
                where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
                and "kit"."id"="campaignParticipantKit"."kit_id"
                and "campaignParticipantKit"."participant_id"="participant"."id"
                and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
                and "tabletPositionApp"."timestamp"
                between '''+start_date+''' and '''+end_date+'''
                and "tabletPositionApp"."tablet_id"="tablet"."id"
                ) as res1
                ) as t1, (
                select DISTINCT ON (res1."time")"time","lat","lon"
                from
                (select date_trunc('minute', "timestamp") AS "time",
                  "tabletPositionApp"."lat",
                  "tabletPositionApp"."lon"
                from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
                where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
                and "kit"."id"="campaignParticipantKit"."kit_id"
                and "campaignParticipantKit"."participant_id"="participant"."id"
                and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
                and "tabletPositionApp"."timestamp"
                between '''+start_date+''' and '''+end_date+'''
                and "tabletPositionApp"."tablet_id"="tablet"."id"
                ) as res1
                ) as t2
                where t2."time"=t1."time"+ interval '1 minutes'                
                ) as r1,
                (select "tabletActivityApp"."timestamp" AS time,
                lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row,
                "tabletActivityApp"."activity" as activity
                FROM "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
                where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
                and "kit"."id"="campaignParticipantKit"."kit_id"
                and "campaignParticipantKit"."participant_id"="participant"."id"
                and "kit"."id"='''+str(kit_id)+'''
                and "participant"."id"='''+str(participant_id)+'''
                and "tabletActivityApp"."timestamp"
                between '''+start_date+''' and '''+end_date+'''
                and "tabletActivityApp"."tablet_id"="tablet"."id") as r2
                where  date_trunc('minute',r1."time") between date_trunc('minute',r2."time") and date_trunc('minute',r2.next_row)
                group by "jour", activity
                order by 1,2
                ) as Test

                    '''
            else:
                canarin_measure_type={"PM2.5":"7","PM10":"8","PM1.0":"9","Temperature":"4","Humidity":"5"}
                if columnName in canarin_measure_type.keys():
                
                    table = '''
                    (select activity as activity,to_date(to_char(date_trunc('day',r2."time"), 'MMM dd yyyy '), 'MMM dd yyyy ') "jour", 
                    min("'''+columnName+'''"),max("'''+columnName+'''"),avg("'''+columnName+'''") as moyen, percentile_disc(0.5) within group (order by "'''+columnName+'''") as median
                    from
                    (select  "canarinMeasure"."timestamp" AS "time",
                    "canarinMeasure"."value_num" AS "'''+columnName+'''"
                    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
                    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
                    and "kit"."id"= '''+str(kit_id)+'''
                    and "participant"."id"='''+str(participant_id)+'''
                    and "kit"."id"="campaignParticipantKit"."kit_id"
                    and "campaignParticipantKit"."participant_id"="participant"."id"
                    and "canarinMeasure"."timestamp"
                    between '''+start_date+''' and '''+end_date+'''
                    and "canarinMeasure"."canarin_id"="canarin"."id"
                    and "canarinMeasure"."type_id" = '''+canarin_measure_type[columnName]+'''
                    ) as r1,
                    (select "tabletActivityApp"."timestamp" AS time,
                    lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row,
                    "tabletActivityApp"."activity" as activity
                    FROM "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
                    where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
                    and "kit"."id"="campaignParticipantKit"."kit_id"
                    and "campaignParticipantKit"."participant_id"="participant"."id"
                    and "kit"."id"='''+str(kit_id)+'''
                    and "participant"."id"='''+str(participant_id)+'''
                    and "tabletActivityApp"."timestamp"
                    between '''+start_date+''' and '''+end_date+'''
                    and "tabletActivityApp"."tablet_id"="tablet"."id") as r2
                    where  date_trunc('minute',r1."time") between date_trunc('minute',r2."time") and date_trunc('minute',r2.next_row)
                    group by "jour", activity
                    order by 1,2
                    ) as Test

                        '''
                else:
                    print("Column not found")
                    return

    properties = {'user': 'postgres', 'password': 'postgres'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df=df.toPandas()
    return df


# In[11]:


def get_statistical_features_all_participants(start_date,end_date,url='jdbc:postgresql://localhost:5432/VGP',columnName="BC"):
    start_date="'"+start_date+"'"
    end_date="'"+end_date+"'"
    if columnName=="BC":        
        table = '''
        (select activity as activity,to_date(to_char(date_trunc('day',r2."time"), 'MMM dd yyyy '), 'MMM dd yyyy ') "jour", r1.p_v_id as "participant_virtual_id",
        min("'''+columnName+'''"),max("'''+columnName+'''"),avg("'''+columnName+'''") as moyen, percentile_disc(0.5) within group (order by "'''+columnName+'''") as median
        from
        (select  "ae51Measure"."timestamp" AS "time",
          "ae51Measure"."bc" AS "'''+columnName+'''",
          "participant"."participant_virtual_id" as p_v_id
        from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
        where "ae51Measure"."ae51_id"="kit"."ae51_id"
        and "kit"."id"="campaignParticipantKit"."kit_id"
        and "campaignParticipantKit"."participant_id"="participant"."id"
        and "campaignParticipantKit"."start_date"<='''+start_date+''' and "campaignParticipantKit"."end_date">='''+end_date+'''
        and "ae51Measure"."timestamp"
        between '''+start_date+''' and '''+end_date+'''
        and "ae51Measure"."ae51_id"="ae51"."id"
        ) as r1,
        (select "tabletActivityApp"."timestamp" AS time,
        lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row,
        "participant"."participant_virtual_id" as p_v_id,
        "tabletActivityApp"."activity" as activity
        FROM "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
        where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
        and "kit"."id"="campaignParticipantKit"."kit_id"
        and "campaignParticipantKit"."participant_id"="participant"."id"
        and "campaignParticipantKit"."start_date"<='''+start_date+''' and "campaignParticipantKit"."end_date">='''+end_date+'''
        and "tabletActivityApp"."timestamp"
        between '''+start_date+''' and '''+end_date+'''
        and "tabletActivityApp"."tablet_id"="tablet"."id") as r2
        where  date_trunc('minute',r1."time") between date_trunc('minute',r2."time") and date_trunc('minute',r2.next_row) and r1.p_v_id=r2.p_v_id
        group by "jour", activity, r1.p_v_id
        order by 1,2
        ) as Test

            '''
    else:
        if columnName=="NO2":        
            table = '''
            (select activity as activity,to_date(to_char(date_trunc('day',r2."time"), 'MMM dd yyyy '), 'MMM dd yyyy ') "jour", r1.p_v_id as "participant_virtual_id",
            min("'''+columnName+'''"),max("'''+columnName+'''"),avg("'''+columnName+'''") as moyen, percentile_disc(0.5) within group (order by "'''+columnName+'''") as median
            from
            (select  "cairsensMeasure"."timestamp" AS "time",
             "cairsensMeasure"."level" AS "'''+columnName+'''",
             "participant"."participant_virtual_id" as p_v_id
            from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
            where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"            
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "campaignParticipantKit"."start_date"<='''+start_date+''' and "campaignParticipantKit"."end_date">='''+end_date+'''
            and "cairsensMeasure"."timestamp"
            between '''+start_date+''' and '''+end_date+'''
            and "cairsensMeasure"."cairsens_id"="cairsens"."id"
            ) as r1,
            (select "tabletActivityApp"."timestamp" AS time,
            lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row,
            "participant"."participant_virtual_id" as p_v_id,
            "tabletActivityApp"."activity" as activity
            FROM "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
            where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "campaignParticipantKit"."start_date"<='''+start_date+''' and "campaignParticipantKit"."end_date">='''+end_date+'''
            and "tabletActivityApp"."timestamp"
            between '''+start_date+''' and '''+end_date+'''
            and "tabletActivityApp"."tablet_id"="tablet"."id") as r2
            where  date_trunc('minute',r1."time") between date_trunc('minute',r2."time") and date_trunc('minute',r2.next_row) and r1.p_v_id=r2.p_v_id
            group by "jour", activity, r1.p_v_id
            order by 1,2
            ) as Test

                '''
        else:
            if columnName=="vitesse(m/s)" or columnName=="Speed":        
                table = '''
                (select activity as activity,to_date(to_char(date_trunc('day',r2."time"), 'MMM dd yyyy '), 'MMM dd yyyy ') "jour", r1.p_v_id as "participant_virtual_id",
                min("'''+columnName+'''"),max("'''+columnName+'''"),avg("'''+columnName+'''") as moyen, percentile_disc(0.5) within group (order by "'''+columnName+'''") as median
                from
                (

                select t1."time",
                st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "'''+columnName+'''",t1.p_v_id
                from(
                select "time","lat","lon",res1.p_v_id
                from
                (select date_trunc('minute', "timestamp") AS "time",
                  "tabletPositionApp"."lat",
                  "tabletPositionApp"."lon",
                  "participant"."participant_virtual_id" as p_v_id
                from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
                where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
                and "kit"."id"="campaignParticipantKit"."kit_id"
                and "campaignParticipantKit"."participant_id"="participant"."id"                
                and "campaignParticipantKit"."start_date"<='''+start_date+''' and "campaignParticipantKit"."end_date">='''+end_date+'''
                and "tabletPositionApp"."timestamp"
                between '''+start_date+''' and '''+end_date+'''
                and "tabletPositionApp"."tablet_id"="tablet"."id"
                ) as res1
                ) as t1, (
                select "time","lat","lon",res1.p_v_id
                from
                (select date_trunc('minute', "timestamp") AS "time",
                  "tabletPositionApp"."lat",
                  "tabletPositionApp"."lon",
                  "participant"."participant_virtual_id" as p_v_id
                from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
                where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
                and "kit"."id"="campaignParticipantKit"."kit_id"
                and "campaignParticipantKit"."participant_id"="participant"."id"
                and "campaignParticipantKit"."start_date"<='''+start_date+''' and "campaignParticipantKit"."end_date">='''+end_date+'''
                and "tabletPositionApp"."timestamp"
                between '''+start_date+''' and '''+end_date+'''
                and "tabletPositionApp"."tablet_id"="tablet"."id"
                ) as res1
                ) as t2
                where t2."time"=t1."time"+ interval '1 minutes'    
                and t1.p_v_id=t2.p_v_id
                ) as r1,
                (select "tabletActivityApp"."timestamp" AS time,
                lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row,
                "participant"."participant_virtual_id" as p_v_id,
                "tabletActivityApp"."activity" as activity
                FROM "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
                where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
                and "kit"."id"="campaignParticipantKit"."kit_id"
                and "campaignParticipantKit"."participant_id"="participant"."id"
                and "campaignParticipantKit"."start_date"<='''+start_date+''' and "campaignParticipantKit"."end_date">='''+end_date+'''
                and "tabletActivityApp"."timestamp"
                between '''+start_date+''' and '''+end_date+'''
                and "tabletActivityApp"."tablet_id"="tablet"."id") as r2
                where  date_trunc('minute',r1."time") between date_trunc('minute',r2."time") and date_trunc('minute',r2.next_row) and r1.p_v_id=r2.p_v_id
                group by "jour", activity, r1.p_v_id
                order by 1,2
                ) as Test

                    '''
            else:
                canarin_measure_type={"PM2.5":"7","PM10":"8","PM1.0":"9","Temperature":"4","Humidity":"5"}
                if columnName in canarin_measure_type.keys():
                
                    table = '''
                    (select activity as activity,to_date(to_char(date_trunc('day',r2."time"), 'MMM dd yyyy '), 'MMM dd yyyy ') "jour", r1.p_v_id as "participant_virtual_id",
                    min("'''+columnName+'''"),max("'''+columnName+'''"),avg("'''+columnName+'''") as moyen, percentile_disc(0.5) within group (order by "'''+columnName+'''") as median
                    from
                    (select  "canarinMeasure"."timestamp" AS "time",
                    "canarinMeasure"."value_num" AS "'''+columnName+'''",
                    "participant"."participant_virtual_id" as p_v_id
                    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
                    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
                    and "kit"."id"="campaignParticipantKit"."kit_id"
                    and "campaignParticipantKit"."participant_id"="participant"."id"
                    and "campaignParticipantKit"."start_date"<='''+start_date+''' and "campaignParticipantKit"."end_date">='''+end_date+'''
                    and "canarinMeasure"."timestamp"
                    between '''+start_date+''' and '''+end_date+'''
                    and "canarinMeasure"."canarin_id"="canarin"."id"
                    and "canarinMeasure"."type_id" = '''+canarin_measure_type[columnName]+'''
                    ) as r1,
                    (select "tabletActivityApp"."timestamp" AS time,
                    lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row,
                    "participant"."participant_virtual_id" as p_v_id,
                    "tabletActivityApp"."activity" as activity
                    FROM "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
                    where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
                    and "kit"."id"="campaignParticipantKit"."kit_id"
                    and "campaignParticipantKit"."participant_id"="participant"."id"
                    and "campaignParticipantKit"."start_date"<='''+start_date+''' and "campaignParticipantKit"."end_date">='''+end_date+'''
                    and "tabletActivityApp"."timestamp"
                    between '''+start_date+''' and '''+end_date+'''
                    and "tabletActivityApp"."tablet_id"="tablet"."id") as r2
                    where  date_trunc('minute',r1."time") between date_trunc('minute',r2."time") and date_trunc('minute',r2.next_row) and r1.p_v_id=r2.p_v_id
                    group by "jour", activity, r1.p_v_id
                    order by 1,2
                    ) as Test

                        '''
                else:
                    print("Column not found")
                    return

    properties = {'user': 'postgres', 'password': 'postgres'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df=df.toPandas()
    return df


# In[1]:


def get_NO2_data(url='jdbc:postgresql://localhost:5432/VGP',kit_id=57,participant_id=31):    
    '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
    ....) and will return them as a dataframe'''
    
    #table= query
    
    table = '''
    (select distinct(res.*)
    from (
    select r1."participant_virtual_id", r1."time", r1."NO2", r4."Temperature", r5."Humidity", r6."BC", r7."PM2.5", r2."PM10", r3."PM1.0", r11."vitesse(m/s)", r8."activity"
    , r9."event"
    from (
    select  "participant"."participant_virtual_id","cairsensMeasure"."timestamp" AS "time",
     "cairsensMeasure"."level" AS "NO2",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
    where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "cairsensMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "cairsensMeasure"."cairsens_id"="cairsens"."id"
    ) as r1
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM10",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 8
    ) as r2 on date_trunc('minute',r1."time")=date_trunc('minute',r2."time")
    and "r2"."kit_id"=r1."kit_id" and "r2"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM1.0",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 9
    ) as r3 on date_trunc('minute',r1."time")=date_trunc('minute',r3."time")
    and "r3"."kit_id"=r1."kit_id" and "r3"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Temperature",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 4
    ) as r4 on date_trunc('minute',r1."time")=date_trunc('minute',r4."time")
    and "r4"."kit_id"=r1."kit_id" and "r4"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Humidity",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 5
    ) as r5 on date_trunc('minute',r1."time")=date_trunc('minute',r5."time")
    and "r5"."kit_id"=r1."kit_id" and "r5"."participant_id"=r1."participant_id" 
    left join
    (
    select  "participant"."participant_virtual_id","ae51Measure"."timestamp" AS "time",
      "ae51Measure"."bc" AS "BC",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
    where
    "ae51Measure"."ae51_id"="kit"."ae51_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "ae51Measure"."bc" is not null
    and "ae51Measure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    )as r6 on date_trunc('minute',r1."time")=date_trunc('minute',r6."time")
    and "r6"."kit_id"=r1."kit_id" and "r6"."participant_id"=r1."participant_id" 
    left join
    (
    select
        "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM2.5",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 7    
    )as r7 on date_trunc('minute',r1."time")=date_trunc('minute',r7."time")
    and "r7"."kit_id"=r1."kit_id" and "r7"."participant_id"=r1."participant_id" 

    left join
    (
    select t1."time",
    st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "vitesse(m/s)"
    from(
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t1, (
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t2
    where t2."time"=t1."time"+ interval '1 minutes' 
    ) as r11 on date_trunc('minute',r1."time")=date_trunc('minute',r11."time")

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
    ) as r8 on date_trunc('minute',r1."time") between r8."time" and r8.next_row
    and "r8"."kit_id"=r1."kit_id" and "r8"."participant_id"=r1."participant_id" 
    left join
    (
    select "tabletEventApp"."timestamp" AS "time",
    "tabletEventApp"."event",
    lead("timestamp") over (order by "tabletEventApp".id asc) as next_row,
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
    where "tabletEventApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "tabletEventApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletEventApp"."tablet_id"="tablet"."id"
    ) as r9 on date_trunc('minute',r1."time") between r9."time" and r9.next_row
    and "r9"."kit_id"=r1."kit_id" and "r9"."participant_id"=r1."participant_id" 
    )as res
    order by res."time") as Test

    '''

    properties = {'user': 'postgres', 'password': 'postgres'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2


# In[2]:


def get_Temperature_data(url='jdbc:postgresql://localhost:5432/VGP',kit_id=57,participant_id=31):    
    '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
    ....) and will return them as a dataframe'''
    
    #table= query
    
    table = '''
    (select distinct(res.*)
    from (
    select r1."participant_virtual_id", r1."kit_id", r1."participant_id", r1."time", r1."Temperature",  r5."Humidity",r4."NO2", r6."BC", r7."PM2.5", r2."PM10", r3."PM1.0", r11."vitesse(m/s)", r8."activity"
    , r9."event"
    from (
    
    
    select  "participant"."participant_virtual_id","canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Temperature",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 4
    ) as r1
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM10",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 8
    ) as r2 on date_trunc('minute',r1."time")=date_trunc('minute',r2."time")
    and "r2"."kit_id"=r1."kit_id" and "r2"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM1.0",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 9
    ) as r3 on date_trunc('minute',r1."time")=date_trunc('minute',r3."time")
    and "r3"."kit_id"=r1."kit_id" and "r3"."participant_id"=r1."participant_id" 
    Left join (    
    select "cairsensMeasure"."timestamp" AS "time",
     "cairsensMeasure"."level" AS "NO2",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
    where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "cairsensMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "cairsensMeasure"."cairsens_id"="cairsens"."id"
    ) as r4 on date_trunc('minute',r1."time")=date_trunc('minute',r4."time")
    and "r4"."kit_id"=r1."kit_id" and "r4"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Humidity",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 5
    ) as r5 on date_trunc('minute',r1."time")=date_trunc('minute',r5."time")
    and "r5"."kit_id"=r1."kit_id" and "r5"."participant_id"=r1."participant_id" 
    left join
    (
    select  "participant"."participant_virtual_id","ae51Measure"."timestamp" AS "time",
      "ae51Measure"."bc" AS "BC",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
    where
    "ae51Measure"."ae51_id"="kit"."ae51_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "ae51Measure"."bc" is not null
    and "ae51Measure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    )as r6 on date_trunc('minute',r1."time")=date_trunc('minute',r6."time")
    and "r6"."kit_id"=r1."kit_id" and "r6"."participant_id"=r1."participant_id" 
    left join
    (
    select
        "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM2.5",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 7    
    )as r7 on date_trunc('minute',r1."time")=date_trunc('minute',r7."time")
    and "r7"."kit_id"=r1."kit_id" and "r7"."participant_id"=r1."participant_id" 

    left join
    (
    select t1."time",
    st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "vitesse(m/s)"
    from(
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t1, (
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t2
    where t2."time"=t1."time"+ interval '1 minutes' 
    ) as r11 on date_trunc('minute',r1."time")=date_trunc('minute',r11."time")

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
    ) as r8 on date_trunc('minute',r1."time") between r8."time" and r8.next_row
    and "r8"."kit_id"=r1."kit_id" and "r8"."participant_id"=r1."participant_id" 
    left join
    (
    select "tabletEventApp"."timestamp" AS "time",
    "tabletEventApp"."event",
    lead("timestamp") over (order by "tabletEventApp".id asc) as next_row,
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
    where "tabletEventApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "tabletEventApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletEventApp"."tablet_id"="tablet"."id"
    ) as r9 on date_trunc('minute',r1."time") between r9."time" and r9.next_row
    and "r9"."kit_id"=r1."kit_id" and "r9"."participant_id"=r1."participant_id" 
    )as res
    order by res."time") as Test

    '''

    properties = {'user': 'postgres', 'password': 'postgres'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2


# In[3]:


def get_Humidity_data(url='jdbc:postgresql://localhost:5432/VGP',kit_id=57,participant_id=31):    
    '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
    ....) and will return them as a dataframe'''
    
    #table= query
    
    table = '''
    (select distinct(res.*)
    from (
    select r1."participant_virtual_id", r1."time", r1."Humidity",  r5."Temperature",r4."NO2", r6."BC", r7."PM2.5", r2."PM10", r3."PM1.0", r11."vitesse(m/s)", r8."activity"
    , r9."event"
    from (    
    select  "participant"."participant_virtual_id","canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Humidity",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 5
    ) as r1
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM10",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 8
    ) as r2 on date_trunc('minute',r1."time")=date_trunc('minute',r2."time")
    and "r2"."kit_id"=r1."kit_id" and "r2"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM1.0",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 9
    ) as r3 on date_trunc('minute',r1."time")=date_trunc('minute',r3."time")
    and "r3"."kit_id"=r1."kit_id" and "r3"."participant_id"=r1."participant_id" 
    Left join (    
    select "cairsensMeasure"."timestamp" AS "time",
     "cairsensMeasure"."level" AS "NO2",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
    where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "cairsensMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "cairsensMeasure"."cairsens_id"="cairsens"."id"
    ) as r4 on date_trunc('minute',r1."time")=date_trunc('minute',r4."time")
    and "r4"."kit_id"=r1."kit_id" and "r4"."participant_id"=r1."participant_id" 
    Left join (
    select "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Temperature",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 4
    ) as r5 on date_trunc('minute',r1."time")=date_trunc('minute',r5."time")
    and "r5"."kit_id"=r1."kit_id" and "r5"."participant_id"=r1."participant_id" 
    left join
    (
    select  "participant"."participant_virtual_id","ae51Measure"."timestamp" AS "time",
      "ae51Measure"."bc" AS "BC",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
    where
    "ae51Measure"."ae51_id"="kit"."ae51_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "ae51Measure"."bc" is not null
    and "ae51Measure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    )as r6 on date_trunc('minute',r1."time")=date_trunc('minute',r6."time")
    and "r6"."kit_id"=r1."kit_id" and "r6"."participant_id"=r1."participant_id" 
    left join
    (
    select
        "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM2.5",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 7    
    )as r7 on date_trunc('minute',r1."time")=date_trunc('minute',r7."time")
    and "r7"."kit_id"=r1."kit_id" and "r7"."participant_id"=r1."participant_id" 

    left join
    (
    select t1."time",
    st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "vitesse(m/s)"
    from(
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t1, (
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t2
    where t2."time"=t1."time"+ interval '1 minutes' 
    ) as r11 on date_trunc('minute',r1."time")=date_trunc('minute',r11."time")

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
    ) as r8 on date_trunc('minute',r1."time") between r8."time" and r8.next_row
    and "r8"."kit_id"=r1."kit_id" and "r8"."participant_id"=r1."participant_id" 
    left join
    (
    select "tabletEventApp"."timestamp" AS "time",
    "tabletEventApp"."event",
    lead("timestamp") over (order by "tabletEventApp".id asc) as next_row,
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
    where "tabletEventApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "tabletEventApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletEventApp"."tablet_id"="tablet"."id"
    ) as r9 on date_trunc('minute',r1."time") between r9."time" and r9.next_row
    and "r9"."kit_id"=r1."kit_id" and "r9"."participant_id"=r1."participant_id" 
    )as res
    order by res."time") as Test

    '''

    properties = {'user': 'postgres', 'password': 'postgres'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2


# In[4]:


def get_PM10_data(url='jdbc:postgresql://localhost:5432/VGP',kit_id=57,participant_id=31):    
    '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
    ....) and will return them as a dataframe'''
    
    #table= query
    
    table = '''
    (select distinct(res.*)
    from (
    select r1."participant_virtual_id", r1."time", r1."PM10", r7."PM2.5", r3."PM1.0",  r5."Temperature", r2."Humidity",r4."NO2", r6."BC", r11."vitesse(m/s)", r8."activity"
    , r9."event"
    from (
    select  "participant"."participant_virtual_id","canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM10",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 8
    ) as r1
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Humidity",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 5
    ) as r2 on date_trunc('minute',r1."time")=date_trunc('minute',r2."time")
    and "r2"."kit_id"=r1."kit_id" and "r2"."participant_id"=r1."participant_id" 
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM1.0",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 9
    ) as r3 on date_trunc('minute',r1."time")=date_trunc('minute',r3."time")
    and "r3"."kit_id"=r1."kit_id" and "r3"."participant_id"=r1."participant_id" 
    Left join (    
    select "cairsensMeasure"."timestamp" AS "time",
     "cairsensMeasure"."level" AS "NO2",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
    where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "cairsensMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "cairsensMeasure"."cairsens_id"="cairsens"."id"
    ) as r4 on date_trunc('minute',r1."time")=date_trunc('minute',r4."time")
    and "r4"."kit_id"=r1."kit_id" and "r4"."participant_id"=r1."participant_id" 
    Left join (
    select "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Temperature",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 4
    ) as r5 on date_trunc('minute',r1."time")=date_trunc('minute',r5."time")
    and "r5"."kit_id"=r1."kit_id" and "r5"."participant_id"=r1."participant_id" 
    left join
    (
    select  "participant"."participant_virtual_id","ae51Measure"."timestamp" AS "time",
      "ae51Measure"."bc" AS "BC",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
    where
    "ae51Measure"."ae51_id"="kit"."ae51_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "ae51Measure"."bc" is not null
    and "ae51Measure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    )as r6 on date_trunc('minute',r1."time")=date_trunc('minute',r6."time")
    and "r6"."kit_id"=r1."kit_id" and "r6"."participant_id"=r1."participant_id" 
    left join
    (
    select
        "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM2.5",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 7    
    )as r7 on date_trunc('minute',r1."time")=date_trunc('minute',r7."time")
    and "r7"."kit_id"=r1."kit_id" and "r7"."participant_id"=r1."participant_id" 

    left join
    (
    select t1."time",
    st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "vitesse(m/s)"
    from(
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t1, (
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t2
    where t2."time"=t1."time"+ interval '1 minutes' 
    ) as r11 on date_trunc('minute',r1."time")=date_trunc('minute',r11."time")

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
    ) as r8 on date_trunc('minute',r1."time") between r8."time" and r8.next_row
    and "r8"."kit_id"=r1."kit_id" and "r8"."participant_id"=r1."participant_id" 
    left join
    (
    select "tabletEventApp"."timestamp" AS "time",
    "tabletEventApp"."event",
    lead("timestamp") over (order by "tabletEventApp".id asc) as next_row,
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
    where "tabletEventApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "tabletEventApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletEventApp"."tablet_id"="tablet"."id"
    ) as r9 on date_trunc('minute',r1."time") between r9."time" and r9.next_row
    and "r9"."kit_id"=r1."kit_id" and "r9"."participant_id"=r1."participant_id" 
    )as res
    order by res."time") as Test

    '''

    properties = {'user': 'postgres', 'password': 'postgres'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2


# In[5]:


def get_PM1_data(url='jdbc:postgresql://localhost:5432/VGP',kit_id=57,participant_id=31):    
    '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
    ....) and will return them as a dataframe'''
    
    #table= query
    
    table = '''
    (select distinct(res.*)
    from (
    select r1."participant_virtual_id", r1."time", r1."PM1.0", r7."PM2.5",r3."PM10",  r5."Temperature", r2."Humidity",r4."NO2", r6."BC",  r11."vitesse(m/s)", r8."activity"
    , r9."event"
    from (    
    select  "participant"."participant_virtual_id","canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM1.0",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 9    
    ) as r1
    Left join (
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Humidity",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 5
    ) as r2 on date_trunc('minute',r1."time")=date_trunc('minute',r2."time")
    and "r2"."kit_id"=r1."kit_id" and "r2"."participant_id"=r1."participant_id" 
    Left join (    
    select  "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM10",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 8
    ) as r3 on date_trunc('minute',r1."time")=date_trunc('minute',r3."time")
    and "r3"."kit_id"=r1."kit_id" and "r3"."participant_id"=r1."participant_id" 
    Left join (    
    select "cairsensMeasure"."timestamp" AS "time",
     "cairsensMeasure"."level" AS "NO2",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
    where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "cairsensMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "cairsensMeasure"."cairsens_id"="cairsens"."id"
    ) as r4 on date_trunc('minute',r1."time")=date_trunc('minute',r4."time")
    and "r4"."kit_id"=r1."kit_id" and "r4"."participant_id"=r1."participant_id" 
    Left join (
    select "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "Temperature",
    "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 4
    ) as r5 on date_trunc('minute',r1."time")=date_trunc('minute',r5."time")
    and "r5"."kit_id"=r1."kit_id" and "r5"."participant_id"=r1."participant_id" 
    left join
    (
    select  "participant"."participant_virtual_id","ae51Measure"."timestamp" AS "time",
      "ae51Measure"."bc" AS "BC",
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
    where
    "ae51Measure"."ae51_id"="kit"."ae51_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "ae51Measure"."bc" is not null
    and "ae51Measure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    )as r6 on date_trunc('minute',r1."time")=date_trunc('minute',r6."time")
    and "r6"."kit_id"=r1."kit_id" and "r6"."participant_id"=r1."participant_id" 
    left join
    (
    select
        "canarinMeasure"."timestamp" AS "time",
      "canarinMeasure"."value_num" AS "PM2.5",
     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
    where "canarinMeasure"."canarin_id"="kit"."canarin_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"    
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "canarinMeasure"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "canarinMeasure"."canarin_id"="canarin"."id"
    and "canarinMeasure"."type_id" = 7    
    )as r7 on date_trunc('minute',r1."time")=date_trunc('minute',r7."time")
    and "r7"."kit_id"=r1."kit_id" and "r7"."participant_id"=r1."participant_id" 

    left join
    (
    select t1."time",
    st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "vitesse(m/s)"
    from(
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t1, (
    select DISTINCT ON (res1."time")"time","lat","lon"
    from
    (select date_trunc('minute', "timestamp") AS "time",
      "tabletPositionApp"."lat",
      "tabletPositionApp"."lon"
    from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
    where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
    and "tabletPositionApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletPositionApp"."tablet_id"="tablet"."id"
    ) as res1
    ) as t2
    where t2."time"=t1."time"+ interval '1 minutes' 
    ) as r11 on date_trunc('minute',r1."time")=date_trunc('minute',r11."time")

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
    ) as r8 on date_trunc('minute',r1."time") between r8."time" and r8.next_row
    and "r8"."kit_id"=r1."kit_id" and "r8"."participant_id"=r1."participant_id" 
    left join
    (
    select "tabletEventApp"."timestamp" AS "time",
    "tabletEventApp"."event",
    lead("timestamp") over (order by "tabletEventApp".id asc) as next_row,
        "kit"."id" as "kit_id", "participant"."id" as "participant_id"
    from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
    where "tabletEventApp"."tablet_id"="kit"."tablet_id"
    and "kit"."id"="campaignParticipantKit"."kit_id"
    and "campaignParticipantKit"."participant_id"="participant"."id"
    and "tabletEventApp"."timestamp"
    between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
    and "tabletEventApp"."tablet_id"="tablet"."id"
    ) as r9 on date_trunc('minute',r1."time") between r9."time" and r9.next_row
    and "r9"."kit_id"=r1."kit_id" and "r9"."participant_id"=r1."participant_id" 
    )as res
    order by res."time") as Test

    '''

    properties = {'user': 'postgres', 'password': 'postgres'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2


# In[ ]:




