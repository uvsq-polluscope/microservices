# Imports
import pandas as pd
import numpy as np
import os
import findspark 
import pyspark 
from pyspark.sql import SparkSession
import sqlalchemy as db
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import select
from sqlalchemy.sql import text
#from data_operations import *



# #java directory
# jardrv = "/home/jovyan/.ivy2/jars/org.postgresql_postgresql-42.2.12.jar"

# # create the SparkSession while pointing to the driver
# findspark.init()
# spark = SparkSession.builder                        .master("local")                         .appName("connect-to-db")                         .enableHiveSupport()                         .config("spark.driver.extraClassPath", jardrv)                         .getOrCreate()
# # spark




#get the participant id and kit id this function takes the participant virtual id as an input in addition to the url
def get_participantID_and_kitID(participant_virtual_id,url='jdbc:postgresql://193.55.95.225:32263/polluscopev5'):
    '''This function will return the corresponding kit id and participant id as a dataframe. so you can lookup these ids and 
    use them in the function of getting data fro postgres'''
    table='''
    (select distinct(res.*)
    from(
    select "participant"."id" as "participant_id", "kit"."id" as "kit_id" from "campaignParticipantKit","participant","kit"
    where "campaignParticipantKit"."participant_id"="participant"."id"
    and "campaignParticipantKit"."kit_id"="kit"."id"
    and "participant"."participant_virtual_id"='''+get_str_of_id(participant_virtual_id)+''')as res)as test'''
    properties = {'user': 'docker', 'password': 'docker'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2


def get_participant_virtual_ids(url='postgresql://postgres:root@localhost:5432/VGP',table_name='data_processed_vgp'):
    engine = db.create_engine(url) 
    connection = engine.connect()
    metadata = db.MetaData()
    data = db.Table(table_name, metadata, autoload=True, autoload_with=engine)
    results = connection.execute(db.select([data.columns.participant_virtual_id.distinct()]))    
    df = pd.DataFrame(results)
    df.columns = ['participant_virtual_id']
    return df.participant_virtual_id.to_list()

#get all participants id and kits id of all participants this function takes the campaign id as an input in addition to the url
def get_all_participantIDs_and_kitIDs(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',campaign_id=1):
    '''This function will return the corresponding kit id and participant id as a dataframe. so you can lookup these ids and 
    use them in the function of getting data fro postgres'''
    table='''
    (select distinct(res.*)
    from(
    select "participant"."participant_virtual_id","participant"."id" as "participant_id", "kit"."id" as "kit_id","campaignParticipantKit"."start_date", "campaignParticipantKit"."end_date" from "campaignParticipantKit","participant","kit"
    where "campaignParticipantKit"."participant_id"="participant"."id"
    and "campaignParticipantKit"."kit_id"="kit"."id" and campaign_id='''+str(campaign_id)+'''
    )as res)as test'''
    properties = {'user': 'docker', 'password': 'docker'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2




#get stored data in postgres based on left join on PM values (we should specify the url[containing username and password], also kit id and participant id)
def get_postgres_data(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',kit_id=57,participant_id=31):    
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

    properties = {'user': 'docker', 'password': 'docker'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2




def get_str_of_id(id):
    return "'"+str(id)+"'"




#get stored data in postgres based on left join on BC values (we should specify the url[containing username and password], also kit id and participant id)
def get_BC_data(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',kit_id=57,participant_id=31):    
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

    properties = {'user': 'docker', 'password': 'docker'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2




#get stored data in postgres based on left join on NO2 values (we should specify the url[containing username and password], also kit id and participant id)
def get_NO2_data(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',kit_id=57,participant_id=31):    
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

    properties = {'user': 'docker', 'password': 'docker'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2




#get stored data in postgres based on left join on Temperature values (we should specify the url[containing username and password], also kit id and participant id)
def get_Temperature_data(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',kit_id=57,participant_id=31):    
    '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
    ....) and will return them as a dataframe'''
    
    #table= query
    
    table = '''
    (select distinct(res.*)
    from (
    select r1."participant_virtual_id", r1."time", r1."Temperature",  r5."Humidity",r4."NO2", r6."BC", r7."PM2.5", r2."PM10", r3."PM1.0", r11."vitesse(m/s)", r8."activity"
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

    properties = {'user': 'docker', 'password': 'docker'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2




#get stored data in postgres based on left join on Humidity values (we should specify the url[containing username and password], also kit id and participant id)
def get_Humidity_data(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',kit_id=57,participant_id=31):    
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

    properties = {'user': 'docker', 'password': 'docker'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2




#get stored activities of a specific participant depending on his kit an participant id
def get_activity_data(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',kit_id=57,participant_id=31):    
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

    properties = {'user': 'docker', 'password': 'docker'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2




def get_GPS_data(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',kit_id=57,participant_id=31):    
    '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
    ....) and will return them as a dataframe'''
    
    #table= query
    
    table = '''
    (select distinct(res.*)
    from (
    select t1."time",t1."lat",t1."lon"
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
    ) as t1
    )as res
    order by res."time") as Test

    '''

    properties = {'user': 'docker', 'password': 'docker'}
    df = spark.read.jdbc(url=url, table=table, properties=properties)
    df2 = df.toPandas()
    return df2



# # Imports
# import pandas as pd
# import numpy as np
# import os
# import findspark 
# import pyspark 
# from pyspark.sql import SparkSession

# #java directory
# jardrv = "/home/jovyan/.ivy2/jars/org.postgresql_postgresql-42.2.12.jar"

# # create the SparkSession while pointing to the driver
# findspark.init()
# spark = SparkSession.builder                        .master("local")                         .appName("connect-to-db")                         .enableHiveSupport()                         .config("spark.driver.extraClassPath", jardrv)                         .getOrCreate()
# spark

# #get the participant id and kit id this function takes the participant virtual id as an input in addition to the url
# def get_participantID_and_kitID(participant_virtual_id,url='jdbc:postgresql://193.55.95.225:32263/polluscopev5'):
#     '''This function will return the corresponding kit id and participant id as a dataframe. so you can lookup these ids and 
#     use them in the function of getting data fro postgres'''
#     table='''
#     (select distinct(res.*)
#     from(
#     select "participant"."id" as "participant_id", "kit"."id" as "kit_id" from "campaignParticipantKit","participant","kit"
#     where "campaignParticipantKit"."participant_id"="participant"."id"
#     and "campaignParticipantKit"."kit_id"="kit"."id"
#     and "participant"."participant_virtual_id"='''+get_str_of_id(participant_virtual_id)+''')as res)as test'''
#     properties = {'user': 'docker', 'password': 'docker'}
#     df = spark.read.jdbc(url=url, table=table, properties=properties)
#     df2 = df.toPandas()
#     return df2

# #get all participants id and kits id of all participants this function takes the campaign id as an input in addition to the url
# def get_all_participantIDs_and_kitIDs(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',campaign_id=1):
#     '''This function will return the corresponding kit id and participant id as a dataframe. so you can lookup these ids and 
#     use them in the function of getting data fro postgres'''
#     table='''
#     (select distinct(res.*)
#     from(
#     select "participant"."participant_virtual_id","participant"."id" as "participant_id", "kit"."id" as "kit_id","campaignParticipantKit"."start_date", "campaignParticipantKit"."end_date" from "campaignParticipantKit","participant","kit"
#     where "campaignParticipantKit"."participant_id"="participant"."id"
#     and "campaignParticipantKit"."kit_id"="kit"."id" and campaign_id='''+str(campaign_id)+'''
#     )as res)as test'''
#     properties = {'user': 'docker', 'password': 'docker'}
#     df = spark.read.jdbc(url=url, table=table, properties=properties)
#     df2 = df.toPandas()
#     return df2

# #get stored data in postgres based on left join on PM values (we should specify the url[containing username and password], also kit id and participant id)
# def get_postgres_data(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',kit_id=57,participant_id=31):    
#     '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
#     ....) and will return them as a dataframe'''
    
#     #table= query
    
#     table = '''
#     (select distinct(res.*)
#     from (
#     select r1."participant_virtual_id", r1."time", r1."PM2.5", r2."PM10", r3."PM1.0", r4."Temperature", r5."Humidity", r6."NO2", r7."BC", r11."vitesse(m/s)", r8."activity"
#     , r9."event"
#     from (
#     select  "participant"."participant_virtual_id",
#         "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM2.5",
#      "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 7
#     ) as r1
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM10",
#     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 8
#     ) as r2 on date_trunc('minute',r1."time")=date_trunc('minute',r2."time")
#     and "r2"."kit_id"=r1."kit_id" and "r2"."participant_id"=r1."participant_id" 
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM1.0",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 9
#     ) as r3 on date_trunc('minute',r1."time")=date_trunc('minute',r3."time")
#     and "r3"."kit_id"=r1."kit_id" and "r3"."participant_id"=r1."participant_id" 
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "Temperature",
#     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 4
#     ) as r4 on date_trunc('minute',r1."time")=date_trunc('minute',r4."time")
#     and "r4"."kit_id"=r1."kit_id" and "r4"."participant_id"=r1."participant_id" 
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "Humidity",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 5
#     ) as r5 on date_trunc('minute',r1."time")=date_trunc('minute',r5."time")
#     and "r5"."kit_id"=r1."kit_id" and "r5"."participant_id"=r1."participant_id" 
#     left join
#     (select  "cairsensMeasure"."timestamp" AS "time",
#      "cairsensMeasure"."level" AS "NO2",
#      "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
#     where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "cairsensMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "cairsensMeasure"."cairsens_id"="cairsens"."id"
#     )as r6 on date_trunc('minute',r1."time")=date_trunc('minute',r6."time")
#     and "r6"."kit_id"=r1."kit_id" and "r6"."participant_id"=r1."participant_id" 
#     left join
#     (
#     select  "ae51Measure"."timestamp" AS "time",
#       "ae51Measure"."bc" AS "BC",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
#     where
#     "ae51Measure"."ae51_id"="kit"."ae51_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "ae51Measure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     )as r7 on date_trunc('minute',r1."time")=date_trunc('minute',r7."time")
#     and "r7"."kit_id"=r1."kit_id" and "r7"."participant_id"=r1."participant_id" 

#     left join
#     (
#     select t1."time",
#     st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "vitesse(m/s)"
#     from(
#     select DISTINCT ON (res1."time")"time","lat","lon"
#     from
#     (select date_trunc('minute', "timestamp") AS "time",
#       "tabletPositionApp"."lat",
#       "tabletPositionApp"."lon"
#     from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
#     where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "tabletPositionApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletPositionApp"."tablet_id"="tablet"."id"
#     ) as res1
#     ) as t1, (
#     select DISTINCT ON (res1."time")"time","lat","lon"
#     from
#     (select date_trunc('minute', "timestamp") AS "time",
#       "tabletPositionApp"."lat",
#       "tabletPositionApp"."lon"
#     from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
#     where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "tabletPositionApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletPositionApp"."tablet_id"="tablet"."id"
#     ) as res1
#     ) as t2
#     where t2."time"=t1."time"+ interval '1 minutes' 
#     ) as r11 on date_trunc('minute',r1."time")=date_trunc('minute',r11."time")

#     left join
#     (
#     select "tabletActivityApp"."timestamp" AS "time",
#      "tabletActivityApp"."activity", 
#     lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row,
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
#     where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "tabletActivityApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletActivityApp"."tablet_id"="tablet"."id"
#     ) as r8 on date_trunc('minute',r1."time") between r8."time" and r8.next_row
#     and "r8"."kit_id"=r1."kit_id" and "r8"."participant_id"=r1."participant_id" 
#     left join
#     (
#     select "tabletEventApp"."timestamp" AS "time",
#     "tabletEventApp"."event",
#     lead("timestamp") over (order by "tabletEventApp".id asc) as next_row,
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
#     where "tabletEventApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "tabletEventApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletEventApp"."tablet_id"="tablet"."id"
#     ) as r9 on date_trunc('minute',r1."time") between r9."time" and r9.next_row
#     and "r9"."kit_id"=r1."kit_id" and "r9"."participant_id"=r1."participant_id" 
#     )as res
#     order by res."time") as Test

#     '''

#     properties = {'user': 'docker', 'password': 'docker'}
#     df = spark.read.jdbc(url=url, table=table, properties=properties)
#     df2 = df.toPandas()
#     return df2

# def get_str_of_id(id):
#     return "'"+str(id)+"'"

# #get stored data in postgres based on left join on BC values (we should specify the url[containing username and password], also kit id and participant id)
# def get_BC_data(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',kit_id=57,participant_id=31):    
#     '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
#     ....) and will return them as a dataframe'''
    
#     #table= query
    
#     table = '''
#     (select distinct(res.*)
#     from (
#     select r1."participant_virtual_id", r1."time", r1."BC", r4."Temperature", r5."Humidity", r6."NO2", r7."PM2.5", r2."PM10", r3."PM1.0", r11."vitesse(m/s)", r8."activity"
#     , r9."event"
#     from (
    
#     select  "participant"."participant_virtual_id","ae51Measure"."timestamp" AS "time",
#       "ae51Measure"."bc" AS "BC",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
#     where
#     "ae51Measure"."ae51_id"="kit"."ae51_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "ae51Measure"."bc" is not null
#     and "ae51Measure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     ) as r1
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM10",
#     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 8
#     ) as r2 on date_trunc('minute',r1."time")=date_trunc('minute',r2."time")
#     and "r2"."kit_id"=r1."kit_id" and "r2"."participant_id"=r1."participant_id" 
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM1.0",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 9
#     ) as r3 on date_trunc('minute',r1."time")=date_trunc('minute',r3."time")
#     and "r3"."kit_id"=r1."kit_id" and "r3"."participant_id"=r1."participant_id" 
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "Temperature",
#     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 4
#     ) as r4 on date_trunc('minute',r1."time")=date_trunc('minute',r4."time")
#     and "r4"."kit_id"=r1."kit_id" and "r4"."participant_id"=r1."participant_id" 
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "Humidity",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 5
#     ) as r5 on date_trunc('minute',r1."time")=date_trunc('minute',r5."time")
#     and "r5"."kit_id"=r1."kit_id" and "r5"."participant_id"=r1."participant_id" 
#     left join
#     (select  "cairsensMeasure"."timestamp" AS "time",
#      "cairsensMeasure"."level" AS "NO2",
#      "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
#     where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "cairsensMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "cairsensMeasure"."cairsens_id"="cairsens"."id"
#     )as r6 on date_trunc('minute',r1."time")=date_trunc('minute',r6."time")
#     and "r6"."kit_id"=r1."kit_id" and "r6"."participant_id"=r1."participant_id" 
#     left join
#     (
#     select
#         "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM2.5",
#      "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"    
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 7    
#     )as r7 on date_trunc('minute',r1."time")=date_trunc('minute',r7."time")
#     and "r7"."kit_id"=r1."kit_id" and "r7"."participant_id"=r1."participant_id" 

#     left join
#     (
#     select t1."time",
#     st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "vitesse(m/s)"
#     from(
#     select DISTINCT ON (res1."time")"time","lat","lon"
#     from
#     (select date_trunc('minute', "timestamp") AS "time",
#       "tabletPositionApp"."lat",
#       "tabletPositionApp"."lon"
#     from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
#     where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "tabletPositionApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletPositionApp"."tablet_id"="tablet"."id"
#     ) as res1
#     ) as t1, (
#     select DISTINCT ON (res1."time")"time","lat","lon"
#     from
#     (select date_trunc('minute', "timestamp") AS "time",
#       "tabletPositionApp"."lat",
#       "tabletPositionApp"."lon"
#     from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
#     where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "tabletPositionApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletPositionApp"."tablet_id"="tablet"."id"
#     ) as res1
#     ) as t2
#     where t2."time"=t1."time"+ interval '1 minutes' 
#     ) as r11 on date_trunc('minute',r1."time")=date_trunc('minute',r11."time")

#     left join
#     (
#     select "tabletActivityApp"."timestamp" AS "time",
#      "tabletActivityApp"."activity", 
#     lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row,
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
#     where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "tabletActivityApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletActivityApp"."tablet_id"="tablet"."id"
#     ) as r8 on date_trunc('minute',r1."time") between r8."time" and r8.next_row
#     and "r8"."kit_id"=r1."kit_id" and "r8"."participant_id"=r1."participant_id" 
#     left join
#     (
#     select "tabletEventApp"."timestamp" AS "time",
#     "tabletEventApp"."event",
#     lead("timestamp") over (order by "tabletEventApp".id asc) as next_row,
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
#     where "tabletEventApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "tabletEventApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletEventApp"."tablet_id"="tablet"."id"
#     ) as r9 on date_trunc('minute',r1."time") between r9."time" and r9.next_row
#     and "r9"."kit_id"=r1."kit_id" and "r9"."participant_id"=r1."participant_id" 
#     )as res
#     order by res."time") as Test

#     '''

#     properties = {'user': 'docker', 'password': 'docker'}
#     df = spark.read.jdbc(url=url, table=table, properties=properties)
#     df2 = df.toPandas()
#     return df2

# #get stored data in postgres based on left join on NO2 values (we should specify the url[containing username and password], also kit id and participant id)
# def get_NO2_data(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',kit_id=57,participant_id=31):    
#     '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
#     ....) and will return them as a dataframe'''
    
#     #table= query
    
#     table = '''
#     (select distinct(res.*)
#     from (
#     select r1."participant_virtual_id", r1."time", r1."NO2", r4."Temperature", r5."Humidity", r6."BC", r7."PM2.5", r2."PM10", r3."PM1.0", r11."vitesse(m/s)", r8."activity"
#     , r9."event"
#     from (
#     select  "participant"."participant_virtual_id","cairsensMeasure"."timestamp" AS "time",
#      "cairsensMeasure"."level" AS "NO2",
#      "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
#     where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "cairsensMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "cairsensMeasure"."cairsens_id"="cairsens"."id"
#     ) as r1
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM10",
#     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 8
#     ) as r2 on date_trunc('minute',r1."time")=date_trunc('minute',r2."time")
#     and "r2"."kit_id"=r1."kit_id" and "r2"."participant_id"=r1."participant_id" 
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM1.0",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 9
#     ) as r3 on date_trunc('minute',r1."time")=date_trunc('minute',r3."time")
#     and "r3"."kit_id"=r1."kit_id" and "r3"."participant_id"=r1."participant_id" 
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "Temperature",
#     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 4
#     ) as r4 on date_trunc('minute',r1."time")=date_trunc('minute',r4."time")
#     and "r4"."kit_id"=r1."kit_id" and "r4"."participant_id"=r1."participant_id" 
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "Humidity",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 5
#     ) as r5 on date_trunc('minute',r1."time")=date_trunc('minute',r5."time")
#     and "r5"."kit_id"=r1."kit_id" and "r5"."participant_id"=r1."participant_id" 
#     left join
#     (
#     select  "participant"."participant_virtual_id","ae51Measure"."timestamp" AS "time",
#       "ae51Measure"."bc" AS "BC",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
#     where
#     "ae51Measure"."ae51_id"="kit"."ae51_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "ae51Measure"."bc" is not null
#     and "ae51Measure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     )as r6 on date_trunc('minute',r1."time")=date_trunc('minute',r6."time")
#     and "r6"."kit_id"=r1."kit_id" and "r6"."participant_id"=r1."participant_id" 
#     left join
#     (
#     select
#         "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM2.5",
#      "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"    
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 7    
#     )as r7 on date_trunc('minute',r1."time")=date_trunc('minute',r7."time")
#     and "r7"."kit_id"=r1."kit_id" and "r7"."participant_id"=r1."participant_id" 

#     left join
#     (
#     select t1."time",
#     st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "vitesse(m/s)"
#     from(
#     select DISTINCT ON (res1."time")"time","lat","lon"
#     from
#     (select date_trunc('minute', "timestamp") AS "time",
#       "tabletPositionApp"."lat",
#       "tabletPositionApp"."lon"
#     from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
#     where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "tabletPositionApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletPositionApp"."tablet_id"="tablet"."id"
#     ) as res1
#     ) as t1, (
#     select DISTINCT ON (res1."time")"time","lat","lon"
#     from
#     (select date_trunc('minute', "timestamp") AS "time",
#       "tabletPositionApp"."lat",
#       "tabletPositionApp"."lon"
#     from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
#     where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "tabletPositionApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletPositionApp"."tablet_id"="tablet"."id"
#     ) as res1
#     ) as t2
#     where t2."time"=t1."time"+ interval '1 minutes' 
#     ) as r11 on date_trunc('minute',r1."time")=date_trunc('minute',r11."time")

#     left join
#     (
#     select "tabletActivityApp"."timestamp" AS "time",
#      "tabletActivityApp"."activity", 
#     lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row,
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
#     where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "tabletActivityApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletActivityApp"."tablet_id"="tablet"."id"
#     ) as r8 on date_trunc('minute',r1."time") between r8."time" and r8.next_row
#     and "r8"."kit_id"=r1."kit_id" and "r8"."participant_id"=r1."participant_id" 
#     left join
#     (
#     select "tabletEventApp"."timestamp" AS "time",
#     "tabletEventApp"."event",
#     lead("timestamp") over (order by "tabletEventApp".id asc) as next_row,
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
#     where "tabletEventApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "tabletEventApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletEventApp"."tablet_id"="tablet"."id"
#     ) as r9 on date_trunc('minute',r1."time") between r9."time" and r9.next_row
#     and "r9"."kit_id"=r1."kit_id" and "r9"."participant_id"=r1."participant_id" 
#     )as res
#     order by res."time") as Test

#     '''

#     properties = {'user': 'docker', 'password': 'docker'}
#     df = spark.read.jdbc(url=url, table=table, properties=properties)
#     df2 = df.toPandas()
#     return df2

# #get stored data in postgres based on left join on Temperature values (we should specify the url[containing username and password], also kit id and participant id)
# def get_Temperature_data(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',kit_id=57,participant_id=31):    
#     '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
#     ....) and will return them as a dataframe'''
    
#     #table= query
    
#     table = '''
#     (select distinct(res.*)
#     from (
#     select r1."participant_virtual_id", r1."time", r1."Temperature",  r5."Humidity",r4."NO2", r6."BC", r7."PM2.5", r2."PM10", r3."PM1.0", r11."vitesse(m/s)", r8."activity"
#     , r9."event"
#     from (
    
    
#     select  "participant"."participant_virtual_id","canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "Temperature",
#     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 4
#     ) as r1
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM10",
#     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 8
#     ) as r2 on date_trunc('minute',r1."time")=date_trunc('minute',r2."time")
#     and "r2"."kit_id"=r1."kit_id" and "r2"."participant_id"=r1."participant_id" 
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM1.0",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 9
#     ) as r3 on date_trunc('minute',r1."time")=date_trunc('minute',r3."time")
#     and "r3"."kit_id"=r1."kit_id" and "r3"."participant_id"=r1."participant_id" 
#     Left join (    
#     select "cairsensMeasure"."timestamp" AS "time",
#      "cairsensMeasure"."level" AS "NO2",
#      "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
#     where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"    
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "cairsensMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "cairsensMeasure"."cairsens_id"="cairsens"."id"
#     ) as r4 on date_trunc('minute',r1."time")=date_trunc('minute',r4."time")
#     and "r4"."kit_id"=r1."kit_id" and "r4"."participant_id"=r1."participant_id" 
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "Humidity",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 5
#     ) as r5 on date_trunc('minute',r1."time")=date_trunc('minute',r5."time")
#     and "r5"."kit_id"=r1."kit_id" and "r5"."participant_id"=r1."participant_id" 
#     left join
#     (
#     select  "participant"."participant_virtual_id","ae51Measure"."timestamp" AS "time",
#       "ae51Measure"."bc" AS "BC",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
#     where
#     "ae51Measure"."ae51_id"="kit"."ae51_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "ae51Measure"."bc" is not null
#     and "ae51Measure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     )as r6 on date_trunc('minute',r1."time")=date_trunc('minute',r6."time")
#     and "r6"."kit_id"=r1."kit_id" and "r6"."participant_id"=r1."participant_id" 
#     left join
#     (
#     select
#         "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM2.5",
#      "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"    
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 7    
#     )as r7 on date_trunc('minute',r1."time")=date_trunc('minute',r7."time")
#     and "r7"."kit_id"=r1."kit_id" and "r7"."participant_id"=r1."participant_id" 

#     left join
#     (
#     select t1."time",
#     st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "vitesse(m/s)"
#     from(
#     select DISTINCT ON (res1."time")"time","lat","lon"
#     from
#     (select date_trunc('minute', "timestamp") AS "time",
#       "tabletPositionApp"."lat",
#       "tabletPositionApp"."lon"
#     from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
#     where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "tabletPositionApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletPositionApp"."tablet_id"="tablet"."id"
#     ) as res1
#     ) as t1, (
#     select DISTINCT ON (res1."time")"time","lat","lon"
#     from
#     (select date_trunc('minute', "timestamp") AS "time",
#       "tabletPositionApp"."lat",
#       "tabletPositionApp"."lon"
#     from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
#     where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "tabletPositionApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletPositionApp"."tablet_id"="tablet"."id"
#     ) as res1
#     ) as t2
#     where t2."time"=t1."time"+ interval '1 minutes' 
#     ) as r11 on date_trunc('minute',r1."time")=date_trunc('minute',r11."time")

#     left join
#     (
#     select "tabletActivityApp"."timestamp" AS "time",
#      "tabletActivityApp"."activity", 
#     lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row,
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
#     where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "tabletActivityApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletActivityApp"."tablet_id"="tablet"."id"
#     ) as r8 on date_trunc('minute',r1."time") between r8."time" and r8.next_row
#     and "r8"."kit_id"=r1."kit_id" and "r8"."participant_id"=r1."participant_id" 
#     left join
#     (
#     select "tabletEventApp"."timestamp" AS "time",
#     "tabletEventApp"."event",
#     lead("timestamp") over (order by "tabletEventApp".id asc) as next_row,
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
#     where "tabletEventApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "tabletEventApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletEventApp"."tablet_id"="tablet"."id"
#     ) as r9 on date_trunc('minute',r1."time") between r9."time" and r9.next_row
#     and "r9"."kit_id"=r1."kit_id" and "r9"."participant_id"=r1."participant_id" 
#     )as res
#     order by res."time") as Test

#     '''

#     properties = {'user': 'docker', 'password': 'docker'}
#     df = spark.read.jdbc(url=url, table=table, properties=properties)
#     df2 = df.toPandas()
#     return df2

# #get stored data in postgres based on left join on Humidity values (we should specify the url[containing username and password], also kit id and participant id)
# def get_Humidity_data(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',kit_id=57,participant_id=31):    
#     '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
#     ....) and will return them as a dataframe'''
    
#     #table= query
    
#     table = '''
#     (select distinct(res.*)
#     from (
#     select r1."participant_virtual_id", r1."time", r1."Humidity",  r5."Temperature",r4."NO2", r6."BC", r7."PM2.5", r2."PM10", r3."PM1.0", r11."vitesse(m/s)", r8."activity"
#     , r9."event"
#     from (    
#     select  "participant"."participant_virtual_id","canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "Humidity",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 5
#     ) as r1
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM10",
#     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 8
#     ) as r2 on date_trunc('minute',r1."time")=date_trunc('minute',r2."time")
#     and "r2"."kit_id"=r1."kit_id" and "r2"."participant_id"=r1."participant_id" 
#     Left join (
#     select  "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM1.0",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 9
#     ) as r3 on date_trunc('minute',r1."time")=date_trunc('minute',r3."time")
#     and "r3"."kit_id"=r1."kit_id" and "r3"."participant_id"=r1."participant_id" 
#     Left join (    
#     select "cairsensMeasure"."timestamp" AS "time",
#      "cairsensMeasure"."level" AS "NO2",
#      "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
#     where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"    
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "cairsensMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "cairsensMeasure"."cairsens_id"="cairsens"."id"
#     ) as r4 on date_trunc('minute',r1."time")=date_trunc('minute',r4."time")
#     and "r4"."kit_id"=r1."kit_id" and "r4"."participant_id"=r1."participant_id" 
#     Left join (
#     select "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "Temperature",
#     "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"    
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 4
#     ) as r5 on date_trunc('minute',r1."time")=date_trunc('minute',r5."time")
#     and "r5"."kit_id"=r1."kit_id" and "r5"."participant_id"=r1."participant_id" 
#     left join
#     (
#     select  "participant"."participant_virtual_id","ae51Measure"."timestamp" AS "time",
#       "ae51Measure"."bc" AS "BC",
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
#     where
#     "ae51Measure"."ae51_id"="kit"."ae51_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "ae51Measure"."bc" is not null
#     and "ae51Measure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     )as r6 on date_trunc('minute',r1."time")=date_trunc('minute',r6."time")
#     and "r6"."kit_id"=r1."kit_id" and "r6"."participant_id"=r1."participant_id" 
#     left join
#     (
#     select
#         "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM2.5",
#      "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"    
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 7    
#     )as r7 on date_trunc('minute',r1."time")=date_trunc('minute',r7."time")
#     and "r7"."kit_id"=r1."kit_id" and "r7"."participant_id"=r1."participant_id" 

#     left join
#     (
#     select t1."time",
#     st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "vitesse(m/s)"
#     from(
#     select DISTINCT ON (res1."time")"time","lat","lon"
#     from
#     (select date_trunc('minute', "timestamp") AS "time",
#       "tabletPositionApp"."lat",
#       "tabletPositionApp"."lon"
#     from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
#     where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "tabletPositionApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletPositionApp"."tablet_id"="tablet"."id"
#     ) as res1
#     ) as t1, (
#     select DISTINCT ON (res1."time")"time","lat","lon"
#     from
#     (select date_trunc('minute', "timestamp") AS "time",
#       "tabletPositionApp"."lat",
#       "tabletPositionApp"."lon"
#     from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
#     where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "tabletPositionApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletPositionApp"."tablet_id"="tablet"."id"
#     ) as res1
#     ) as t2
#     where t2."time"=t1."time"+ interval '1 minutes' 
#     ) as r11 on date_trunc('minute',r1."time")=date_trunc('minute',r11."time")

#     left join
#     (
#     select "tabletActivityApp"."timestamp" AS "time",
#      "tabletActivityApp"."activity", 
#     lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row,
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
#     where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "tabletActivityApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletActivityApp"."tablet_id"="tablet"."id"
#     ) as r8 on date_trunc('minute',r1."time") between r8."time" and r8.next_row
#     and "r8"."kit_id"=r1."kit_id" and "r8"."participant_id"=r1."participant_id" 
#     left join
#     (
#     select "tabletEventApp"."timestamp" AS "time",
#     "tabletEventApp"."event",
#     lead("timestamp") over (order by "tabletEventApp".id asc) as next_row,
#         "kit"."id" as "kit_id", "participant"."id" as "participant_id"
#     from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
#     where "tabletEventApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "tabletEventApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletEventApp"."tablet_id"="tablet"."id"
#     ) as r9 on date_trunc('minute',r1."time") between r9."time" and r9.next_row
#     and "r9"."kit_id"=r1."kit_id" and "r9"."participant_id"=r1."participant_id" 
#     )as res
#     order by res."time") as Test

#     '''

#     properties = {'user': 'docker', 'password': 'docker'}
#     df = spark.read.jdbc(url=url, table=table, properties=properties)
#     df2 = df.toPandas()
#     return df2









# #get stored activities of a specific participant depending on his kit an participant id
# def get_activity_data(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',kit_id=57,participant_id=31):    
#     '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
#     ....) and will return them as a dataframe'''
    
#     #table= query
    
#     table = '''
#     (select Test.participant_virtual_id,Test.time,Test.activity,Test.act_time as actual_activity_time from (
#         (select distinct(res.*),lead("time") over (order by res desc) as res_pre_row
#     from (
#     select r1."participant_virtual_id",r1.time,r1.next_row as t,r1.kit_id,r1.participant_id
#     from (
#     select  "participant"."participant_virtual_id",
#         "canarinMeasure"."timestamp" AS "time",
#       "canarinMeasure"."value_num" AS "PM2.5",
#      "kit"."id" as "kit_id", "participant"."id" as "participant_id",
#      lead("timestamp") over (order by "canarinMeasure".id asc) as next_row,
#      lead("timestamp") over (order by "canarinMeasure".id desc) as pre_row
#     from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
#     where "canarinMeasure"."canarin_id"="kit"."canarin_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "campaignParticipantKit"."participant_id"="participant"."id"    
#     and "canarinMeasure"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "canarinMeasure"."canarin_id"="canarin"."id"
#     and "canarinMeasure"."type_id" = 7
#     ) as r1    )
    
#      as res order by res."time") as r2
#      left join
#     (select "tabletActivityApp"."timestamp" AS "act_time",
#              "tabletActivityApp"."activity", 
#             lead("timestamp") over (order by "tabletActivityApp".id asc) as act_next_row,
#                 "kit"."id" as "kkit_id", "participant"."id" as "pparticipant_id"
#         from "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
#         where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
#         and "kit"."id"="campaignParticipantKit"."kit_id"
#         and "campaignParticipantKit"."participant_id"="participant"."id"
#         and "tabletActivityApp"."timestamp"
#         between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#         and "tabletActivityApp"."tablet_id"="tablet"."id"
#         and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''

#         UNION

#         select date_trunc('minute',"tabletEventApp"."timestamp") AS "act_time",
#         "tabletEventApp"."event",
#             lead("timestamp") over (order by "tabletEventApp".id asc) as act_next_row,
#                 "kit"."id" as "kkit_id", "participant"."id" as "pparticipant_id"
#         from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
#         where "tabletEventApp"."tablet_id"="kit"."tablet_id"
#         and "kit"."id"="campaignParticipantKit"."kit_id"
#         and "campaignParticipantKit"."participant_id"="participant"."id"
#         and "tabletEventApp"."timestamp"
#         between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#         and "tabletEventApp"."tablet_id"="tablet"."id"
#         and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
        
#     ) as r8 on (date_trunc('minute',r2."time") between r8."act_time" and r8.act_next_row
#     and (date_trunc('minute',r2.res_pre_row) not between r8."act_time" and r8.act_next_row )
#     and "r8"."kkit_id"=r2."kit_id" and "r8"."pparticipant_id"=r2."participant_id")
#     or (date_trunc('minute',r2."time") between r8."act_time" and r8.act_next_row
#     and (r2.res_pre_row is null)
#     and "r8"."kkit_id"=r2."kit_id" and "r8"."pparticipant_id"=r2."participant_id")
#     )
    
#     as Test
#     where Test.activity is not null) as ae 

#     '''

#     properties = {'user': 'docker', 'password': 'docker'}
#     df = spark.read.jdbc(url=url, table=table, properties=properties)
#     df2 = df.toPandas()
#     return df2

# def get_GPS_data(url='jdbc:postgresql://193.55.95.225:32263/polluscopev5',kit_id=57,participant_id=31):    
#     '''This Function aims to retrieve data from database it will return all data if exists e.g. (time,tempreture,humidity,BC
#     ....) and will return them as a dataframe'''
    
#     #table= query
    
#     table = '''
#     (select distinct(res.*)
#     from (
#     select t1."time",t1."lat",t1."lon"
#     from(
#     select DISTINCT ON (res1."time")"time","lat","lon"
#     from
#     (select date_trunc('minute', "timestamp") AS "time",
#       "tabletPositionApp"."lat",
#       "tabletPositionApp"."lon"
#     from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
#     where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
#     and "kit"."id"="campaignParticipantKit"."kit_id"
#     and "campaignParticipantKit"."participant_id"="participant"."id"
#     and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
#     and "tabletPositionApp"."timestamp"
#     between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
#     and "tabletPositionApp"."tablet_id"="tablet"."id"
#     ) as res1
#     ) as t1
#     )as res
#     order by res."time") as Test

#     '''

#     properties = {'user': 'docker', 'password': 'docker'}
#     df = spark.read.jdbc(url=url, table=table, properties=properties)
#     df2 = df.toPandas()
#     return df2

def get_preprocessed_data_RECORD(url='postgresql://postgres:root@localhost:5432/db_RECORD',table_name='data_processed_RECORD',participant_virtual_id=None):
    engine = db.create_engine(url) 
    connection = engine.connect()
    metadata = db.MetaData()
    data = db.Table(table_name, metadata, autoload=True, autoload_with=engine)
    if participant_virtual_id==None:
        results = connection.execute(db.select([data])).fetchall()
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        df=df.sort_values(by=["time"])
        df=df.drop_duplicates(subset=["time"])
        return df
    else:
        results = connection.execute(db.select([data]).where(data.columns.participant_virtual_id==str(participant_virtual_id))).fetchall()
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        df=df.sort_values(by=["time"])
        df=df.drop_duplicates(subset=["time"])
        return df


def get_preprocessed_data_VGP(url='postgresql://postgres:root@localhost:5432/preprocessed_data',table_name='data_processed_vgp',participant_virtual_id=None):
    engine = db.create_engine(url) 
    connection = engine.connect()
    metadata = db.MetaData()
    data = db.Table(table_name, metadata, autoload=True, autoload_with=engine)
    if participant_virtual_id==None:
        results = connection.execute(db.select([data])).fetchall()
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        df=df.sort_values(by=["time"])
        df=df.drop_duplicates(subset=["time"])
        return df
    else:
        results = connection.execute(db.select([data]).where(data.columns.participant_virtual_id==str(participant_virtual_id))).fetchall()
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        df=df.sort_values(by=["time"])
        df=df.drop_duplicates(subset=["time"])
        return df
    

def get_postgres_data_sqlelchemy(url='postgresql://postgres:root@localhost:5432/vgp',participant_virtual_id=None,campaign_id='2'):
    engine = db.create_engine(url) 
    connection = engine.connect()
    metadata = db.MetaData()
    query = '''
    select distinct(res.*)
    from(
    select "participant"."id" as "participant_id", "kit"."id" as "kit_id" from "campaignParticipantKit","participant","kit"
    where "campaignParticipantKit"."participant_id"="participant"."id"
    and "campaignParticipantKit"."kit_id"="kit"."id"
    and "participant"."participant_virtual_id"='''+get_str_of_id(participant_virtual_id)+''')as res
    '''
    results = connection.execute(query).fetchall()
    df = pd.DataFrame(results)
    df.columns = results[0].keys()
    kit_id = df['kit_id'].iloc[0]
    participant_id = df['participant_id'].iloc[0]
    
    query = '''
    select distinct(res.*)
    from (
    select r1."participant_virtual_id", r1."time" :: timestamp, r4."Temperature", r5."Humidity", r6."NO2", r7."BC",r3."PM1.0",q."PM2.5",r2."PM10", r11."vitesse(m/s)", r8."activity"
    , r9."event"
    from (
		SELECT "participant"."participant_virtual_id", "kit"."id" as "kit_id", "participant"."id" as "participant_id",generate_series(
	(select min(start_date) from public."campaignParticipantKit" where campaign_id = '''+campaign_id+''') ,
	(select max(end_date) from public."campaignParticipantKit" where campaign_id = '''+campaign_id+'''),
	'1 min'::interval) as time from "participant", "kit","campaignParticipantKit"
	where
	"kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
	and "kit"."id"="campaignParticipantKit"."kit_id"
	and "campaignParticipantKit"."participant_id"="participant"."id"
	) as r1 
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
		
	left join(
	select "canarinMeasure"."timestamp" AS "time",
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
		
	) as q on date_trunc('minute',r1."time")=date_trunc('minute',q."time")
    and "q"."kit_id"=r1."kit_id" and "q"."participant_id"=r1."participant_id" 
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
		
    ) as res
    '''
    
    results = connection.execute(query).fetchall()
    df = pd.DataFrame(results)
    df.columns = results[0].keys()
    df=df.sort_values(by=["time"])
    df=df.drop_duplicates(subset=["time"])
    indexNames = df[ (df['Temperature'].isnull()) & (df['Humidity'].isnull()) & (df['NO2'].isnull()) & (df['BC'].isnull()) & (df['PM1.0'].isnull()) & (df['PM2.5'].isnull()) & (df['PM10'].isnull()) & (df['vitesse(m/s)'].isnull()) ].index
    df.drop(indexNames , inplace=True)
    df.reset_index(inplace=True)
    return df