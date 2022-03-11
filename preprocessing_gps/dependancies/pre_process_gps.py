from datetime import datetime
from pyproj import Proj, transform
import math
import os
from hilbertcurve.hilbertcurve import HilbertCurve
from sqlalchemy.orm import scoped_session, sessionmaker
import sqlalchemy as db
from skmob.preprocessing import filtering
from skmob.utils import constants, utils, gislib
import six
from sqlalchemy import create_engine
from datetime import timedelta
import skmob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'
distance = gislib.getDistance

engine = create_engine(
    'postgresql://dwaccount:password@127.0.0.1:5435/dwaccount')


def data_pre_processing_gps(data):
    # Rasterize  the data
    Long_min, Long_max, Lat_min, Lat_max = lambert_to_gps(
        Lambert_Long_min, Lambert_Long_max, Lambert_Lat_min, Lambert_Lat_max)

    # Get the max message id from the dataframe
    max_id = data["id"].max()
    # Get the participant_virtual_id of the processed data
    participant_virtual_id = data["participant_virtual_id"][0]

    # Create a hilbert index to each point and store the data in the same table
    hilbert_index(data, max_id, Long_min, Long_max, Lat_min, Lat_max)

    # Store data in clean_gps table
    clean_gps(participant_virtual_id=participant_virtual_id)

    # Create clean_gps_with_activities dataframe and return it
    df = clean_gps_with_activities(participant_virtual_id)
    return df


# Create a hilbert index to each point and store the data in the same table
def hilbert_index(data, max_id, Long_min, Long_max, Lat_min, Lat_max):
    id = 0
    while (id < max_id):
        # apply the hilbert function on this pandas dataframe
        data = add_hilbert_index(
            data, Long_min, Long_max, Lat_min, Lat_max, nb_c, nb_r, dimensions, iterations)

        # Create a database session to store updated data
        db = scoped_session(sessionmaker(bind=engine))
        # loop to update the postgres table by adding the col_num, row-num and hilbert according to the id of the tabletPositionApp table
        for index, row in data.iterrows():
            print("update " + str(row['id']))
            db.execute('UPDATE "tabletPositionApp" SET col_num='+str(row['col_num'])+', row_num='+str(
                row['row_num'])+', hilbert='+str(row['hilbert'])+' WHERE id='+str(row['id']))
        db.commit()
        db.close()
        print("fin boucle")
        id = data['id'].max()+1
    return True


def get_str_of_id(id):
    return "'"+str(id)+"'"


# def clean_gps(participant_virtual_id=987014104):
def clean_gps(participant_virtual_id):
    df = pd.read_sql('''select "tabletPositionApp".id, "participant"."participant_virtual_id", "tabletPositionApp"."timestamp"::timestamp AS "datetime",
  "tabletPositionApp"."lat" as lat,
  "tabletPositionApp"."lon" as lng, "tabletPositionApp"."hilbert" as hilbert
from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
and "kit"."id"="campaignParticipantKit"."kit_id"
and "campaignParticipantKit"."participant_id"="participant"."id"
and "tabletPositionApp"."timestamp"
between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
and "tabletPositionApp"."tablet_id"="tablet"."id"
and "participant"."participant_virtual_id"='''+get_str_of_id(participant_virtual_id)+'''
order by 2''', engine)
    print("clean_gps resultat req:")
    print(df)
    tdf = skmob.TrajDataFrame(df, latitude=2, longitude=3, datetime=1)

    ftdf = filtering.filter(tdf, max_speed_kmh=130.)

    data_list = [list(row) for row in ftdf.itertuples(index=False)]

    db1 = scoped_session(sessionmaker(bind=engine))
    for item in data_list:
        db1.execute('''INSERT INTO clean_gps VALUES ('''+get_str_of_id(item[0])+''' , '''+get_str_of_id(
            item[1])+''','''+str(item[2])+''','''+str(item[3])+''','''+str(item[4])+'''); ''')
    db1.commit()
    db1.close()
    print("end of clean_gps updates")

# Long_min, Long_max, Lat_min, Lat_max are the bord of the desired grid
# nb_c,nb_r are the number of verctical and horizontal split of the grid
# dimensions is the dimensions of the grid, dor gps data it should be 2
# iterations: the number of iterations used in constructing the Hilbert curve (must be > 0)
# constraint(should be): nb_c > 2^iterations -1 and nb_r > 2^iterations -1
# h,w means the high and the width of the grid
# cw,ch means the high and the width of each cell


def add_hilbert_index(tablet_position_df, Long_min, Long_max, Lat_min, Lat_max, nb_c, nb_r, dimensions, iterations):
    h = Lat_max - Lat_min
    w = Long_max - Long_min
    cw = w/nb_c
    ch = h/nb_r

    # call the class HilbertCurve
    hilbert_curve = HilbertCurve(iterations, dimensions)

    # Add col_num and row_num to the pandas dataframe
    tablet_position_df['col_num'] = tablet_position_df.apply(
        lambda row: math.floor((row['lon'] - Long_min) / cw), axis=1)
    tablet_position_df['row_num'] = tablet_position_df.apply(
        lambda row: math.floor((row['lat'] - Lat_min) / ch), axis=1)
    # Add hilbert column to pandas datafram with the hilbert values calculated from the col_num and row_num columns, if the gps is out of ile-de-france put hilbert=-1
    # The function hilbert_curve.distance_from_coordinates only works for hilbert_curve V1.0.5, so you must install this specific version (pip install --upgrade hilbertcurve=1.0.5)
    tablet_position_df['hilbert'] = tablet_position_df.apply(lambda row: hilbert_curve.distance_from_coordinates(
        [row['col_num'], row['row_num']]) if row['col_num'] > 0 and row['row_num'] > 0 and row['col_num'] < 3600 and row['row_num'] < 3600 else -1, axis=1)
    print(tablet_position_df)
    return tablet_position_df


# We need this function to transform the lambert to gps because the hilbert funtion work with the gps
def lambert_to_gps(Long_min, Long_max, Lat_min, Lat_max):
    inProj = Proj(init='epsg:27572')
    outProj = Proj(init='epsg:4326')
    Long_min, Lat_min = transform(inProj, outProj, Long_min, Lat_min)
    Long_max, Lat_max = transform(inProj, outProj, Long_max, Lat_max)

    return Long_min, Long_max, Lat_min, Lat_max


def clean_gps_with_activities(participant_virtual_id):

    clean_gps_with_activities = pd.read_sql(''' select r1.*, r2.activity
    from
    (SELECT * FROM clean_gps WHERE participant_virtual_id=''' + get_str_of_id(participant_virtual_id) + ''') as r1
    left join
    (select T."timestamp"::timestamp AS "time",
    T."activity", 
    lead(T."timestamp"::timestamp) over (order by T.id, T."timestamp"::timestamp asc) as next_row, 
        K."id" as "kit_id", P."id" as "participant_id", 
        P."participant_virtual_id" as "participant_virtual_id"
    from "tablet" B,"tabletActivityApp" T,"campaignParticipantKit" C,"kit" K,"participant" P
    where T."tablet_id"=K."tablet_id"
    and K."id"=C."kit_id"
    and C."participant_id"=P."id"
    and T."timestamp"::timestamp
    between C."start_date"::timestamp and C."end_date"::timestamp
    and T."tablet_id"=B."id"
    ) as r2
    on r1.participant_virtual_id = r2.participant_virtual_id
    and date_trunc('minute',r1."time") between date_trunc('minute',r2."time") and date_trunc('minute',r2.next_row - interval '1 sec')
    ''', engine)
    print(clean_gps_with_activities)
    return clean_gps_with_activities


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
Lambert_Lat_min = 2319000
Lambert_Lat_max = 2498950

nb_c, nb_r = 3600, 3600
dimensions, iterations = 2, 12
