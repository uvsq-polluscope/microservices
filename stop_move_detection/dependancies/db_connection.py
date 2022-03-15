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
import datetime as dt
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None

#in general: engine = create_engine('dialect+driver://username:password@host:port/database'
# engine = create_engine(''postgresql://dwaccount:password@127.0.0.1:5435/'+"RECORD2")
# engine_vgp = create_engine('postgresql://dwaccount:password@127.0.0.1:5435/'+"VGP")

url_original_data='postgresql://dwaccount:password@127.0.0.1:5435/dwaccount'

# engine_processed = create_engine(''postgresql://dwaccount:password@127.0.0.1:5435/'+"processed_data")
# engine_vgp = create_engine('postgresql://postgres:postgres@localhost:5432/'+"VGP")

engine = create_engine(url_original_data)
engine_vgp = create_engine(url_original_data)


# In[5]:

def get_str_of_id(id):
    return "'"+str(id)+"'"


# In[12]:


def get_gps(participant_virtual_id):
    df = pd.read_sql('''select participant_virtual_id, time_gps as time, lon, lat, activity
    from data_processed_RECORD_v3
    where participant_virtual_id='''+get_str_of_id(participant_virtual_id)+'''
    and lon is not null
    order by 2''', engine)
    
    return df

# In[3]:

def get_gps_2(participant_virtual_id):
    df = pd.read_sql('''select r1.*, r2.activity
    from
    (select * from clean_gps
    where participant_virtual_id='''+get_str_of_id(participant_virtual_id)+''') as r1
    left join
    (select participant_virtual_id, time, activity
    from "data_processed_RECORD") as r2
    on r1.participant_virtual_id=r2.participant_virtual_id
    and date_trunc('minute', r1.time ) = date_trunc('minute', r2.time )
    order by 2''', engine)
    
    return df

# In[4]:

def get_gps_hilbert(participant_virtual_id):
    df = pd.read_sql('''select participant_virtual_id, time_gps as time, lon, lat, hilbert, activity
    from data_processed_RECORD_v3
    where participant_virtual_id='''+get_str_of_id(participant_virtual_id)+'''
    and lon is not null
    order by 2''', engine)
    
    return df

def get_gps_hilbert_vgp(participant_virtual_id):
    df = pd.read_sql('''select participant_virtual_id, time, lon, lat, hilbert, activity
    from clean_gps_with_activity
    where participant_virtual_id='''+get_str_of_id(participant_virtual_id)+'''
    and lon is not null
    order by 2''', engine)
    
    return df