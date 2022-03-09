#!/usr/bin/env python


#imports
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.metrics import accuracy_score
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from imblearn.under_sampling import RandomUnderSampler
from pandas import read_csv
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from csv import writer
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from ydata_synthetic.preprocessing.timeseries import processed_stock
#from ydata_synthetic.synthesizers.timeseries import TimeGAN
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from statistics import mean
from dependancies.preprocessing import *



def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


    
def get_real_activities(data,activities):
    data["time"]=pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    activities["time"]=pd.to_datetime(activities['time'], format='%Y-%m-%d %H:%M:%S')
    activities["time"]=activities["time"].dt.round('min')
    print(activities)
    activities["time"]=pd.to_datetime(activities['time'], format='%Y-%m-%d %H:%M:%S')
    result = pd.merge(data, activities, on="time",how="left")
#     result=df
    real_activity=[]
    c=0
    for i in range(len(result)):        
        if result.iloc[i]['activity_y'] is np.nan and c==0:
            real_activity.append(np.nan)
        else:
            if not result.iloc[i]['activity_y'] is np.nan:
                c=1
                current_activity=result.iloc[i]['activity_y']
            real_activity.append(current_activity)
    result['activity']=real_activity
    result['time'] = pd.to_datetime(result['time'], format='%Y-%m-%d %H:%M:%S')
    return result[['participant_virtual_id','time','Temperature','Humidity','NO2','BC','PM1.0','PM2.5','PM10','vitesse(m/s)','activity','event']]

url_processed_data='postgresql://postgres:postgres@192.168.33.123:5432/processed_data'
url_original_data='postgresql://postgres:postgres@192.168.33.123:5432/polluscopev5-last-version-2020'

#change URL and table name
def store_pre_processed_data(dfs=[],url=url_processed_data,table_name='data_processed_RECORD'):
    engine = db.create_engine(url) 
    connection = engine.connect()
    metadata = db.MetaData()
    data = db.Table(table_name, metadata, autoload=True, autoload_with=engine)
    
    query = db.insert(data) 
    c=0    
    for df in dfs:
#         df = df.astype('object')
#         df['id_code'].values.astype(int)
        values_list = []
        df=df.rename(columns={'vitesse(m/s)': 'Speed'})
        c+=1
        print("Storing Data frame number: "+str(c))
        for i in range(len(df)-1):
            d=df.iloc[i].to_dict()            
            for key,value in d.items():
                if str(value)=='nan':
#                     print(key)
                    d[key]=None
            values_list.append(d)
#         print(values_list[0])
        ResultProxy = connection.execute(query,values_list)        
        
    connection.close()    
    print("#################END##############")
    
def retrieve_DB_data_RECORD(participant_virtual_ids=[]): 
    dfs=[]    
    for participant_virtual_id in participant_virtual_ids:
        df=get_preprocessed_data_RECORD(participant_virtual_id=participant_virtual_id)[["participant_virtual_id","time","Temperature","Humidity","NO2","BC","PM1.0","PM2.5","PM10","Speed","activity"]]
        df=df.dropna(subset=["activity"])
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
        
        dffs=splitting(df)
        for j in range(len(dffs)):
            dfs.append(dffs[j]) 
    return dfs
    
def retrieve_DB_data_VGP(participant_virtual_ids=[]): 
    dfs=[]    
    for participant_virtual_id in participant_virtual_ids:
        df=get_preprocessed_data_VGP(participant_virtual_id=participant_virtual_id)[["participant_virtual_id","time","Temperature","Humidity","NO2","BC","PM1.0","PM2.5","PM10","Speed","activity"]]
        df=df.dropna(subset=["activity"])
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
        
        dffs=splitting(df)
        for j in range(len(dffs)):
            dfs.append(dffs[j]) 
    return dfs

def store_data_for_learning_(dfs,activities,columnNames=["PM2.5","PM1.0","PM10","Temperature","Humidity","NO2","BC","vitesse(m/s)"],fileName="out_Train.csv",percentage_split=20,sample_length=5,Id=0):
    dictionary=activities
    counter=1    
    record=1
    
    arr=[]
    classes=[]
    rue_list=[]
    bus_list=[]
    bureau_list=[]
    restaurant_list=[]
    domicile_list=[]
    velo_list=[]
    voiture_list=[]
    magasin_list=[]
    metro_list=[]
    train_list=[]
    cinema_list=[]
    parc_list=[]
    Gare_list=[]
    moto_list=[]
    indoor_list=[]
    running_list=[]
    
    if columnNames !=None:
        cl=columnNames+['activity']
    else:
        cl=None
    
    for dff in dfs:
        if cl==None:
            columnNames=list(set(dff.columns))
        else:
            columnNames=None
            columnNames=cl
        dropNA_columns=columnNames#+['activity']        
    
        #print(list(set(dff.columns)))
        
        if "NO2" not in list(set(dff.columns)) and (columnNames[0]=="NO2" or cl==None):
            print("continue")
            continue
        
        if "BC" not in list(set(dff.columns)) and (columnNames[0]=="BC" or cl==None) :
            print("continue")
            continue
            

            
        dff=dff.dropna(subset=dropNA_columns)
        Id+=1
        
        for i in range(len(dff)):
            
            
            arr=[]
            
            if dff.iloc[i]["activity"]!="Inconnu" and dff.iloc[i]["activity"]!="Train" and dff.iloc[i]["activity"]!="Cinéma"  and dff.iloc[i]["activity"]!="Indoor":
                arr.append(Id)
                arr.append(i+1)

                arr.append(dictionary[dff.iloc[i]["activity"]])
                if len(classes)==0:
                    classes.append(dictionary[dff.iloc[i]["activity"]])
                else:
                    if dictionary[dff.iloc[i]["activity"]] not in classes:
                        classes.append(dictionary[dff.iloc[i]["activity"]])
                for name in columnNames:
                    if name!="time" and name!="activity":
                        arr.append(dff.iloc[i][name])
                if dff.iloc[i]["activity"]=="Walk":
                    rue_list.append(arr)
                else:
                    if dff.iloc[i]["activity"]=="Bus":
                        bus_list.append(arr)
                    else:
                        if dff.iloc[i]["activity"]=="Bureau" or dff.iloc[i]["activity"]=="Bureau_tiers":
                            bureau_list.append(arr)
                        else:
                            if dff.iloc[i]["activity"]=="Restaurant":
                                restaurant_list.append(arr)
                            else:
                                if dff.iloc[i]["activity"]=="Domicile" or dff.iloc[i]["activity"]=="Domicile_tiers":
                                    domicile_list.append(arr)
                                else:
                                    if dff.iloc[i]["activity"]=="Vélo":
                                        velo_list.append(arr)
                                    else:                                        
                                        if dff.iloc[i]["activity"]=="Voiture" or dff.iloc[i]["activity"]=="Voiture_arrêt":
                                            voiture_list.append(arr)
                                        else:             
                                            if dff.iloc[i]["activity"]=="Magasin":
                                                magasin_list.append(arr)
                                            else:                       
                                                if dff.iloc[i]["activity"]=="Métro":
                                                    metro_list.append(arr)                 
                                                else:
                                                    if dff.iloc[i]["activity"]=="Train" or dff.iloc[i]["activity"]=="Train_arrêt":
                                                        train_list.append(arr)                 
                                                    else:
                                                        if dff.iloc[i]["activity"]=="Cinéma":
                                                            cinema_list.append(arr)                 
                                                        else:
                                                            if dff.iloc[i]["activity"]=="Parc":
                                                                parc_list.append(arr)                 
                                                            else:
                                                                if dff.iloc[i]["activity"]=="Gare":
                                                                    Gare_list.append(arr)        
                                                                else:
                                                                    if dff.iloc[i]["activity"]=="Motorcycle":
                                                                        moto_list.append(arr)        
                                                                    else:
                                                                        if dff.iloc[i]["activity"]=="Indoor":
                                                                            indoor_list.append(arr)        
                                                                        else:
                                                                            if dff.iloc[i]["activity"]=="Running":
                                                                                running_list.append(arr)        
                                                                            
                                                                        
                                                                    
                                            
                                    
                            
                
            record+=1
        counter+=1
    
    c=0
    
    prev_rue=None
    first_attempt=True
    for rue in rue_list:     
        append_list_as_row(fileName,rue)
    
    c=0
    prev_bus=None
    first_attempt=True
    for bus in bus_list:
        append_list_as_row(fileName,bus)

    c=0
    prev_bureau=None
    first_attempt=True
    for bureau in bureau_list:
        append_list_as_row(fileName,bureau)
        
    c=0
    prev_restaurant=None
    first_attempt=True
    for restaurant in restaurant_list:
        append_list_as_row(fileName,restaurant)

    c=0
    prev_domicile=None
    first_attempt=True
    for domicile in domicile_list:
        append_list_as_row(fileName,domicile)
        
    c=0
    prev_velo=None
    first_attempt=True
    for velo in velo_list:
        append_list_as_row(fileName,velo)
        
    c=0
    prev_voiture=None
    first_attempt=True
    for voiture in voiture_list:
        append_list_as_row(fileName,voiture)
        
    c=0
    prev_magasin=None
    first_attempt=True
    for magasin in magasin_list:
        append_list_as_row(fileName,magasin)
        
    c=0
    prev_metro=None
    first_attempt=True
    for metro in metro_list:
        append_list_as_row(fileName,metro)
    
    c=0
    prev_train=None
    first_attempt=True
    for train in train_list:
        append_list_as_row(fileName,train)
        
    c=0
    prev_cinema=None
    first_attempt=True
    for cinema in cinema_list:
        append_list_as_row(fileName,cinema)
        
    c=0    
    prev_parc=None
    first_attempt=True
    for parc in parc_list:
        append_list_as_row(fileName,parc)

    c=0    
    prev_gare=None
    first_attempt=True
    for gare in Gare_list:
        append_list_as_row(fileName,gare)
    
    c=0    
    prev_moto=None
    first_attempt=True
    for moto in moto_list:
        append_list_as_row(fileName,moto)
        
    c=0    
    prev_indoor=None
    first_attempt=True
    for indoor in indoor_list:
        append_list_as_row(fileName,indoor)
        
    
    c=0    
    prev_running=None
    first_attempt=True
    for running in running_list:
        append_list_as_row(fileName,running)
    
    
    return classes




def store_data_for_learning_same_dimensions(dfs,activities,columnNames=["PM2.5","PM1.0","PM10","Temperature","Humidity","NO2","BC","vitesse(m/s)"],fileName="out_Train.csv",percentage_split=20,sample_length=5,Id=0):
    dictionary=activities
    counter=1    
    record=1
#  
    
    classes=[]
    rue_list=[]
    bus_list=[]
    bureau_list=[]
    restaurant_list=[]
    domicile_list=[]
    velo_list=[]
    voiture_list=[]
    magasin_list=[]
    metro_list=[]
    train_list=[]
    cinema_list=[]
    parc_list=[]
    Gare_list=[]
    moto_list=[]
    indoor_list=[]
    running_list=[]
    
    if columnNames !=None:
        cl=["PM2.5","PM1.0","PM10","Temperature","Humidity","NO2","BC","Speed"]+['activity']
        cl=columnNames+["activity"]
    else:
        cl=None
    for dff in dfs:
        
        if cl==None:
            columnNames=list(set(dff.columns))
        else:
            columnNames=None
            columnNames=cl
        dropNA_columns=columnNames#+['activity']        
        print(list(set(dff.columns)))
        if "NO2" not in list(set(dff.columns)):
            print("continue")
            continue
        
        if "BC" not in list(set(dff.columns)):
            print("continue")
            continue
        
            
        dropNA_columns=["PM2.5","PM1.0","PM10","Temperature","Humidity","NO2","BC","Speed",'activity']
        dff=dff.dropna(subset=dropNA_columns)
        Id+=1
        for i in range(len(dff)):                 
            

            arr=[]

            if dff.iloc[i]["activity"]!="Inconnu" and dff.iloc[i]["activity"]!="Train" and dff.iloc[i]["activity"]!="Cinéma"  and dff.iloc[i]["activity"]!="Indoor":
                arr.append(Id)
                arr.append(i+1)

                arr.append(dictionary[dff.iloc[i]["activity"]])
                if len(classes)==0:
                    classes.append(dictionary[dff.iloc[i]["activity"]])
                else:
                    if dictionary[dff.iloc[i]["activity"]] not in classes:
                        classes.append(dictionary[dff.iloc[i]["activity"]])
                for name in columnNames:
                    if name!="time" and name!="activity":
                        arr.append(dff.iloc[i][name])
                if dff.iloc[i]["activity"]=="Walk":
                    rue_list.append(arr)
                else:
                    if dff.iloc[i]["activity"]=="Bus":
                        bus_list.append(arr)
                    else:
                        if dff.iloc[i]["activity"]=="Bureau" or dff.iloc[i]["activity"]=="Bureau_tiers":
                            bureau_list.append(arr)
                        else:
                            if dff.iloc[i]["activity"]=="Restaurant":
                                restaurant_list.append(arr)
                            else:
                                if dff.iloc[i]["activity"]=="Domicile" or dff.iloc[i]["activity"]=="Domicile_tiers":
                                    domicile_list.append(arr)
                                else:
                                    if dff.iloc[i]["activity"]=="Vélo":
                                        velo_list.append(arr)
                                    else:                                        
                                        if dff.iloc[i]["activity"]=="Voiture" or dff.iloc[i]["activity"]=="Voiture_arrêt":
                                            voiture_list.append(arr)
                                        else:             
                                            if dff.iloc[i]["activity"]=="Magasin":
                                                magasin_list.append(arr)
                                            else:                       
                                                if dff.iloc[i]["activity"]=="Métro":
                                                    metro_list.append(arr)                 
                                                else:
                                                    if dff.iloc[i]["activity"]=="Train" or dff.iloc[i]["activity"]=="Train_arrêt":
                                                        train_list.append(arr)                 
                                                    else:
                                                        if dff.iloc[i]["activity"]=="Cinéma":
                                                            cinema_list.append(arr)                 
                                                        else:
                                                            if dff.iloc[i]["activity"]=="Parc":
                                                                parc_list.append(arr)                 
                                                            else:
                                                                if dff.iloc[i]["activity"]=="Gare":
                                                                    Gare_list.append(arr)        
                                                                else:
                                                                    if dff.iloc[i]["activity"]=="Motorcycle":
                                                                        moto_list.append(arr)        
                                                                    else:
                                                                        if dff.iloc[i]["activity"]=="Indoor":
                                                                            indoor_list.append(arr)        
                                                                        else:
                                                                            if dff.iloc[i]["activity"]=="Running":
                                                                                running_list.append(arr)        
                                                                            
                                                                        
                                                                    
                                            
                                    
                            
                
            record+=1
        counter+=1
    
    c=0
    prev_rue=None
    first_attempt=True
    for rue in rue_list:        
        append_list_as_row(fileName,rue)

    
    c=0
    prev_bus=None
    first_attempt=True
    for bus in bus_list:
        append_list_as_row(fileName,bus)


    c=0
    prev_bureau=None
    first_attempt=True
    for bureau in bureau_list:
        append_list_as_row(fileName,bureau)

        
    c=0
    prev_restaurant=None
    first_attempt=True
    for restaurant in restaurant_list:
        append_list_as_row(fileName,restaurant)


    c=0
    prev_domicile=None
    first_attempt=True
    for domicile in domicile_list:
        append_list_as_row(fileName,domicile)

        
    c=0
    prev_velo=None
    first_attempt=True
    for velo in velo_list:
        append_list_as_row(fileName,velo)

        
    c=0
    prev_voiture=None
    first_attempt=True
    for voiture in voiture_list:
        append_list_as_row(fileName,voiture)

        
    c=0
    prev_magasin=None
    first_attempt=True
    for magasin in magasin_list:
        append_list_as_row(fileName,magasin)

        
    c=0
    prev_metro=None
    first_attempt=True
    for metro in metro_list:
        append_list_as_row(fileName,metro)

    
    c=0
    prev_train=None
    first_attempt=True
    for train in train_list:
        append_list_as_row(fileName,train)

        
    c=0
    prev_cinema=None
    first_attempt=True
    for cinema in cinema_list:
        append_list_as_row(fileName,cinema)

        
    c=0    
    prev_parc=None
    first_attempt=True
    for parc in parc_list:
        append_list_as_row(fileName,parc)


    c=0    
    prev_gare=None
    first_attempt=True
    for gare in Gare_list:
        append_list_as_row(fileName,gare)

    
    c=0    
    prev_moto=None
    first_attempt=True
    for moto in moto_list:
        append_list_as_row(fileName,moto)

        
    c=0    
    prev_indoor=None
    first_attempt=True
    for indoor in indoor_list:
        append_list_as_row(fileName,indoor)

        
    
    c=0    
    prev_running=None
    first_attempt=True
    for running in running_list:
        append_list_as_row(fileName,running)
    
    
    return classes




def real_data_loading(data: np.array, seq_len):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: Numpy array with the values from a a Dataset
      - seq_len: sequence length
    Returns:
      - data: preprocessed data.
    """
    # Flip the data to make chronological data
    ori_data = data[::-1]
    # Normalize the data
    scaler = MinMaxScaler().fit(ori_data)
    ori_data = scaler.transform(ori_data)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    return data




def z_normalization(mts):    
    scaler = MinMaxScaler()
    scaler.fit(mts)
    mts = scaler.transform(mts)
    return mts,scaler




# #Specific to TimeGANs
# seq_len=5
# n_seq = 1
# hidden_dim=5
# gamma=1

# noise_dim = 6
# dim = 128
# batch_size = 128

# log_step = 100
# learning_rate = 5e-4

# gan_args = [batch_size, learning_rate, noise_dim, 24, 2, (0, 1), dim]




def split_data_by_class(data):
    rue=[]
    bus=[]
    bureau=[]
    restaurant=[]
    domicile=[]
    voiture=[]
    velo=[]
    metro=[]
    running=[]
    magasin=[]
    gare=[]
    motorcycle=[]
    parc=[]

    for d,label in zip(data[0],data[1]):
        if label==1:
            rue.append(d)
        if label==2:
            bus.append(d)
        if label==3:
            bureau.append(d)
        if label==4:
            restaurant.append(d)
        if label==5:
            domicile.append(d)
        if label==6:
            velo.append(d)
        if label==7:
            voiture.append(d)
        if label==8:
            magasin.append(d)
        if label==9:
            metro.append(d)
        if label==10:
            gare.append(d)
        if label==11:
            motorcycle.append(d)
        if label==12:
            running.append(d)        
        if label==13:
            parc.append(d)
    
    return rue,bus,bureau,restaurant,domicile,velo,voiture,magasin,metro,gare,motorcycle,running,parc
        



def store_data_after_generation(file_name,data,labels):    
    Id=0
    
    for d,l in zip(data,labels):                
        for i in range(len(d)):
            Id+=1
            c=0

            for elem in d[i]:
                arr=[]
                c+=1
                arr.append(Id)
                arr.append(c)
                arr.append(l[i])
                arr.append(elem[0])
                append_list_as_row(file_name,arr)
        




def count_majority_class(Y):
    d={}
    for y in set(Y):
        d[y]=0
    for i in Y:
        d[i]+=1
    m=max(d.values())
    max_key=get_key(d,m)
    
    dd={}
    for key in d.keys():
        if key !=max_key:
            dd[key]=d[key]
            
    next_majority_class=max(dd.values())
    next_key=get_key(dd,next_majority_class)
    return max_key,m,next_key,next_majority_class,d



def get_key(val,dictionary): 
    for key, value in dictionary.items(): 
         if val == value: 
            return key 
  
    return "key doesn't exist"



def reshape_data(X,sample_size=5,ndim=1):
    c=np.empty((1,sample_size,ndim))
    for x in X:
#         print(X[2])
        l=[]
        if ndim==1:
            for i in range(len(x)):
                l.append([x[i]])
        else:
            p=[]
            for i in range(len(x)):
                p.append(x[i])
                if len(p)%ndim==0:
                    l.append(p)
                    p=[]
        c=np.append(c,np.array([l]),axis=0)
    c=np.delete(c,0,axis=0)
    return c




def resample_data(file='../../dataset_3/Temperature.csv',test_size=0.3,normalize=True):
    X,y,scalar=convert_mts_polluscope(file,z_normal=False)

    
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    
    majority_class,count,next_majority_class,next_count,d=count_majority_class(y)
    dt_under={}
    dt={}
    print(d)
    
    for key,value in d.items():
        if key not in dt.keys():
            if value>3000:
                dt_under[key]=3000
            else:
                if value<400:
                    dt[key]=400
                
                
    under = RandomUnderSampler(sampling_strategy=dt_under)
        # fit and apply the transform
    X, y = under.fit_resample(X, y)
    
    smote = SMOTE(sampling_strategy=dt)

    print('Original dataset shape', Counter(y))
    
    # fit predictor and target variable
    X, y = smote.fit_resample(X, y)    
    
    
    print('Resample dataset shape', Counter(y))
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    if normalize==True:
        X,scalar=z_normalization(X)
    else:
        scalar=None
    
    X=X.reshape(len(X),nx,ny)
    
    return X,  y, scalar
    




# #example
# file='../../dataset_3/Temperature.csv'
# data=resample_data(file=file)
# rue,bus,bureau,restaurant,domicile,velo,voiture,magasin,metro,gare,motorcycle,running=split_data_by_class(data=data)
# all_data={}
# stock_data=rue
# synth = TimeGAN(model_parameters=gan_args, hidden_dim=5, seq_len=seq_len, n_seq=n_seq, gamma=1)
# synth.train(stock_data, train_steps=300)
# synth_data = synth.sample(1000)
# print(synth_data.shape)
# nsamples, nx, ny = synth_data.shape
# synth_data= synth_data.reshape((nsamples,nx*ny))

# t=np.asarray(rue)

# nsamples, nx, ny = t.shape
# rue= t.reshape((nsamples,nx*ny))

# unscaled = data[2].inverse_transform(rue)

# unscaled= unscaled.reshape((nsamples,nx,ny))
# p=np.concatenate((all_data["Rue"],unscaled))

# all_data["Rue"]=p

# ##same for other classes


# d=[]
# d.append(all_data["Rue"])
# d.append(all_data["Bus"])
# d.append(all_data["Bureau"])
# d.append(all_data["Restaurant"])
# d.append(all_data["Domicile"])
# d.append(all_data["Velo"])
# d.append(all_data["Voiture"])
# d.append(all_data["Magasin"])
# d.append(all_data["Metro"])
# d.append(all_data["Gare"])
# d.append(all_data["Motorcycle"])
# d.append(all_data["Running"])
# d.append(all_data["Parc"])

# labels=[]
# labels.append([1]*len(all_data["Rue"]))
# labels.append([2]*len(all_data["Bus"]))
# labels.append([3]*len(all_data["Bureau"]))
# labels.append([4]*len(all_data["Restaurant"]))
# labels.append([5]*len(all_data["Domicile"]))
# labels.append([6]*len(all_data["Velo"]))
# labels.append([7]*len(all_data["Voiture"]))
# labels.append([8]*len(all_data["Magasin"]))
# labels.append([9]*len(all_data["Metro"]))
# labels.append([10]*len(all_data["Gare"]))
# labels.append([11]*len(all_data["Motorcycle"]))
# labels.append([12]*len(all_data["Running"]))
# labels.append([13]*len(all_data["Parc"]))



# store_data_after_generation('../../dataset_3/data/Temperature.csv',d,labels)




#Extract activities from the file provided by Basile
def extract_activities_RECORD(df):
    activity_time=[]
    activities=[]
    true_activity=[]
    loactions=[]
    participant=[]
    for i in range(len(df)):        
        if '02:59:00' in str(df.iloc[i]["deptime_fin"]):
            if 'Domicile' in df.iloc[i]["locname"] or 'Hébergement alternatif, hôtel' in df.iloc[i]["locname"] or 'Résidence secondaire' in df.iloc[i]["locname"]:
#                 print(df.iloc[i]["locname"])
                activity_time.append(df.iloc[i]["arrtime_fin"])
                activity_time.append(df.iloc[i]["deptime_fin"])
                activities.append(df.iloc[i]["Meaning"])
                loactions.append(df.iloc[i]["locname"])
                loactions.append(np.nan)
                participant.append(df.iloc[i]['participant_id'])
                participant.append(df.iloc[i]['participant_id'])
                true_activity.append(df.iloc[i]["category"])
                activities.append(df.shift(-1).iloc[i]["Mode"])
                true_activity.append("Domicile")
        else:
            
            activity_time.append(df.iloc[i]["arrtime_fin"])
            activity_time.append(df.iloc[i]["deptime_fin"])
            participant.append(df.iloc[i]['participant_id'])
            participant.append(df.iloc[i]['participant_id'])
            activities.append(df.iloc[i]["Meaning"])
            loactions.append(df.iloc[i]["locname"])
            loactions.append(np.nan)
            true_activity.append(df.iloc[i]["category"])
            activities.append(df.shift(-1).iloc[i]["Mode"])
            true_activity.append(df.shift(-1).iloc[i]["activity_mode"])

    df_activity = pd.DataFrame(list(zip(activity_time,participant, activities,true_activity,loactions)),
                   columns =['time','participant_virtual_id','activity','true_activity','locname'])
    return df_activity




# #extracting activities in RECORD
# ### Read Files
# tpurp=pd.read_excel('tpurp.xlsx',index_col=0)
# tpurp.rename(columns={"activity":"activity_tpurp"},inplace=True)
# mode=pd.read_excel('modes.xlsx',index_col=0)
# mode.rename(columns={"TBW value - matches modeid":"mode","Meaning":"Mode","activity":"activity_mode"},inplace=True)
# mode=mode.drop(['Unnamed: 2','As seen in TBW'],axis=1)

# #==========================================#

# #Merge files
# df_all=pd.read_excel('TBW_RECORD_Polluscope.xlsx')
# df=df_all[["participant_id","travtime","distance","actdur","tpurp","arrtime_fin","deptime_fin", "locname","route","originalmode","mode"]]
# result = pd.merge(df, tpurp, on="tpurp",how="left")
# df=result
# result = pd.merge(df, mode, on="mode",how="left")

# #==========================================#

# dfs_activities=[]
# for participant_id in participant_ids:
#     print(participant_id)
#     df=result[result['participant_id']==participant_id]
#     activities=extract_activities_RECORD(df)
#     dfs_activities.append(activities)
    
# for key,value in activities.items():
#     value.loc[value['locname'] == 'Domicile' , ['true_activity']] = 'Domicile'
#     value.loc[value['locname'] == 'Résidence secondaire', ['true_activity']] = 'Domicile'
#     value.loc[value['locname'] == 'Hébergement alternatif, hôtel', ['true_activity']] = 'Domicile'
    
# new_activities={}
# for key,value in activities.items():
#     new_activities[key]=value[['time','true_activity']].rename(columns={'true_activity':'activity'})

# for key,value in new_activities.items():
#     value.to_csv('activities/'+key+'.csv',index=False)
    




def convert_mts_polluscope(rep, z_normal = False):
    seq = np.genfromtxt(rep, delimiter=',', dtype=str, encoding="utf8")
    ids, counts = np.unique(seq[:,0], return_counts=True)
    
    No = ids.shape[0]
    D = seq.shape[1] - 3
    arr = np.asarray((ids, counts)).T
    Max_Seq_Len = np.max(arr[:,1].astype(np.int))
    
    print('Max_Seq_Len',Max_Seq_Len)
    
    out_X = np.zeros((No, Max_Seq_Len, D))
    out_Y = np.zeros((No, ))

    for idx, id in enumerate(ids):
        seq_cpy = seq[seq[:,0] == id]
        l_seq = seq_cpy.shape[0]
        out_X[idx, :l_seq, :] = seq_cpy[:, 3:]
        out_Y[idx] = seq_cpy[0, 2] 
        if z_normal: 
            out_X[idx, :l_seq, :] = z_normalization(out_X[idx, :l_seq, :])
        
    return out_X, out_Y


