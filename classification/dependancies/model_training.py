#!/usr/bin/env python


#imports
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
from sklearn.utils import shuffle

from imblearn.over_sampling import SMOTE
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from dependancies.data_preparations import *




def create_and_train_classifier_RF(x_train,y_train,x_test,y_test,name="Temperature",cross_val=False,save_model=False,filepath='./models/new_models/'):
    
    clf=RandomForestClassifier()
    clf.fit(x_train, y=y_train)
    y_test_pred = clf.predict(x_test)
    
    
    ac=accuracy_score(y_true=y_test, y_pred=y_test_pred)
    print("the accuracy score of the testing data is : " + str(ac))
    
    if save_model==True:
        filename = filepath+str(name)+'_model.sav'
        pickle.dump(clf, open(filename, 'wb'))
    
    
    if cross_val==True:
        X=np.append(x_train,x_test,axis=0)
        y=np.append(y_train,y_test,axis=0)
        result=model_selection.cross_val_score(clf,X,y,scoring='accuracy')
    else:
        result=None
    
    
    return clf,ac,result




def create_and_train_classifier_KNN(x_train,y_train,x_test,y_test,name="Temperature",cross_val=False,save_model=False,filepath='./models/new_models_/',n_neighbors=2):
    
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=n_neighbors, metric="dtw",n_jobs=-1)
    clf.fit(x_train, y=y_train)
    
    y_test_pred = clf.predict(x_test)
    
    
    ac=accuracy_score(y_true=y_test, y_pred=y_test_pred)
    print("the accuracy score of the testing data is : " + str(ac))
    
    if save_model==True:
        filename = filepath+str(name)+'_model.sav'
        pickle.dump(clf, open(filename, 'wb'))
    
    
    if cross_val==True:
        X=np.append(x_train,x_test,axis=0)
        y=np.append(y_train,y_test,axis=0)
        result=model_selection.cross_val_score(clf,X,y,scoring='accuracy')
    else:
        result=None
    
    
    return clf,ac,result




def read_data(file_path="./models/models_with_speed_RF_4/test_2.csv",test_size=0.3,show=True,campaign='RECORD'):    
    X,y=convert_mts_polluscope(file_path,z_normal=False)
    
    X,y=shuffle(X,y)   
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#     # summarize distribution
    counter = Counter(y_train)
    for k,v in counter.items():
        per = v / len(y_train) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    if show==True:
        # plot the distribution
        pyplot.bar(counter.keys(), counter.values())
        pyplot.xticks(rotation="vertical")
        pyplot.show()
#     # summarize distribution
    counter = Counter(y_test)
    for k,v in counter.items():
        per = v / len(y_test) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    if show==True:
        # plot the distribution
        pyplot.bar(counter.keys(), counter.values())
        pyplot.show()

    counter = Counter(y)
    if campaign == 'RECORD':
        dictionary={
        'Walk':1,
        'Bus': 2,
         'Office': 3,
         'Restaurant': 4,
         'Home': 5,
         'Bike': 6,
         'Car': 7,
         'Store': 8,
         'Metro': 9,    
         'Station': 10,
         'Motorcycle': 11,
         'Running': 12,
         'Train':16,
         'Indoor':15,
         'Cinéma': 14,
         'Parc':13,

        }
    else:
        if campaign == 'VGP':
            dictionary={
            'Rue':1,
            'Bus': 2,
             'Office': 3,
             'Restaurant': 4,
             'Home': 5,
             'Car': 6,
             'Store': 7,
             'Train': 8
            }
        
    keys=[]
    for key in counter.keys():
        keys.append(get_key(my_dict=dictionary,val=key))
    
    fig = plt.figure(figsize=(16,8))    
    ax = fig.add_subplot(111)
    
    if show==True:
        # plot the distribution
        pyplot.bar(keys, counter.values())
        pyplot.show()
    
    return X_train,  y_train, X_test,y_test



# ###Example
# names=[] #model names
# results=[] #cross validation results for each model
# file_path='./models/'
# train_test_file_path='./dataset_3/data/'
# filename=train_test_file_path+"Temperature.csv"

# x_train_Temperature,y_train_Temperature,x_test_Temperature,y_test_Temperature=read_data(file_path=filename,show=True)

# #KNN-DTW
# clf_Temperature,accuracy_Temperature,result_Temperature=create_and_train_classifier_KNN(name="Temperature",x_train=x_train_Temperature,y_train=y_train_Temperature,x_test=x_test_Temperature,y_test=y_test_Temperature,cross_val=True,save_model=True,filepath=file_path)

# #RF
# clf_Temperature,accuracy_Temperature,result_Temperature=create_and_train_classifier_RF(name="Temperature",x_train=x_train_Temperature,y_train=y_train_Temperature,x_test=x_test_Temperature,y_test=y_test_Temperature,cross_val=True,save_model=True,filepath=file_path)

# names.append("Temperature\nView")
# results.append(result_Temperature)



#We use this function when we have all the data in the same file
def get_data_from_all_dimensions(file_path):
    X,y=convert_mts_polluscope(file_path)
    X,y=shuffle(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.33)
    
    x_train_Temperature=[]
    x_train_Humidity=[]
    x_train_NO2=[]
    x_train_BC=[]
    x_train_PM1=[]
    x_train_PM25=[]
    x_train_PM10=[]
    x_train_Speed=[]
    
    y_train_Temperature=y_train
    y_train_Humidity=y_train
    y_train_NO2=y_train
    y_train_BC=y_train
    y_train_PM1=y_train
    y_train_PM25=y_train
    y_train_PM10=y_train
    y_train_Speed=y_train
    
    x_test_Temperature=[]
    x_test_Humidity=[]
    x_test_NO2=[]
    x_test_BC=[]
    x_test_PM1=[]
    x_test_PM25=[]
    x_test_PM10=[]
    x_test_Speed=[]
    
    y_test_Temperature=y_test
    y_test_Humidity=y_test
    y_test_NO2=y_test
    y_test_BC=y_test
    y_test_PM1=y_test
    y_test_PM25=y_test
    y_test_PM10=y_test
    y_test_Speed=y_test
    
    for x in X_train:
        temperature=[]
        humidity=[]
        no2=[]
        bc=[]
        pm1=[]
        pm25=[]
        pm10=[]
        speed=[]
        for i in range(len(x)):
            temperature.append(x[i][0])
            humidity.append(x[i][1])
            no2.append(x[i][2])
            bc.append(x[i][3])
            pm1.append(x[i][4])
            pm25.append(x[i][5])
            pm10.append(x[i][6])
            speed.append(x[i][7])
        
        x_train_Temperature.append(temperature)
        x_train_Humidity.append(humidity)
        x_train_NO2.append(no2)
        x_train_BC.append(bc)
        x_train_PM1.append(pm1)
        x_train_PM25.append(pm25)
        x_train_PM10.append(pm10)
        x_train_Speed.append(speed)
        
    for x in X_test:
        temperature=[]
        humidity=[]
        no2=[]
        bc=[]
        pm1=[]
        pm25=[]
        pm10=[]
        speed=[]
        for i in range(len(x)):
            temperature.append(x[i][0])
            humidity.append(x[i][1])
            no2.append(x[i][2])
            bc.append(x[i][3])
            pm1.append(x[i][4])
            pm25.append(x[i][5])
            pm10.append(x[i][6])
            speed.append(x[i][7])
        
        x_test_Temperature.append(temperature)
        x_test_Humidity.append(humidity)
        x_test_NO2.append(no2)
        x_test_BC.append(bc)
        x_test_PM1.append(pm1)
        x_test_PM25.append(pm25)
        x_test_PM10.append(pm10)
        x_test_Speed.append(speed)
        
        

    return x_train_Temperature,y_train_Temperature,x_train_Humidity,y_train_Humidity,x_train_NO2,y_train_NO2,x_train_BC,y_train_BC,x_train_PM1,y_train_PM1,x_train_PM25,y_train_PM25,x_train_PM10,y_train_PM10,x_train_Speed,y_train_Speed,x_test_Temperature,y_test_Temperature,x_test_Humidity,y_test_Humidity,x_test_NO2,y_test_NO2,x_test_BC,y_test_BC,x_test_PM1,y_test_PM1,x_test_PM25,y_test_PM25,x_test_PM10,y_test_PM10,x_test_Speed,y_test_Speed
    
    




#Read the models using pickle library
#models=[Temperature_model,Humidity_model,NO2_model,BC_model,PM1_model,PM25_model,PM10_model,Speed_model]



#Uses the trained models in order to predict the class and the probability to be able to generate the new data set D' that trains the meta-learner
def prepare_D_prime_dataset(classifiers=[],files=[],extarct_from_one_file=False):
    clf_Temperature=classifiers[0]
    clf_Humidity=classifiers[1]
    clf_NO2=classifiers[2]
    clf_BC=classifiers[3]
    clf_PM1=classifiers[4]
    clf_PM25=classifiers[5]
    clf_PM10=classifiers[6]
    clf_Speed=classifiers[7]
    
    x_train_Temperature,y_train_Temperature,x_train_Humidity,y_train_Humidity,x_train_NO2,y_train_NO2,x_train_BC,y_train_BC,x_train_PM1,y_train_PM1,x_train_PM25,y_train_PM25,x_train_PM10,y_train_PM10,x_train_Speed,y_train_Speed,x_test_Temperature,y_test_Temperature,x_test_Humidity,y_test_Humidity,x_test_NO2,y_test_NO2,x_test_BC,y_test_BC,x_test_PM1,y_test_PM1,x_test_PM25,y_test_PM25,x_test_PM10,y_test_PM10,x_test_Speed,y_test_Speed=get_data_from_all_dimensions(files[0])
    
    if np.array_equal(y_train_Temperature , y_train_Humidity) and np.array_equal(y_train_Temperature , y_train_NO2) and np.array_equal(y_train_Temperature , y_train_BC) and np.array_equal(y_train_Temperature , y_train_PM1) and np.array_equal(y_train_Temperature , y_train_PM25) and np.array_equal(y_train_Temperature , y_train_PM10):# and np.array_equal(y_train_Temperature , y_train_Speed):
        print("Every thing is okay in train")
    else:
        print("Error in train")
        return
    
    if np.array_equal(y_test_Temperature , y_test_Humidity) and np.array_equal(y_test_Temperature , y_test_NO2) and np.array_equal(y_test_Temperature , y_test_BC) and np.array_equal(y_test_Temperature , y_test_PM1) and np.array_equal(y_test_Temperature , y_test_PM25) and np.array_equal(y_test_Temperature , y_test_PM10):# and np.array_equal(y_test_Temperature , y_test_Speed):
        print("Every thing is okay in test")
    else:
        print("Error in test")
        return
    
#     print(x_train_Temperature.shape)
    predicted_train_Temperature=clf_Temperature.predict(x_train_Temperature)
    print("Temperature train predicited")
    predicted_train_Humidity=clf_Humidity.predict(x_train_Humidity)
    print("Humidity train predicited")
    predicted_train_NO2=clf_NO2.predict(x_train_NO2)
    print("NO2 train predicited")
    predicted_train_BC=clf_BC.predict(x_train_BC)
    print("BC train predicited")
    predicted_train_PM1=clf_PM1.predict(x_train_PM1)
    print("PM1.0 train predicited")
    predicted_train_PM25=clf_PM1.predict(x_train_PM25)
    print("PM2.5 train predicited")
    predicted_train_PM10=clf_PM10.predict(x_train_PM10)
    print("PM10 train predicited")
#     predicted_train_Speed=clf_Speed.predict(x_train_Speed)
#     print("Speed train predicited")
    
    predicted_prob_train_Temperature=clf_Temperature.predict_proba(x_train_Temperature)
    print("Temperature train predicited prob")
    predicted_prob_train_Humidity=clf_Humidity.predict_proba(x_train_Humidity)
    print("Humidity train predicited prob")
    predicted_prob_train_NO2=clf_NO2.predict_proba(x_train_NO2)
    print("NO2 train predicited prob")
    predicted_prob_train_BC=clf_BC.predict_proba(x_train_BC)
    print("BC train predicited prob")
    predicted_prob_train_PM1=clf_PM1.predict_proba(x_train_PM1)
    print("PM1.0 train predicited prob")
    predicted_prob_train_PM25=clf_PM1.predict_proba(x_train_PM25)
    print("PM2.5 train predicited prob")
    predicted_prob_train_PM10=clf_PM10.predict_proba(x_train_PM10)
    print("PM10 train predicited prob")
#     predicted_prob_train_Speed=clf_Speed.predict_proba(x_train_Speed)
#     print("Speed train predicited prob")
    
    
    predicted_test_Temperature=clf_Temperature.predict(x_test_Temperature)
    print("Temperature test predicited")
    predicted_test_Humidity=clf_Humidity.predict(x_test_Humidity)
    print("Humidity test predicited")
    predicted_test_NO2=clf_NO2.predict(x_test_NO2)
    print("NO2 test predicited")
    predicted_test_BC=clf_BC.predict(x_test_BC)
    print("BC test predicited")
    predicted_test_PM1=clf_PM1.predict(x_test_PM1)
    print("PM1.0 test predicited")
    predicted_test_PM25=clf_PM1.predict(x_test_PM25)
    print("PM2.5 test predicited")
    predicted_test_PM10=clf_PM10.predict(x_test_PM10)
    print("PM10 test predicited")
#     predicted_test_Speed=clf_Speed.predict(x_test_Speed)
#     print("Speed test predicited")
    
    predicted_prob_test_Temperature=clf_Temperature.predict_proba(x_test_Temperature)
    print("Temperature test predicited prob")
    predicted_prob_test_Humidity=clf_Humidity.predict_proba(x_test_Humidity)
    print("Humidity test predicited prob")
    predicted_prob_test_NO2=clf_NO2.predict_proba(x_test_NO2)
    print("NO2 test predicited prob")
    predicted_prob_test_BC=clf_BC.predict_proba(x_test_BC)
    print("BC test predicited prob")
    predicted_prob_test_PM1=clf_PM1.predict_proba(x_test_PM1)
    print("PM1.0 test predicited prob")
    predicted_prob_test_PM25=clf_PM1.predict_proba(x_test_PM25)
    print("PM2.5 test predicited prob")
    predicted_prob_test_PM10=clf_PM10.predict_proba(x_test_PM10)
    print("PM10 test predicited prob")
#     predicted_prob_test_Speed=clf_Speed.predict_proba(x_test_Speed)
#     print("Speed test predicited prob")
    
#     return predicted_train_Temperature,predicted_prob_train_Temperature,predicted_train_Humidity,predicted_prob_train_Humidity,predicted_train_NO2,predicted_prob_train_NO2,predicted_train_BC,predicted_prob_train_BC,predicted_train_PM1,predicted_prob_train_PM1,predicted_train_PM25,predicted_prob_train_PM25,predicted_train_PM10,predicted_prob_train_PM10,predicted_test_Temperature,predicted_prob_test_Temperature,predicted_test_Humidity,predicted_prob_test_Humidity,predicted_test_NO2,predicted_prob_test_NO2,predicted_test_BC,predicted_prob_test_BC,predicted_test_PM1,predicted_prob_test_PM1,predicted_test_PM25,predicted_prob_test_PM25,predicted_test_PM10,predicted_prob_test_PM10
    
    train_data_RF=[]
    for i in range(len(predicted_prob_train_Temperature)):
        a=[]
        a.append(predicted_train_Temperature[i])
        a.append(predicted_train_Humidity[i])
        a.append(predicted_train_NO2[i])
        a.append(predicted_train_BC[i])
        a.append(predicted_train_PM1[i])
        a.append(predicted_train_PM25[i])
        a.append(predicted_train_PM10[i])
#         a.append(predicted_train_Speed[i])
        a.append(predicted_prob_train_Temperature[i][int(predicted_train_Temperature[i])-1])
        a.append(predicted_prob_train_Humidity[i][int(predicted_train_Humidity[i])-1])
        a.append(predicted_prob_train_NO2[i][int(predicted_train_NO2[i])-1])
        a.append(predicted_prob_train_BC[i][int(predicted_train_BC[i])-1])
        a.append(predicted_prob_train_PM1[i][int(predicted_train_PM1[i])-1])
        a.append(predicted_prob_train_PM25[i][int(predicted_train_PM25[i])-1])
        a.append(predicted_prob_train_PM10[i][int(predicted_train_PM10[i])-1])    
#         a.append(predicted_prob_train_Speed[i][int(predicted_train_Speed[i])-1])    
        train_data_RF.append(a)
        
    #All data
    x_train_data_RF=train_data_RF
    y_train_data_RF=y_train_Temperature
    
    test_data_RF=[]
    for i in range(len(predicted_prob_test_Temperature)):
        a=[]
        a.append(predicted_test_Temperature[i])
        a.append(predicted_test_Humidity[i])
        a.append(predicted_test_NO2[i])
        a.append(predicted_test_BC[i])
        a.append(predicted_test_PM1[i])
        a.append(predicted_test_PM25[i])
        a.append(predicted_test_PM10[i])
#         a.append(predicted_test_Speed[i])
        a.append(predicted_prob_test_Temperature[i][int(predicted_test_Temperature[i])-1])
        a.append(predicted_prob_test_Humidity[i][int(predicted_test_Humidity[i])-1])
        a.append(predicted_prob_test_NO2[i][int(predicted_test_NO2[i])-1])
        a.append(predicted_prob_test_BC[i][int(predicted_test_BC[i])-1])
        a.append(predicted_prob_test_PM1[i][int(predicted_test_PM1[i])-1])
        a.append(predicted_prob_test_PM25[i][int(predicted_test_PM25[i])-1])
        a.append(predicted_prob_test_PM10[i][int(predicted_test_PM10[i])-1])    
#         a.append(predicted_prob_test_Speed[i][int(predicted_test_Speed[i])-1])    
        test_data_RF.append(a)
    
    x_test_data_RF=test_data_RF
    y_test_data_RF=y_test_Temperature
    
    #Data without BC
    train_data_RF_without_BC=[]
    for i in range(len(predicted_prob_train_Temperature)):
        a=[]
        a.append(predicted_train_Temperature[i])
        a.append(predicted_train_Humidity[i])
        a.append(predicted_train_NO2[i])
        a.append(predicted_train_PM1[i])
        a.append(predicted_train_PM25[i])
        a.append(predicted_train_PM10[i])
        a.append(predicted_prob_train_Temperature[i][int(predicted_train_Temperature[i])-1])
        a.append(predicted_prob_train_Humidity[i][int(predicted_train_Humidity[i])-1])
        a.append(predicted_prob_train_NO2[i][int(predicted_train_NO2[i])-1])
        a.append(predicted_prob_train_PM1[i][int(predicted_train_PM1[i])-1])
        a.append(predicted_prob_train_PM25[i][int(predicted_train_PM25[i])-1])
        a.append(predicted_prob_train_PM10[i][int(predicted_train_PM10[i])-1])    
        train_data_RF_without_BC.append(a)
    
    x_train_data_RF_without_BC=train_data_RF_without_BC
    y_train_data_RF_without_BC=y_train_Temperature
    
    test_data_RF_without_BC=[]
    for i in range(len(predicted_prob_test_Temperature)):
        a=[]
        a.append(predicted_test_Temperature[i])
        a.append(predicted_test_Humidity[i])
        a.append(predicted_test_NO2[i])
        a.append(predicted_test_PM1[i])
        a.append(predicted_test_PM25[i])
        a.append(predicted_test_PM10[i])
        a.append(predicted_prob_test_Temperature[i][int(predicted_test_Temperature[i])-1])
        a.append(predicted_prob_test_Humidity[i][int(predicted_test_Humidity[i])-1])
        a.append(predicted_prob_test_NO2[i][int(predicted_test_NO2[i])-1])
        a.append(predicted_prob_test_PM1[i][int(predicted_test_PM1[i])-1])
        a.append(predicted_prob_test_PM25[i][int(predicted_test_PM25[i])-1])
        a.append(predicted_prob_test_PM10[i][int(predicted_test_PM10[i])-1])    
        test_data_RF_without_BC.append(a)
        
    x_test_data_RF_without_BC=test_data_RF_without_BC
    y_test_data_RF_without_BC=y_test_Temperature
    
    #Data without NO2 and BC
    train_data_RF_without_BC_NO2=[]
    for i in range(len(predicted_prob_train_Temperature)):
        a=[]
        a.append(predicted_train_Temperature[i])
        a.append(predicted_train_Humidity[i])
        a.append(predicted_train_PM1[i])
        a.append(predicted_train_PM25[i])
        a.append(predicted_train_PM10[i])
        a.append(predicted_prob_train_Temperature[i][int(predicted_train_Temperature[i])-1])
        a.append(predicted_prob_train_Humidity[i][int(predicted_train_Humidity[i])-1])
        a.append(predicted_prob_train_PM1[i][int(predicted_train_PM1[i])-1])
        a.append(predicted_prob_train_PM25[i][int(predicted_train_PM25[i])-1])
        a.append(predicted_prob_train_PM10[i][int(predicted_train_PM10[i])-1])    
        train_data_RF_without_BC_NO2.append(a)
    x_train_data_RF_without_BC_NO2=train_data_RF_without_BC_NO2
    y_train_data_RF_without_BC_NO2=y_train_Temperature
    
    test_data_RF_without_BC_NO2=[]
    for i in range(len(predicted_prob_test_Temperature)):
        a=[]
        a.append(predicted_test_Temperature[i])
        a.append(predicted_test_Humidity[i])
        a.append(predicted_test_PM1[i])
        a.append(predicted_test_PM25[i])
        a.append(predicted_test_PM10[i])
        a.append(predicted_prob_test_Temperature[i][int(predicted_test_Temperature[i])-1])
        a.append(predicted_prob_test_Humidity[i][int(predicted_test_Humidity[i])-1])
        a.append(predicted_prob_test_PM1[i][int(predicted_test_PM1[i])-1])
        a.append(predicted_prob_test_PM25[i][int(predicted_test_PM25[i])-1])
        a.append(predicted_prob_test_PM10[i][int(predicted_test_PM10[i])-1])    
        test_data_RF_without_BC_NO2.append(a)
    
    x_test_data_RF_without_BC_NO2=test_data_RF_without_BC_NO2
    y_test_data_RF_without_BC_NO2=y_test_Temperature
    
    #Data without PMS and BC
    train_data_RF_without_BC_PMS=[]
    for i in range(len(predicted_prob_train_Temperature)):
        a=[]
        a.append(predicted_train_Temperature[i])
        a.append(predicted_train_Humidity[i])
        a.append(predicted_train_NO2[i])
        a.append(predicted_prob_train_Temperature[i][int(predicted_train_Temperature[i])-1])
        a.append(predicted_prob_train_Humidity[i][int(predicted_train_Humidity[i])-1])
        a.append(predicted_prob_train_NO2[i][int(predicted_train_NO2[i])-1])    
        train_data_RF_without_BC_PMS.append(a)
    x_train_data_RF_without_BC_PMS=train_data_RF_without_BC_PMS
    y_train_data_RF_without_BC_PMS=y_train_Temperature
    
    test_data_RF_without_BC_PMS=[]
    for i in range(len(predicted_prob_test_Temperature)):
        a=[]
        a.append(predicted_test_Temperature[i])
        a.append(predicted_test_Humidity[i])
        a.append(predicted_test_NO2[i])
        a.append(predicted_prob_test_Temperature[i][int(predicted_test_Temperature[i])-1])
        a.append(predicted_prob_test_Humidity[i][int(predicted_test_Humidity[i])-1])
        a.append(predicted_prob_test_NO2[i][int(predicted_test_NO2[i])-1])
        test_data_RF_without_BC_PMS.append(a)
    
    x_test_data_RF_without_BC_PMS=test_data_RF_without_BC_PMS
    y_test_data_RF_without_BC_PMS=y_test_Temperature

    #Data without NO2
    train_data_RF_without_NO2=[]
    for i in range(len(predicted_prob_train_Temperature)):
        a=[]
        a.append(predicted_train_Temperature[i])
        a.append(predicted_train_Humidity[i])
        a.append(predicted_train_BC[i])
        a.append(predicted_train_PM1[i])
        a.append(predicted_train_PM25[i])
        a.append(predicted_train_PM10[i])
        a.append(predicted_prob_train_Temperature[i][int(predicted_train_Temperature[i])-1])
        a.append(predicted_prob_train_Humidity[i][int(predicted_train_Humidity[i])-1])
        a.append(predicted_prob_train_BC[i][int(predicted_train_BC[i])-1])
        a.append(predicted_prob_train_PM1[i][int(predicted_train_PM1[i])-1])
        a.append(predicted_prob_train_PM25[i][int(predicted_train_PM25[i])-1])
        a.append(predicted_prob_train_PM10[i][int(predicted_train_PM10[i])-1])    
        train_data_RF_without_NO2.append(a)
        
    x_train_data_RF_without_NO2=train_data_RF_without_NO2
    y_train_data_RF_without_NO2=y_train_Temperature
    
    test_data_RF_without_NO2=[]
    for i in range(len(predicted_prob_test_Temperature)):
        a=[]
        a.append(predicted_test_Temperature[i])
        a.append(predicted_test_Humidity[i])
        a.append(predicted_test_BC[i])
        a.append(predicted_test_PM1[i])
        a.append(predicted_test_PM25[i])
        a.append(predicted_test_PM10[i])
        a.append(predicted_prob_test_Temperature[i][int(predicted_test_Temperature[i])-1])
        a.append(predicted_prob_test_Humidity[i][int(predicted_test_Humidity[i])-1])
        a.append(predicted_prob_test_BC[i][int(predicted_test_BC[i])-1])
        a.append(predicted_prob_test_PM1[i][int(predicted_test_PM1[i])-1])
        a.append(predicted_prob_test_PM25[i][int(predicted_test_PM25[i])-1])
        a.append(predicted_prob_test_PM10[i][int(predicted_test_PM10[i])-1])    
        test_data_RF_without_NO2.append(a)
        
    x_test_data_RF_without_NO2=test_data_RF_without_NO2
    y_test_data_RF_without_NO2=y_test_Temperature
    
    #Data without PMS
    train_data_RF_without_PMS=[]
    for i in range(len(predicted_prob_train_Temperature)):
        a=[]
        a.append(predicted_train_Temperature[i])
        a.append(predicted_train_Humidity[i])
        a.append(predicted_train_NO2[i])
        a.append(predicted_train_BC[i])
        a.append(predicted_prob_train_Temperature[i][int(predicted_train_Temperature[i])-1])
        a.append(predicted_prob_train_Humidity[i][int(predicted_train_Humidity[i])-1])
        a.append(predicted_prob_train_NO2[i][int(predicted_train_NO2[i])-1])
        a.append(predicted_prob_train_BC[i][int(predicted_train_BC[i])-1])
        train_data_RF_without_PMS.append(a)
    
    x_train_data_RF_without_PMS=train_data_RF_without_PMS
    y_train_data_RF_without_PMS=y_train_Temperature
    
    test_data_RF_without_PMS=[]
    for i in range(len(predicted_prob_test_Temperature)):
        a=[]
        a.append(predicted_test_Temperature[i])
        a.append(predicted_test_Humidity[i])
        a.append(predicted_test_NO2[i])
        a.append(predicted_test_BC[i])
        a.append(predicted_prob_test_Temperature[i][int(predicted_test_Temperature[i])-1])
        a.append(predicted_prob_test_Humidity[i][int(predicted_test_Humidity[i])-1])
        a.append(predicted_prob_test_NO2[i][int(predicted_test_NO2[i])-1])
        a.append(predicted_prob_test_BC[i][int(predicted_test_BC[i])-1])
        test_data_RF_without_PMS.append(a)
        
    x_test_data_RF_without_PMS=test_data_RF_without_PMS
    y_test_data_RF_without_PMS=y_test_Temperature
    
    #Data containing only Temperature and Humidity
    train_data_RF_only_Temperature_Humidity=[]
    for i in range(len(predicted_prob_train_Temperature)):
        a=[]
        a.append(predicted_train_Temperature[i])
        a.append(predicted_train_Humidity[i])
        a.append(predicted_prob_train_Temperature[i][int(predicted_train_Temperature[i])-1])
        a.append(predicted_prob_train_Humidity[i][int(predicted_train_Humidity[i])-1])
        train_data_RF_only_Temperature_Humidity.append(a)
    
    x_train_data_RF_only_Temperature_Humidity=train_data_RF_only_Temperature_Humidity
    y_train_data_RF_only_Temperature_Humidity=y_train_Temperature
    
    test_data_RF_only_Temperature_Humidity=[]
    for i in range(len(predicted_prob_test_Temperature)):
        a=[]
        a.append(predicted_test_Temperature[i])
        a.append(predicted_test_Humidity[i])
        a.append(predicted_prob_test_Temperature[i][int(predicted_test_Temperature[i])-1])
        a.append(predicted_prob_test_Humidity[i][int(predicted_test_Humidity[i])-1])
        test_data_RF_only_Temperature_Humidity.append(a)
    
    x_test_data_RF_only_Temperature_Humidity=test_data_RF_only_Temperature_Humidity
    y_test_data_RF_only_Temperature_Humidity=y_test_Temperature
    
    return x_train_data_RF,y_train_data_RF,x_test_data_RF,y_test_data_RF,x_train_data_RF_without_BC,y_train_data_RF_without_BC,x_test_data_RF_without_BC,y_test_data_RF_without_BC,x_train_data_RF_without_BC_NO2,y_train_data_RF_without_BC_NO2,x_test_data_RF_without_BC_NO2,y_test_data_RF_without_BC_NO2,x_train_data_RF_without_NO2,y_train_data_RF_without_NO2,x_test_data_RF_without_NO2,y_test_data_RF_without_NO2,x_train_data_RF_without_PMS,y_train_data_RF_without_PMS,x_test_data_RF_without_PMS,y_test_data_RF_without_PMS,x_train_data_RF_only_Temperature_Humidity,y_train_data_RF_only_Temperature_Humidity,x_test_data_RF_only_Temperature_Humidity,y_test_data_RF_only_Temperature_Humidity,x_train_data_RF_without_BC_PMS,y_train_data_RF_without_BC_PMS,x_test_data_RF_without_BC_PMS,y_test_data_RF_without_BC_PMS,x_train_Temperature,y_train_Temperature,x_train_Humidity,y_train_Humidity,x_train_NO2,y_train_NO2,x_train_BC,y_train_BC,x_train_PM1,y_train_PM1,x_train_PM25,y_train_PM25,x_train_PM10,y_train_PM10,x_train_Speed,y_train_Speed,x_test_Temperature,y_test_Temperature,x_test_Humidity,y_test_Humidity,x_test_NO2,y_test_NO2,x_test_BC,y_test_BC,x_test_PM1,y_test_PM1,x_test_PM25,y_test_PM25,x_test_PM10,y_test_PM10,x_test_Speed,y_test_Speed




#Performs the same thing as the previous function but this time with speed dimension it takes an input the x of each dimension from the previous function
def prepare_D_prime_dataset_speed(x_train_Temperature,y_train_Temperature,x_train_Humidity,y_train_Humidity,x_train_NO2,y_train_NO2,x_train_BC,y_train_BC,x_train_PM1,y_train_PM1,x_train_PM25,y_train_PM25,x_train_PM10,y_train_PM10,x_train_Speed,y_train_Speed,x_test_Temperature,y_test_Temperature,x_test_Humidity,y_test_Humidity,x_test_NO2,y_test_NO2,x_test_BC,y_test_BC,x_test_PM1,y_test_PM1,x_test_PM25,y_test_PM25,x_test_PM10,y_test_PM10,x_test_Speed,y_test_Speed,classifiers=[],files=[],extarct_from_one_file=False,):    
    clf_Temperature=classifiers[0]
    clf_Humidity=classifiers[1]
    clf_NO2=classifiers[2]
    clf_BC=classifiers[3]
    clf_PM1=classifiers[4]
    clf_PM25=classifiers[5]
    clf_PM10=classifiers[6]
    clf_Speed=classifiers[7]
    
    predicted_train_Temperature=[]
    predicted_train_Humidity=[]
    predicted_train_NO2=[]
    predicted_train_BC=[]
    predicted_train_PM1=[]
    predicted_train_PM10=[]
    predicted_train_PM25=[]
    
    predicted_prob_train_Temperature=[]
    predicted_prob_train_Humidity=[]
    predicted_prob_train_NO2=[]
    predicted_prob_train_BC=[]
    predicted_prob_train_PM1=[]
    predicted_prob_train_PM25=[]
    predicted_prob_train_PM10=[]
    
    predicted_test_Temperature=[]
    predicted_test_Humidity=[]
    predicted_test_NO2=[]
    predicted_test_BC=[]
    predicted_test_PM1=[]
    predicted_test_PM10=[]
    predicted_test_PM25=[]
    
    predicted_prob_test_Temperature=[]
    predicted_prob_test_Humidity=[]
    predicted_prob_test_NO2=[]
    predicted_prob_test_BC=[]
    predicted_prob_test_PM1=[]
    predicted_prob_test_PM25=[]
    predicted_prob_test_PM10=[]
    
    
    
    if np.array_equal(y_train_Temperature , y_train_Humidity) and np.array_equal(y_train_Temperature , y_train_NO2) and np.array_equal(y_train_Temperature , y_train_BC) and np.array_equal(y_train_Temperature , y_train_PM1) and np.array_equal(y_train_Temperature , y_train_PM25) and np.array_equal(y_train_Temperature , y_train_PM10) and np.array_equal(y_train_Temperature , y_train_Speed):
        print("Every thing is okay in train")
    else:
        print("Error in train")
        return
    
    if np.array_equal(y_test_Temperature , y_test_Humidity) and np.array_equal(y_test_Temperature , y_test_NO2) and np.array_equal(y_test_Temperature , y_test_BC) and np.array_equal(y_test_Temperature , y_test_PM1) and np.array_equal(y_test_Temperature , y_test_PM25) and np.array_equal(y_test_Temperature , y_test_PM10) and np.array_equal(y_test_Temperature , y_test_Speed):
        print("Every thing is okay in test")
    else:
        print("Error in test")
        return
    
    
    predicted_train_Temperature=clf_Temperature.predict(x_train_Temperature)
    print("Temperature train predicited")
    predicted_train_Humidity=clf_Humidity.predict(x_train_Humidity)
    print("Humidity train predicited")
    predicted_train_NO2=clf_NO2.predict(x_train_NO2)
    print("NO2 train predicited")
    predicted_train_BC=clf_BC.predict(x_train_BC)
    print("BC train predicited")
    predicted_train_PM1=clf_PM1.predict(x_train_PM1)
    print("PM1.0 train predicited")
    predicted_train_PM25=clf_PM1.predict(x_train_PM25)
    print("PM2.5 train predicited")
    predicted_train_PM10=clf_PM10.predict(x_train_PM10)
    print("PM10 train predicited")
    predicted_train_Speed=clf_Speed.predict(x_train_Speed)
    print("Speed train predicited")
    
    predicted_prob_train_Temperature=clf_Temperature.predict_proba(x_train_Temperature)
    print("Temperature train predicited prob")
    predicted_prob_train_Humidity=clf_Humidity.predict_proba(x_train_Humidity)
    print("Humidity train predicited prob")
    predicted_prob_train_NO2=clf_NO2.predict_proba(x_train_NO2)
    print("NO2 train predicited prob")
    predicted_prob_train_BC=clf_BC.predict_proba(x_train_BC)
    print("BC train predicited prob")
    predicted_prob_train_PM1=clf_PM1.predict_proba(x_train_PM1)
    print("PM1.0 train predicited prob")
    predicted_prob_train_PM25=clf_PM1.predict_proba(x_train_PM25)
    print("PM2.5 train predicited prob")
    predicted_prob_train_PM10=clf_PM10.predict_proba(x_train_PM10)
    print("PM10 train predicited prob")
    predicted_prob_train_Speed=clf_Speed.predict_proba(x_train_Speed)
    print("Speed train predicited prob")
    
    
    predicted_test_Temperature=clf_Temperature.predict(x_test_Temperature)
    print("Temperature test predicited")
    predicted_test_Humidity=clf_Humidity.predict(x_test_Humidity)
    print("Humidity test predicited")
    predicted_test_NO2=clf_NO2.predict(x_test_NO2)
    print("NO2 test predicited")
    predicted_test_BC=clf_BC.predict(x_test_BC)
    print("BC test predicited")
    predicted_test_PM1=clf_PM1.predict(x_test_PM1)
    print("PM1.0 test predicited")
    predicted_test_PM25=clf_PM1.predict(x_test_PM25)
    print("PM2.5 test predicited")
    predicted_test_PM10=clf_PM10.predict(x_test_PM10)
    print("PM10 test predicited")
    predicted_test_Speed=clf_Speed.predict(x_test_Speed)
    print("Speed test predicited")
    
    predicted_prob_test_Temperature=clf_Temperature.predict_proba(x_test_Temperature)
    print("Temperature test predicited prob")
    predicted_prob_test_Humidity=clf_Humidity.predict_proba(x_test_Humidity)
    print("Humidity test predicited prob")
    predicted_prob_test_NO2=clf_NO2.predict_proba(x_test_NO2)
    print("NO2 test predicited prob")
    predicted_prob_test_BC=clf_BC.predict_proba(x_test_BC)
    print("BC test predicited prob")
    predicted_prob_test_PM1=clf_PM1.predict_proba(x_test_PM1)
    print("PM1.0 test predicited prob")
    predicted_prob_test_PM25=clf_PM1.predict_proba(x_test_PM25)
    print("PM2.5 test predicited prob")
    predicted_prob_test_PM10=clf_PM10.predict_proba(x_test_PM10)
    print("PM10 test predicited prob")
    predicted_prob_test_Speed=clf_Speed.predict_proba(x_test_Speed)
    print("Speed test predicited prob")
    
    
    
    train_data_RF=[]
    for i in range(len(predicted_prob_train_Temperature)):
        a=[]
        a.append(predicted_train_Temperature[i])
        a.append(predicted_train_Humidity[i])
        a.append(predicted_train_NO2[i])
        a.append(predicted_train_BC[i])
        a.append(predicted_train_PM1[i])
        a.append(predicted_train_PM25[i])
        a.append(predicted_train_PM10[i])
        a.append(predicted_train_Speed[i])
        a.append(predicted_prob_train_Temperature[i][int(predicted_train_Temperature[i])-1])
        a.append(predicted_prob_train_Humidity[i][int(predicted_train_Humidity[i])-1])
        a.append(predicted_prob_train_NO2[i][int(predicted_train_NO2[i])-1])
        a.append(predicted_prob_train_BC[i][int(predicted_train_BC[i])-1])
        a.append(predicted_prob_train_PM1[i][int(predicted_train_PM1[i])-1])
        a.append(predicted_prob_train_PM25[i][int(predicted_train_PM25[i])-1])
        a.append(predicted_prob_train_PM10[i][int(predicted_train_PM10[i])-1])    
        a.append(predicted_prob_train_Speed[i][int(predicted_train_Speed[i])-1])
        
        
        train_data_RF.append(a)
        
    #All data
    x_train_data_RF=train_data_RF
    y_train_data_RF=y_train_Temperature
    
    test_data_RF=[]
    for i in range(len(predicted_prob_test_Temperature)):
        a=[]
        a.append(predicted_test_Temperature[i])
        a.append(predicted_test_Humidity[i])
        a.append(predicted_test_NO2[i])
        a.append(predicted_test_BC[i])
        a.append(predicted_test_PM1[i])
        a.append(predicted_test_PM25[i])
        a.append(predicted_test_PM10[i])
        a.append(predicted_test_Speed[i])
        a.append(predicted_prob_test_Temperature[i][int(predicted_test_Temperature[i])-1])
        a.append(predicted_prob_test_Humidity[i][int(predicted_test_Humidity[i])-1])
        a.append(predicted_prob_test_NO2[i][int(predicted_test_NO2[i])-1])
        a.append(predicted_prob_test_BC[i][int(predicted_test_BC[i])-1])
        a.append(predicted_prob_test_PM1[i][int(predicted_test_PM1[i])-1])
        a.append(predicted_prob_test_PM25[i][int(predicted_test_PM25[i])-1])
        a.append(predicted_prob_test_PM10[i][int(predicted_test_PM10[i])-1])    
        a.append(predicted_prob_test_Speed[i][int(predicted_test_Speed[i])-1]) 
        
        test_data_RF.append(a)
    
    x_test_data_RF=test_data_RF
    y_test_data_RF=y_test_Temperature
    
    #Data without BC
    train_data_RF_without_BC=[]
    for i in range(len(predicted_prob_train_Temperature)):
        a=[]
        a.append(predicted_train_Temperature[i])
        a.append(predicted_train_Humidity[i])
        a.append(predicted_train_NO2[i])
        a.append(predicted_train_PM1[i])
        a.append(predicted_train_PM25[i])
        a.append(predicted_train_PM10[i])
        a.append(predicted_train_Speed[i])
        a.append(predicted_prob_train_Temperature[i][int(predicted_train_Temperature[i])-1])
        a.append(predicted_prob_train_Humidity[i][int(predicted_train_Humidity[i])-1])
        a.append(predicted_prob_train_NO2[i][int(predicted_train_NO2[i])-1])
        a.append(predicted_prob_train_PM1[i][int(predicted_train_PM1[i])-1])
        a.append(predicted_prob_train_PM25[i][int(predicted_train_PM25[i])-1])
        a.append(predicted_prob_train_PM10[i][int(predicted_train_PM10[i])-1])
        a.append(predicted_prob_train_Speed[i][int(predicted_train_Speed[i])-1])
        
        train_data_RF_without_BC.append(a)
    
    x_train_data_RF_without_BC=train_data_RF_without_BC
    y_train_data_RF_without_BC=y_train_Temperature
    
    test_data_RF_without_BC=[]
    for i in range(len(predicted_prob_test_Temperature)):
        a=[]
        a.append(predicted_test_Temperature[i])
        a.append(predicted_test_Humidity[i])
        a.append(predicted_test_NO2[i])
        a.append(predicted_test_PM1[i])
        a.append(predicted_test_PM25[i])
        a.append(predicted_test_PM10[i])
        a.append(predicted_test_Speed[i])
        a.append(predicted_prob_test_Temperature[i][int(predicted_test_Temperature[i])-1])
        a.append(predicted_prob_test_Humidity[i][int(predicted_test_Humidity[i])-1])
        a.append(predicted_prob_test_NO2[i][int(predicted_test_NO2[i])-1])
        a.append(predicted_prob_test_PM1[i][int(predicted_test_PM1[i])-1])
        a.append(predicted_prob_test_PM25[i][int(predicted_test_PM25[i])-1])
        a.append(predicted_prob_test_PM10[i][int(predicted_test_PM10[i])-1])    
        a.append(predicted_prob_test_Speed[i][int(predicted_test_Speed[i])-1])
        
        
        test_data_RF_without_BC.append(a)
        
    x_test_data_RF_without_BC=test_data_RF_without_BC
    y_test_data_RF_without_BC=y_test_Temperature
    
    #Data without NO2 and BC
    train_data_RF_without_BC_NO2=[]
    for i in range(len(predicted_prob_train_Temperature)):
        a=[]
        a.append(predicted_train_Temperature[i])
        a.append(predicted_train_Humidity[i])
        a.append(predicted_train_PM1[i])
        a.append(predicted_train_PM25[i])
        a.append(predicted_train_PM10[i])
        a.append(predicted_train_Speed[i])
        a.append(predicted_prob_train_Temperature[i][int(predicted_train_Temperature[i])-1])
        a.append(predicted_prob_train_Humidity[i][int(predicted_train_Humidity[i])-1])
        a.append(predicted_prob_train_PM1[i][int(predicted_train_PM1[i])-1])
        a.append(predicted_prob_train_PM25[i][int(predicted_train_PM25[i])-1])
        a.append(predicted_prob_train_PM10[i][int(predicted_train_PM10[i])-1])    
        a.append(predicted_prob_train_Speed[i][int(predicted_train_Speed[i])-1])
        
        train_data_RF_without_BC_NO2.append(a)
    x_train_data_RF_without_BC_NO2=train_data_RF_without_BC_NO2
    y_train_data_RF_without_BC_NO2=y_train_Temperature
    
    test_data_RF_without_BC_NO2=[]
    for i in range(len(predicted_prob_test_Temperature)):
        a=[]
        a.append(predicted_test_Temperature[i])
        a.append(predicted_test_Humidity[i])
        a.append(predicted_test_PM1[i])
        a.append(predicted_test_PM25[i])
        a.append(predicted_test_PM10[i])
        a.append(predicted_test_Speed[i])
        a.append(predicted_prob_test_Temperature[i][int(predicted_test_Temperature[i])-1])
        a.append(predicted_prob_test_Humidity[i][int(predicted_test_Humidity[i])-1])
        a.append(predicted_prob_test_PM1[i][int(predicted_test_PM1[i])-1])
        a.append(predicted_prob_test_PM25[i][int(predicted_test_PM25[i])-1])
        a.append(predicted_prob_test_PM10[i][int(predicted_test_PM10[i])-1])    
        a.append(predicted_prob_test_Speed[i][int(predicted_test_Speed[i])-1])  
        
        test_data_RF_without_BC_NO2.append(a)
    
    x_test_data_RF_without_BC_NO2=test_data_RF_without_BC_NO2
    y_test_data_RF_without_BC_NO2=y_test_Temperature
    
    #Data without PMS and BC
    train_data_RF_without_BC_PMS=[]
    for i in range(len(predicted_prob_train_Temperature)):
        a=[]
        a.append(predicted_train_Temperature[i])
        a.append(predicted_train_Humidity[i])
        a.append(predicted_train_NO2[i])
        a.append(predicted_train_Speed[i])
        a.append(predicted_prob_train_Temperature[i][int(predicted_train_Temperature[i])-1])
        a.append(predicted_prob_train_Humidity[i][int(predicted_train_Humidity[i])-1])
        a.append(predicted_prob_train_NO2[i][int(predicted_train_NO2[i])-1])    
        a.append(predicted_prob_train_Speed[i][int(predicted_train_Speed[i])-1])    
        
        train_data_RF_without_BC_PMS.append(a)
    x_train_data_RF_without_BC_PMS=train_data_RF_without_BC_PMS
    y_train_data_RF_without_BC_PMS=y_train_Temperature
    
    test_data_RF_without_BC_PMS=[]
    for i in range(len(predicted_prob_test_Temperature)):
        a=[]
        a.append(predicted_test_Temperature[i])
        a.append(predicted_test_Humidity[i])
        a.append(predicted_test_NO2[i])
        a.append(predicted_test_Speed[i])
        a.append(predicted_prob_test_Temperature[i][int(predicted_test_Temperature[i])-1])
        a.append(predicted_prob_test_Humidity[i][int(predicted_test_Humidity[i])-1])
        a.append(predicted_prob_test_NO2[i][int(predicted_test_NO2[i])-1])
        a.append(predicted_prob_test_Speed[i][int(predicted_test_Speed[i])-1])
        
        test_data_RF_without_BC_PMS.append(a)
    
    x_test_data_RF_without_BC_PMS=test_data_RF_without_BC_PMS
    y_test_data_RF_without_BC_PMS=y_test_Temperature

    #Data without NO2
    train_data_RF_without_NO2=[]
    for i in range(len(predicted_prob_train_Temperature)):
        a=[]
        a.append(predicted_train_Temperature[i])
        a.append(predicted_train_Humidity[i])
        a.append(predicted_train_BC[i])
        a.append(predicted_train_PM1[i])
        a.append(predicted_train_PM25[i])
        a.append(predicted_train_PM10[i])
        a.append(predicted_train_Speed[i])
        a.append(predicted_prob_train_Temperature[i][int(predicted_train_Temperature[i])-1])
        a.append(predicted_prob_train_Humidity[i][int(predicted_train_Humidity[i])-1])
        a.append(predicted_prob_train_BC[i][int(predicted_train_BC[i])-1])
        a.append(predicted_prob_train_PM1[i][int(predicted_train_PM1[i])-1])
        a.append(predicted_prob_train_PM25[i][int(predicted_train_PM25[i])-1])
        a.append(predicted_prob_train_PM10[i][int(predicted_train_PM10[i])-1])    
        a.append(predicted_prob_train_Speed[i][int(predicted_train_Speed[i])-1])  
        
        train_data_RF_without_NO2.append(a)
        
    x_train_data_RF_without_NO2=train_data_RF_without_NO2
    y_train_data_RF_without_NO2=y_train_Temperature
    
    test_data_RF_without_NO2=[]
    for i in range(len(predicted_prob_test_Temperature)):
        a=[]
        a.append(predicted_test_Temperature[i])
        a.append(predicted_test_Humidity[i])
        a.append(predicted_test_BC[i])
        a.append(predicted_test_PM1[i])
        a.append(predicted_test_PM25[i])
        a.append(predicted_test_PM10[i])
        a.append(predicted_test_Speed[i])
        a.append(predicted_prob_test_Temperature[i][int(predicted_test_Temperature[i])-1])
        a.append(predicted_prob_test_Humidity[i][int(predicted_test_Humidity[i])-1])
        a.append(predicted_prob_test_BC[i][int(predicted_test_BC[i])-1])
        a.append(predicted_prob_test_PM1[i][int(predicted_test_PM1[i])-1])
        a.append(predicted_prob_test_PM25[i][int(predicted_test_PM25[i])-1])
        a.append(predicted_prob_test_PM10[i][int(predicted_test_PM10[i])-1])    
        a.append(predicted_prob_test_Speed[i][int(predicted_test_Speed[i])-1])
        
        test_data_RF_without_NO2.append(a)
        
    x_test_data_RF_without_NO2=test_data_RF_without_NO2
    y_test_data_RF_without_NO2=y_test_Temperature
    
    #Data without PMS
    train_data_RF_without_PMS=[]
    for i in range(len(predicted_prob_train_Temperature)):
        a=[]
        a.append(predicted_train_Temperature[i])
        a.append(predicted_train_Humidity[i])
        a.append(predicted_train_NO2[i])
        a.append(predicted_train_BC[i])
        a.append(predicted_train_Speed[i])
        a.append(predicted_prob_train_Temperature[i][int(predicted_train_Temperature[i])-1])
        a.append(predicted_prob_train_Humidity[i][int(predicted_train_Humidity[i])-1])
        a.append(predicted_prob_train_NO2[i][int(predicted_train_NO2[i])-1])
        a.append(predicted_prob_train_BC[i][int(predicted_train_BC[i])-1])
        a.append(predicted_prob_train_Speed[i][int(predicted_train_Speed[i])-1])
        
        train_data_RF_without_PMS.append(a)
    
    x_train_data_RF_without_PMS=train_data_RF_without_PMS
    y_train_data_RF_without_PMS=y_train_Temperature
    
    test_data_RF_without_PMS=[]
    for i in range(len(predicted_prob_test_Temperature)):
        a=[]
        a.append(predicted_test_Temperature[i])
        a.append(predicted_test_Humidity[i])
        a.append(predicted_test_NO2[i])
        a.append(predicted_test_BC[i])
        a.append(predicted_test_Speed[i])
        a.append(predicted_prob_test_Temperature[i][int(predicted_test_Temperature[i])-1])
        a.append(predicted_prob_test_Humidity[i][int(predicted_test_Humidity[i])-1])
        a.append(predicted_prob_test_NO2[i][int(predicted_test_NO2[i])-1])
        a.append(predicted_prob_test_BC[i][int(predicted_test_BC[i])-1])
        a.append(predicted_prob_test_Speed[i][int(predicted_test_Speed[i])-1])
    
        test_data_RF_without_PMS.append(a)
        
    x_test_data_RF_without_PMS=test_data_RF_without_PMS
    y_test_data_RF_without_PMS=y_test_Temperature
    
    #Data containing only Temperature and Humidity
    train_data_RF_only_Temperature_Humidity=[]
    for i in range(len(predicted_prob_train_Temperature)):
        a=[]
        a.append(predicted_train_Temperature[i])
        a.append(predicted_train_Humidity[i])
        a.append(predicted_train_Speed[i])
        a.append(predicted_prob_train_Temperature[i][int(predicted_train_Temperature[i])-1])
        a.append(predicted_prob_train_Humidity[i][int(predicted_train_Humidity[i])-1])
        a.append(predicted_prob_train_Speed[i][int(predicted_train_Speed[i])-1])
        
        train_data_RF_only_Temperature_Humidity.append(a)
    
    x_train_data_RF_only_Temperature_Humidity=train_data_RF_only_Temperature_Humidity
    y_train_data_RF_only_Temperature_Humidity=y_train_Temperature
    
    test_data_RF_only_Temperature_Humidity=[]
    for i in range(len(predicted_prob_test_Temperature)):
        a=[]
        a.append(predicted_test_Temperature[i])
        a.append(predicted_test_Humidity[i])
        a.append(predicted_test_Speed[i])
        a.append(predicted_prob_test_Temperature[i][int(predicted_test_Temperature[i])-1])
        a.append(predicted_prob_test_Humidity[i][int(predicted_test_Humidity[i])-1])
        a.append(predicted_prob_test_Speed[i][int(predicted_test_Speed[i])-1])
        test_data_RF_only_Temperature_Humidity.append(a)
    
    x_test_data_RF_only_Temperature_Humidity=test_data_RF_only_Temperature_Humidity
    y_test_data_RF_only_Temperature_Humidity=y_test_Temperature
    
    return x_train_data_RF,y_train_data_RF,x_test_data_RF,y_test_data_RF,x_train_data_RF_without_BC,y_train_data_RF_without_BC,x_test_data_RF_without_BC,y_test_data_RF_without_BC,x_train_data_RF_without_BC_NO2,y_train_data_RF_without_BC_NO2,x_test_data_RF_without_BC_NO2,y_test_data_RF_without_BC_NO2,x_train_data_RF_without_NO2,y_train_data_RF_without_NO2,x_test_data_RF_without_NO2,y_test_data_RF_without_NO2,x_train_data_RF_without_PMS,y_train_data_RF_without_PMS,x_test_data_RF_without_PMS,y_test_data_RF_without_PMS,x_train_data_RF_only_Temperature_Humidity,y_train_data_RF_only_Temperature_Humidity,x_test_data_RF_only_Temperature_Humidity,y_test_data_RF_only_Temperature_Humidity,x_train_data_RF_without_BC_PMS,y_train_data_RF_without_BC_PMS,x_test_data_RF_without_BC_PMS,y_test_data_RF_without_BC_PMS




# evaluate random forest algorithm for classification
def create_meta_learner(x_train_data_RF,y_train_data_RF,x_test_data_RF, y_test_data_RF):
    model_multi_view = RandomForestClassifier()
    model_multi_view.fit(x_train_data_RF,y_train_data_RF)
    # evaluate the model
    x_test_RF, y_test_RF=shuffle( x_test_data_RF, y_test_data_RF)

    X=np.append(x_train_data_RF,x_test_data_RF,axis=0)
    y=np.append(y_train_data_RF,y_test_data_RF,axis=0)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model_multi_view, X,y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    return model_multi_view,n_scores


