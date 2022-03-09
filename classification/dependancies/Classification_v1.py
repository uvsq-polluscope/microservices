import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from dependancies.classification_functions import *

def classification_v1(df, participant_virtual_id):
    
    # df = get_processed_data(participant_virtual_id=participant_virtual_id, kit_id=kit_id)
    
    dfs=splitting(df)
    labels,labels_Temperature,labels_Humidity,labels_NO2,labels_BC,labels_PM1,labels_PM25,labels_PM10,labels_Speed=predict_labels_raw_data_RF(dfs,model_path='./dependancies/models_RF/',classes_removed=False)
    classes={}
    for key,value in labels.items():
        classes[key]=get_key(most_frequent(list(value)),dictionary=dictionary_RECORD)

    c=validate_results_VGP_annotations(df,classes,correct_annotations=True)
    #file_name='./VGP_Predictions/VGP_Predictions_Original/'+str(participant_virtual_id)+'.csv'
    #data_class = convert_predictions_to_DF(classes,participant_virtual_id)
    #data_class['timestamp'] =  pd.to_datetime(data_class['timestamp'])
    #data_class.to_csv(file_name,index=False)
    
    
    # file_name='./VGP_Predictions/VGP_Predictions_Time_Correction/'+str(participant_virtual_id)+'.csv'
    data_class2 = convert_predictions_to_DF(c[2],participant_virtual_id)
    data_class2['timestamp'] =  pd.to_datetime(data_class2['timestamp'])
    #data_class2.to_csv(file_name,index=False)
    
    return data_class2