import pandas as pd


from data_operations import *



def impute_limit(data, method = 1, window_size = 8, limit = 10,parameters=[]):


    df = data.copy()

    #create a flag column for every parameter (False ==> Not Imputed, True ===> Imputed)
    df['imputed NO2'] = [False for i in range(df.shape[0])]
    df['imputed temperature'] = [False for i in range(df.shape[0])]
    df['imputed humidity'] = [False for i in range(df.shape[0])]
    df['imputed PM1.0'] = [False for i in range(df.shape[0])]
    df['imputed PM10'] = [False for i in range(df.shape[0])]
    df['imputed PM2.5'] = [False for i in range(df.shape[0])]
    df['imputed BC'] = [False for i in range(df.shape[0])]
    #List of parameters to impute.
#     parameters = df.columns.to_list() #['NO2', 'humidity','PM1.0','PM2.5','PM10','temperature', 'BC']
    for parameter in parameters:
        if parameter not in df.columns:
            del df['imputed '+parameter]
            parameters.remove(parameter)
    if method == 1:
        values = []
        for parameter in parameters:
            nullvalues = []
            nullvalues = df[df[parameter].isnull()].index.tolist()
            df[parameter].interpolate(method = "linear", inplace = True,limit = limit)
            for index in nullvalues:
                if  np.isnan(df.at[index, parameter]) == False:
                    # If the value was imputed, change the flag to True.
                    df.at[index, 'imputed ' + parameter] = True
    else:
        values = []
        for parameter in parameters:
            nullvalues = []
            nullvalues = df[df[parameter].isnull()].index.tolist()
            df[parameter].fillna(df[parameter].rolling(window_size, min_periods=1, center=True).mean(), inplace = True)
            for index in nullvalues:
                if  np.isnan(df.at[index, parameter]) == False:
                    # If the value was imputed, change the flag to True.
                    df.at[index, 'imputed ' + parameter] = True

    
    return df


def data_pre_processing(data,columnNames=["Temperature"],min_threshold={"Temperature":-20,"Humidity":0,"NO2":0,"BC":-1500,"PM1.0":0,"PM2.5":0,"PM10":0,"Speed":-1},max_threshold={"Temperature":50,"Humidity":120,"NO2":250,"BC":50000,"PM1.0":300,"PM2.5":300,"PM10":300,"Speed":40}):
    df=data
    parameters=[]
    if len(df.dropna(subset=["Temperature"]))==0:
        print("No data found")
        return []
    if len(df.dropna(subset=["Temperature"]))>0:
        new_df,indices,aggregation,avg_map=spikes_detection_validation_with_changes_negative_replaced_by_abs(df,columnNames,"Temperature",min_threshold=min_threshold["Temperature"],max_threshold=max_threshold["Temperature"],window_size=10,interval=2,show_plot=False)
        df_Temperature=mean_peaks_removing_all_peaks_negative_replaced_by_abs(df,"Temperature",indices,show=False)
        parameters.append("Temperature")
        df_Temperature=df_Temperature[["participant_virtual_id","time","Temperature","activity","event"]]
    if len(df.dropna(subset=["Humidity"]))>0:
        new_df,indices,aggregation,avg_map=spikes_detection_validation_with_changes_negative_replaced_by_abs(df,columnNames,"Humidity",min_threshold=min_threshold["Humidity"],max_threshold=max_threshold["Humidity"],window_size=10,interval=2,show_plot=False)
        df_Humidity=mean_peaks_removing_all_peaks_negative_replaced_by_abs(df,"Humidity",indices,show=False)
        parameters.append("Humidity")
        df_Humidity=df_Humidity[["time","Humidity"]]
    if len(df.dropna(subset=["NO2"]))>0:
        new_df,indices,aggregation,avg_map=spikes_detection_validation_with_changes_negative_replaced_by_abs(df,columnNames,"NO2",min_threshold=min_threshold["NO2"],max_threshold=max_threshold["NO2"],window_size=10,interval=2,show_plot=False)
        df_NO2=mean_peaks_removing_all_peaks_negative_replaced_by_abs(df,"NO2",indices,show=False)
        parameters.append("NO2")
        df_NO2=df_NO2[["time","NO2"]]
    if len(df.dropna(subset=["BC"]))>0:
        new_df,indices,aggregation,avg_map=spikes_detection_validation_with_changes_negative_replaced_by_abs(df,columnNames,"BC",min_threshold=min_threshold["BC"],max_threshold=max_threshold["BC"],window_size=10,interval=2,show_plot=False)
        df_BC=mean_peaks_removing_all_peaks_negative_replaced_by_abs(df,"BC",indices,show=False)
        parameters.append("BC")
        df_BC=df_BC[["time","BC"]]
    if len(df.dropna(subset=["PM1.0"]))>0:
        new_df,indices,aggregation,avg_map=spikes_detection_validation_with_changes_negative_replaced_by_abs(df,columnNames,"PM1.0",min_threshold=min_threshold["PM1.0"],max_threshold=max_threshold["PM1.0"],window_size=10,interval=2,show_plot=False)
        df_PM1=mean_peaks_removing_all_peaks_negative_replaced_by_abs(df,"PM1.0",indices,show=False)
        parameters.append("PM1.0")
        df_PM1=df_PM1[["time","PM1.0"]]
    if len(df.dropna(subset=["PM2.5"]))>0:
        print("herrrrr01")
        new_df,indices,aggregation,avg_map=spikes_detection_validation_with_changes_negative_replaced_by_abs(df,columnNames,"PM2.5",min_threshold=min_threshold["PM2.5"],max_threshold=max_threshold["PM2.5"],window_size=10,interval=2,show_plot=False)
        df_PM25=mean_peaks_removing_all_peaks_negative_replaced_by_abs(df,"PM2.5",indices,show=False)
        parameters.append("PM2.5")
        df_PM25=df_PM25[["time","PM2.5"]]
    if len(df.dropna(subset=["PM10"]))>0:
        new_df,indices,aggregation,avg_map=spikes_detection_validation_with_changes_negative_replaced_by_abs(df,columnNames,"PM10",min_threshold=min_threshold["PM10"],max_threshold=max_threshold["PM10"],window_size=10,interval=2,show_plot=False)
        df_PM10=mean_peaks_removing_all_peaks_negative_replaced_by_abs(df,"PM10",indices,show=False)
        parameters.append("PM10")
        df_PM10=df_PM10[["time","PM10"]]
    if len(df.dropna(subset=["vitesse(m/s)"]))>0:
        new_df,indices,aggregation,avg_map=spikes_detection_validation_with_changes_negative_replaced_by_abs(df,columnNames,"vitesse(m/s)",min_threshold=min_threshold["Speed"],max_threshold=max_threshold["Speed"],window_size=10,interval=2,show_plot=False)
        df_speed=mean_peaks_removing_all_peaks_negative_replaced_by_abs(df,"vitesse(m/s)",indices,show=False)
        df_speed=df_speed[["time","vitesse(m/s)"]]
    
    if len(df.dropna(subset=["Temperature"]))>0:
        df_new=df_Temperature
        if len(df.dropna(subset=["Humidity"]))>0:
            df_new=pd.merge(df_new,df_Humidity,on="time",how="outer")
        
        if len(df.dropna(subset=["NO2"]))>0:
            df_new=pd.merge(df_new,df_NO2,on="time",how="outer")
        
        if len(df.dropna(subset=["BC"]))>0:
            df_new=pd.merge(df_new,df_BC,on="time",how="outer")
        
        if len(df.dropna(subset=["PM1.0"]))>0:
            df_new=pd.merge(df_new,df_PM1,on="time",how="outer")
        
        if len(df.dropna(subset=["PM2.5"]))>0:
            print("herrrrrr2")
            df_new=pd.merge(df_new,df_PM25,on="time",how="outer")
        
        if len(df.dropna(subset=["PM10"]))>0:
            df_new=pd.merge(df_new,df_PM10,on="time",how="outer")
        if len(df.dropna(subset=["vitesse(m/s)"]))>0:
            df_new=pd.merge(df_new,df_speed,on="time",how="outer")
        cols = list(df_new.columns.values)
        cols.pop(cols.index("activity"))
        cols.pop(cols.index("event"))
        df_new=df_new[cols+["activity","event"]]        
        #data imputation
        print(cols)
        df_new = impute_limit(df_new, method = 2,parameters=parameters)
        return df_new
    else:
        print("NO data for temperature")
        return []
    
