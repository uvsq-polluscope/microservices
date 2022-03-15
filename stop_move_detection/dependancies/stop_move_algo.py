#from preprocessing import *
import pandas as pd
from sqlalchemy import Table, Column, MetaData, Integer, Computed
from sqlalchemy import create_engine
from dependancies.stop_move_detection import *
from dependancies.TrajectoryFeatures import *

engine = create_engine(
    'postgresql://dwaccount:password@127.0.0.1:5435/dwaccount')


def def_stop_move_detection(data):    
    # For a single participant_virtual_id :
    # use segmented_data to retrive movement in instance move and stop evenements in data1
    participant_virtual_id = data["participant_virtual_id"][0]
    data1, mixed1, moves = segmented_data(
        participant_virtual_id=participant_virtual_id)
    df_stops = data1[data1._stops_ != 1]
    df_stops = df_stops[['participant_virtual_id',
                         'time', 'activity', 'stops']]
    df_stops.rename(columns={'stops': 'detected_label'}, inplace=True)

    print("_____________________________DF_df_stops___________________________________")
    print(df_stops)
    print("_____________________________DF_df_stops___________________________________")

    # apply features_set 
    if 'time' in moves.columns:
        features_set = get_features(df = moves, participant_virtual_id=participant_virtual_id)
    else:
        moves.reset_index(inplace=True)
        features_set = get_features(df = moves, participant_virtual_id=participant_virtual_id)

    labels = {0:'Vélo',1:'Motorcycle', 2:'Walk', 3:'Bus',
            4:'Voiture', 5:'Métro', 6:'Running', 7:'Train',
        8:'Parc', 9:'indoor'}

    data2 = pd.concat([data, features_set])

    # Detection of transport mode 
    data_ft = data2[important_features]
    X = data_ft.drop(['time', 'userId','target'], axis=1)
    y_pred=loaded_RF.predict(np.nan_to_num(X))
    data_ft['pred']=y_pred
    df_moves = data_ft[['time', 'userId','target', 'pred']]

    df_moves['prediction'] = df_moves.pred.map(labels)
    
    print("_____________________________DF_df_moves___________________________________")
    print(df_moves)
    print("_____________________________DF_df_moves___________________________________")

    # concat df_stops and df_moves to get a single dataframe for post processing
    df = pd.concat([df_stops,df_moves])
    df.sort_values(['participant_virtual_id', 'time'],inplace=True)

    print("_____________________________DF_df___________________________________")
    print(df)
    print("_____________________________DF_df___________________________________")

    return df
