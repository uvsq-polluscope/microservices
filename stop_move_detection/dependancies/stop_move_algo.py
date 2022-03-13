#from preprocessing import *
import pandas as pd
from sqlalchemy import Table, Column, MetaData, Integer, Computed
from sqlalchemy import create_engine
from dependancies.stop_move_detection import *

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

    # On cherche les moments en mouvement avec get features qui renvoie move detection
    #features_set = get_features(df = moves, participant_virtual_id=participant_virtual_id)
    #data2 = pd.concat([data2, features_set])
    #data_ft = data2[important_features]
    #X = data_ft.drop(['time', 'userId','target'], axis=1)
    # X = np.nan_to_num(X) # Replace NaN with zero and infinity with large finite numbers
    # y_pred=loaded_RF.predict(X)
    # data_ft['pred']=y_pred
    #df_moves = data_ft[['time', 'userId','target', 'pred']]

    # concat df_stops and df_moves to get a single dataframe for post processing
    #df = pd.concat([df_stops,df_moves])
    #df.sort_values(['participant_virtual_id', 'time'],inplace=True)
    return df_stops
