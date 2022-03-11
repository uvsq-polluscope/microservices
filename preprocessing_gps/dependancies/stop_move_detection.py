import pandas as pd
import numpy as np
from hilbert_detection import stop_hilbert_vgp
from skmob_detection import stop_skmob
from utils import precision_recall, splitting_2, splitting2
from db_connection import *
import TrajectorySegmentation as ts
import Trajectory as tr
from TrajectoryDescriptor import *
from sklearn.ensemble import RandomForestClassifier
import pickle


participants = pd.read_csv('../list_participants.csv',
                           infer_datetime_format=True, parse_dates=[3, 4])

participant_virtual_ids = participants.participant_virtual_id.unique()


def segmented_data(participant_virtual_id=9999915):

    df_hilbert = stop_hilbert_vgp(
        participant_virtual_id=participant_virtual_id)

    dfs_hilbert = splitting_2(df_hilbert)

    # delete mixed segments
    mixed = []
    moves = []

    for i in reversed(range(len(dfs_hilbert))):

        if len(dfs_hilbert[i].activity.unique()) > 1:

            mixed.append(dfs_hilbert[i])

            del dfs_hilbert[i]

        elif len(dfs_hilbert[i]) == 0:  # delete empty dfs

            del dfs_hilbert[i]

        elif dfs_hilbert[i]._stops_.median() == 1:

            moves.append(dfs_hilbert[i])

    timestamp = [dfs_hilbert[i].time.min() for i in range(len(dfs_hilbert))]
    activities = [dfs_hilbert[i].activity.unique()[0]
                  for i in range(len(dfs_hilbert))]
    stops = [dfs_hilbert[i].stop.median() for i in range(len(dfs_hilbert))]
    _activity_ = [dfs_hilbert[i]._activity_.median()
                  for i in range(len(dfs_hilbert))]
    _stops_ = [dfs_hilbert[i]._stops_.median()
               for i in range(len(dfs_hilbert))]

    data = pd.DataFrame({'participant_virtual_id': dfs_hilbert[0].participant_virtual_id.unique()[0],
                         'time': timestamp,
                         'activity': activities,
                         'stops': stops,
                         '_activity_': _activity_,
                         '_stops_': _stops_})

    moves = pd.concat(moves)

    moves.to_csv('..\..\TrajLib\data\moves_' +
                 str(participant_virtual_id)+'.csv', index=False)

    return data, mixed


def get_features(participant_virtual_id=9999915):

    ts_obj = ts.TrajectorySegmentation()

    ts_obj.load_data(lat='lat', lon='lon', time_date='time', labels=[
                     'activity'], src='data/moves_'+str(participant_virtual_id)+'.csv', seperator=',')

    dfs = splitting2(ts_obj.return_row_data())

    # TrajLib code
    i = 1
    features = []
    for seg in range(len(dfs)):
        # only use segments longer than 10
        if(dfs[seg].shape[0] > 5):
            tr_obj = tr.Trajectory(
                mood='df', trajectory=dfs[seg], labels=['activity'])

            tr_obj.point_features()  # generate point_features
            f = tr_obj.segment_features()  # generate trajectory_features
            userid = dfs[seg].participant_virtual_id.iloc[0]
            time_id = dfs[seg].index.min()

            f.append(userid)
            f.append(time_id)
            features.append(np.array(f))
            i = i+1
            if (i % 300) == 1:
                print(i)

    bearingSet = ['bearing_min', 'bearing_max', 'bearing_mean', 'bearing_median',
                  'bearing_std', 'bearing_p10', 'bearing_p25', 'bearing_p50', 'bearing_p75', 'bearing_p90']
    speedSet = ['speed_min', 'speed_max', 'speed_mean', 'speed_median',
                'speed_std', 'speed_p10', 'speed_p25', 'speed_p50', 'speed_p75', 'speed_p90']
    distanceSet = ['distance_min', 'distance_max', 'distance_mean', 'distance_median',
                   'distance_std', 'distance_p10', 'distance_p25', 'distance_p50', 'distance_p75', 'distance_p90']
    accelerationSet = ['acceleration_min', 'acceleration_max', 'acceleration_mean', 'acceleration_median',
                       'acceleration_std', 'acceleration_p10', 'acceleration_p25', 'acceleration_p50', 'acceleration_p75', 'acceleration_p90']
    jerkSet = ['jerk_min', 'jerk_max', 'jerk_mean', 'jerk_median',
               'jerk_std', 'jerk_p10', 'jerk_p25', 'jerk_p50', 'jerk_p75', 'jerk_p90']
    brateSet = ['bearing_rate_min', 'bearing_rate_max', 'bearing_rate_mean', 'bearing_rate_median', 'bearing_rate_std',
                'bearing_rate_p10', 'bearing_rate_p25', 'bearing_rate_p50', 'bearing_rate_p75', 'bearing_rate_p90']
    brate_rateSet = ['brate_rate_min', 'brate_rate_max', 'brate_rate_mean', 'brate_rate_median',
                     'brate_rate_std', 'brate_rate_p10', 'brate_rate_p25', 'brate_rate_p50', 'brate_rate_p75', 'brate_rate_p90']
    stop_timeSet = ['stop_time_min', 'stop_time_max', 'stop_time_mean', 'stop_time_median',
                    'stop_time_std', 'stop_time_p10', 'stop_time_p25', 'stop_time_p50', 'stop_time_p75', 'stop_time_p90']

    targetset = set(ts_obj.return_row_data().activity)
    col = distanceSet+speedSet+accelerationSet+bearingSet+jerkSet+brateSet+brate_rateSet+stop_timeSet+['isInValid', 'isPure', 'target', 'stopRate', 'starTime', 'endTime',  'isWeekDay', 'dayOfWeek', 'durationInSeconds', 'distanceTravelled', 'startToEndDistance',
                                                                                                       'startLat', 'starLon', 'endLat', 'endLon', 'selfIntersect', 'modayDistance', 'tuesdayDistance', 'wednesdayDay', 'thursdayDistance', 'fridayDistance', 'saturdayDistance', 'sundayDistance', 'stopTotal', 'stopTotalOverDuration', 'userId', 'time']

    features_set = pd.DataFrame(features, columns=col)
    features_set.to_csv('data/features_participant_' +
                        str(participant_virtual_id)+'.csv')

    return features_set


important_features = ['speed_p25', 'distance_p90', 'distance_std', 'distance_p75',
                      'distance_mean', 'sundayDistance', 'speed_p10', 'fridayDistance',
                      'starLon', 'endLat', 'jerk_p25', 'startLat', 'jerk_p75',
                      'acceleration_min', 'endLon', 'speed_p75', 'speed_p90',
                      'bearing_rate_p75', 'selfIntersect', 'target', 'userId', 'time']

loaded_RF = pickle.load(open('RF_transport_detection.sav', 'rb'))
