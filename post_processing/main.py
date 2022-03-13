# import locale file
from ast import Delete, Pass
from uuid import uuid4
from object.helpers import *
from object.ConsumerClassification import *
from object.ConsumerSMD import *
from object.Producer import *


import pandas as pd
from sqlalchemy import Table, Column, MetaData, Integer, Computed
from sqlalchemy import create_engine

# import sqlalchemy
from sqlalchemy import create_engine

# import kafka
from confluent_kafka import DeserializingConsumer, SerializingProducer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer, AvroSerializer
from confluent_kafka.serialization import StringDeserializer, StringSerializer

# help fonctions
import pandas as pd

# import fastApi
from fastapi import FastAPI

app = FastAPI()

# ---------------------------------------------------------------------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_NAME_CONSUME_CLASSIFICATION = "classificationTopic"
TOPIC_NAME_CONSUME_SMD = "stopMoveDetectionTopic"
TOPIC_NAME_PRODUCER = "postProcessingTopic"

schema_registry_client = SchemaRegistryClient({"url": "http://localhost:8085"})

# kafka consumer classification config
consumer_classification_conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                                "key.deserializer": StringDeserializer("utf_8"),
                                "value.deserializer": AvroDeserializer(schema_str=None,  # the schema should be fetched from the registry
                                                                       schema_registry_client=schema_registry_client
                                                                       ),
                                "group.id": "Move_Stop_Detection",
                                "auto.offset.reset": "earliest"}

consumer_classification = DeserializingConsumer(consumer_classification_conf)
consumer_classification.subscribe([TOPIC_NAME_CONSUME_CLASSIFICATION])

# kafka consumer stop_move_detection config
consumer_smd_conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                     "key.deserializer": StringDeserializer("utf_8"),
                     "value.deserializer": AvroDeserializer(schema_str=None,  # the schema should be fetched from the registry
                                                            schema_registry_client=schema_registry_client
                                                            ),
                     "group.id": "Move_Stop_Detection",
                     "auto.offset.reset": "earliest"}

consumer_smd = DeserializingConsumer(consumer_smd_conf)
consumer_smd.subscribe([TOPIC_NAME_CONSUME_SMD])

# kafka producer config
producer_conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                 "key.serializer": StringSerializer("utf_8"),
                 "value.serializer": AvroSerializer(schema_str=Producer.schema,
                                                    schema_registry_client=schema_registry_client,
                                                    to_dict=todict)}

producer = SerializingProducer(producer_conf)
# ---------------------------------------------------------------------------------------------------------------------------


@app.get("/")
def runing():
    return "Hi postprocessing microserivce is runing"


@app.get("/postprocessing")
def postprocessing():
    # Dict contain data for each participant
    data_classification = {}
    data_smd = {}
    # Always listening to kafka consumer
    print(
        f'Listening to messages on topic {TOPIC_NAME_CONSUME_CLASSIFICATION} and {TOPIC_NAME_CONSUME_SMD} ...')
    while True:
        try:
            # RECOVER DATA FROM CLASSIFICATION CONSUMER
            try:
                msg_classification = consumer_classification.poll(1.0)
            except:
                return f"Error: {TOPIC_NAME_CONSUME_CLASSIFICATION} topic is not created, please create it !"

            if msg_classification is None:
                continue
            message_classification = msg_classification.value()
            print(f'Classification message: {message_classification}')

            if message_classification is not None:
                # cast data to consumer object
                rowdata_classification = ConsumerClassification(
                    message_classification)
                # check if get_participant_virtual_id exists in dictionary else create it
                if rowdata_classification.get_participant_virtual_id() in data_classification.keys():
                    data_classification.get(rowdata_classification.get_participant_virtual_id()
                                            ).append(rowdata_classification)
                else:
                    data_classification[rowdata_classification.get_participant_virtual_id()] = [
                        rowdata_classification]

             # RECOVER DATA FROM SMD CONSUMER
            try:
                msg_smd = consumer_smd.poll(1.0)
            except:
                return f"Error: {TOPIC_NAME_CONSUME_SMD} topic is not created, please create it !"

            if msg_smd is None:
                continue
            message_smd = msg_smd.value()
            print(f'SMD message: {message_smd}')

            if message_smd is not None:
                # cast data to consumer object
                rowdata_smd = ConsumerSMD(message_smd)
                # check if get_participant_virtual_id exists in dictionary else create it
                if rowdata_smd.get_participant_virtual_id() in data_smd.keys():
                    data_smd.get(rowdata_smd.get_participant_virtual_id()
                                 ).append(rowdata_smd)
                else:
                    data_smd[rowdata_smd.get_participant_virtual_id()] = [
                        rowdata_smd]

            # check if there is enough data to start the pre-processing
            key = rate_done(data_classification, data_smd)

            if (key != "No"):
                # get dataframe from dictionary values
                df_classification = get_df_classification(
                    data_classification.get(key))
                df_smd = get_df_smd(data_smd.get(key))
                print(df_classification)
                print(df_smd)

                # run preprocessingGPS algo
                df = run(df_classification, df_smd)

                # delete values from memory
                data_smd[key] = []
                data_smd[key] = []

                # save data in kafka topic
                save_data(df)

        except KeyboardInterrupt:
            print(
                f'Stopped listening to messages on topic {TOPIC_NAME_CONSUME_CLASSIFICATION} and {TOPIC_NAME_CONSUME_SMD}')
            break

    consumer_classification.close()
    consumer_smd.close()

    return True


def rate_done(data1, data2):
    for key in data1.keys():
        if (len(data1.get(key)) == 2):
            print("There is enough classification data for key " + str(key))
            for key in data2.keys():
                if (len(data2.get(key)) == 2):
                    print("There is enough SMD data for key " + str(key))
                    return key
                else:
                    return "No"
        else:
            return "No"


def get_df_classification(l):
    # from list of object to dataframe
    data = pd.DataFrame(
        [], columns=['participant_virtual_id', 'time', 'activity'])
    for elm in l:
        data = data.append(elm.__dict__, ignore_index=True)
    return data


def get_df_smd(l):
    # from list of object to dataframe
    data = pd.DataFrame(
        [], columns=['participant_virtual_id', 'time', 'activity', 'detected_label'])
    for elm in l:
        data = data.append(elm.__dict__, ignore_index=True)
    return data


def save_data(df):
    print("save_data")
    print(df)
    if df.empty:
        print("Dataframe is empty, no message to produce !")
    else:
        produced_message_count = 0
        for ind in df.index:
            # TO MODIFY
            msg = Producer(dict(
                participant_virtual_id=str(df['participant_virtual_id'][ind]),
                time=str(df['time'][ind]),
                activity=str(df['activity'][ind])
            ))
            producer.produce(topic=TOPIC_NAME_PRODUCER,
                             key=str(uuid4()), value=msg)
            produced_message_count += 1
            producer.flush()
        print(f"Produced {produced_message_count} message")


def get_df_classification(l):
    # from list of object to dataframe
    data = pd.DataFrame(
        [], columns=['participant_virtual_id', 'time', 'activity'])
    for elm in l:
        data = data.append(elm.__dict__, ignore_index=True)
    return data


def get_df_stop_move_detection(l):
    # from list of object to dataframe
    data = pd.DataFrame(
        [], columns=['participant_virtual_id', 'time', 'activity', 'detected_label'])
    for elm in l:
        data = data.append(elm.__dict__, ignore_index=True)
    return data


def run(df1, df2):
    # run stop move detection algorithme
    try:
        print("run")
        return True
        #df = post_processing(df1, df2)
        # return df
    except Exception as e:
        print(f'erreur in main :{e}')
        return pd.DataFrame()
