# import locale file
from ast import Delete, Pass
from uuid import uuid4
from object.helpers import todict
from object.helpers import default_json_serialize
from object.ConsumerRawDataSMD import *
from object.stopMoveDetectionTopic import *


from dependancies.stop_move_algo import *
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

# Initialize Flask
app = FastAPI()

# kafka consumer config
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_NAME_CONSUME = "rawdataSMD"

schema_registry_client = SchemaRegistryClient({"url": "http://localhost:8085"})
consumer_conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                 "key.deserializer": StringDeserializer("utf_8"),
                 "value.deserializer": AvroDeserializer(schema_str=None,  # the schema should be fetched from the registry
                                                        schema_registry_client=schema_registry_client
                                                        ),
                 "group.id": "Move_Stop_Detection",
                 "auto.offset.reset": "earliest"}

consumer = DeserializingConsumer(consumer_conf)
consumer.subscribe(["rawdataSMD"])

TOPIC_NAME_PRODUCE = "stopMoveDetectionTopic"

# --- Producing part ---

producer_conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                 "key.serializer": StringSerializer("utf_8"),
                 "value.serializer": AvroSerializer(schema_str=stopMoveDetectionTopic.schema,
                                                    schema_registry_client=schema_registry_client,
                                                    to_dict=todict)}

producer = SerializingProducer(producer_conf)


@app.get("/")
def runing():
    return "Hi stop_move_detection microserivce is runing"


@app.get("/stop_move_detection_algo")
def stop_move_detection_algo():
    print("START stop_move_detection_algo ")
    # Dict contain data for each participant
    data = {}
    while True:
        try:
            try:
                msg = consumer.poll(1)
            except Exception as e:
                print (e)
                return "Error: rawdataSMD topic is not created, please create it !"
            if msg is None:
                continue
            message = msg.value()

            if message is not None:
                # cast data to consumer object
                rowdata = ConsumerRawDataSMD(message)
                # check if get_participant_virtual_id exists in dictionary else create it
                if rowdata.get_participant_virtual_id() in data.keys():
                    data.get(rowdata.get_participant_virtual_id()
                             ).append(rowdata)
                else:
                    data[rowdata.get_participant_virtual_id()] = [rowdata]

            # check if there is enough data to start the pre-processing
            key = rate_done(data)
            if (key != "No"):
                # get dataframe from dictionary values
                df = get_df(data.get(key))

                print(df)

                # run stop move detection algorithme
                df = run(df)

                print(df)

                # delete values from memory
                data[key] = []

                # save data in kafka topic
                save_data(df)
        except KeyboardInterrupt as e:
            print(e)
            print(
                f'Stopped listening to messages on topic {TOPIC_NAME_CONSUME}')
            break

    consumer.close()
    print("END stop_move_detection_algo ")
    return True


def rate_done(data):
    for key in data.keys():
        if (len(data.get(key)) == 20):
            return key
        else:
            return "No"


def save_data(df):
    print("START save_data ")
    print("DF =======================================")
    print(df)

    if df.empty:
        print("Dataframe is empty, no message to produce !")
    else:
        produced_message_count = 0
        for ind in df.index:
            msg = stopMoveDetectionTopic(dict(
                participant_virtual_id=str(
                    df['participant_virtual_id'][ind]),
                time=str(df['time'][ind]),
                activity=str(df['activity'][ind]),
                detected_label=str(df['detected_label'][ind]),
                userId=str(df['userId'][ind]),
                target=str(df['target'][ind]),
                pred=str(df['pred'][ind]),
                prediction=str(df['prediction'][ind])
            ))
            print(msg.dict())
            producer.produce(topic=TOPIC_NAME_PRODUCE,
                             key=str(uuid4()), value=msg)
            produced_message_count += 1
            producer.flush()
        print(f"Produced {produced_message_count} message")
    print("END save_data ")


def get_df(l):
    # from list of object to dataframe
    data = pd.DataFrame(
        [], columns=['participant_virtual_id', 'time', 'lat', 'lon', 'hilbert', 'activity'])
    for elm in l:
        data = data.append(elm.__dict__, ignore_index=True)
    return data


def run(df):
    # run stop move detection algorithme
    try:
        df = def_stop_move_detection(df)
        return df
    except Exception as e:
        print(f'erreur in main :{e}')
        return pd.DataFrame()
