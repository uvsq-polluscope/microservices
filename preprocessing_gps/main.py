
# import locale file
from ast import Delete, Pass
#from msilib.schema import Error
from dependancies.pre_process_gps import *
from object.ConsumerRawDataGPS import ConsumerRawDataGPS
from object.ProducerRawDataGPS import ProducerRawDataGPS
from uuid import uuid4

from object.helpers import default_json_serialize, todict

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

# fastapi config
app = FastAPI()

# kafka consumer config
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_NAME_CONSUME = "rawdataGPS"

schema_registry_client = SchemaRegistryClient({"url": "http://localhost:8085"})
consumer_conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                 "key.deserializer": StringDeserializer("utf_8"),
                 "value.deserializer": AvroDeserializer(schema_str=None,  # the schema should be fetched from the registry
                                                        schema_registry_client=schema_registry_client
                                                        ),
                 "group.id": "preprocessGPS",
                 "auto.offset.reset": "earliest"}

consumer = DeserializingConsumer(consumer_conf)
consumer.subscribe(["rawdataGPS"])

TOPIC_NAME_PRODUCE = "ProducerRawDataGPS"

# --- Producing part ---

producer_conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                 "key.serializer": StringSerializer("utf_8"),
                 "value.serializer": AvroSerializer(schema_str=ProducerRawDataGPS.schema,
                                                    schema_registry_client=schema_registry_client,
                                                    to_dict=todict)}

producer = SerializingProducer(producer_conf)

# ---------------------------------------------------------------------------------------------------------------------------


@app.get("/")
def runing():
    return "Hi preprocessing_gps microservice is running"


@app.get("/preprocessing_gps")
def preprocessing_gps():

    # Dict contain data for each participant
    data = {}
    # Lestener consumer kafka
    while True:
        try:
            try:
                msg = consumer.poll(1.0)
            except:
                return "plz create RAWDATA TOPIC"

            if msg is None:
                print("aucun msg")
                continue
            message = msg.value()

            if message is not None:
                # cast data to consumer object
                rowdata = ConsumerRawDataGPS(message)
                print(rowdata)
                # check if get_id exist in data else create it
                if rowdata.get_tablet_id() in data.keys():
                    data.get(rowdata.get_tablet_id()
                             ).append(rowdata)
                else:
                    data[rowdata.get_tablet_id()] = [rowdata]

            # check if the number of values is reached
            key = rate_done(data)

            if (key != "No"):
                # run preprocessingGPS algo
                df = get_df(data.get(key))
                print(df)
                df = run(df)

                # delete values from memory
                data[key] = []

                # save data in kafka topic AND DW
                # save_data(df)

        except KeyboardInterrupt:
            break

    consumer.close()

    return True


def rate_done(data):
    for key in data.keys():
        if (len(data.get(key)) == 60):
            print("rate done for key " + str(key))
            return key
        else:
            return "No"


def save_data(df):
    for ind in df.index:
        msg = ProducerRawDataGPS(dict(
        ))
    return True


def get_data():

    return True


def get_df(l):
    # from list of object to dataframe
    data = pd.DataFrame(
        [], columns=['id', 'tablet_id', 'timestamp', 'lat', 'lon'])
    for elm in l:
        data = data.append(elm.__dict__, ignore_index=True)

    data["timestamp"] = data["time"]
    del data["time"]
    return data


def run(df):
    # run preprocessing algo
    try:
        df = data_pre_processing_gps(df)
        return df
    except:
        print("erreur")
        return pd.DataFrame()
