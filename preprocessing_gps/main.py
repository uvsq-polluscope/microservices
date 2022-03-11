
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
                print(rowdata.dict())
                # check if get_id exist in data else create it
                if rowdata.get_participant_virtual_id() in data.keys():
                    data.get(rowdata.get_participant_virtual_id()
                             ).append(rowdata)
                else:
                    data[rowdata.get_participant_virtual_id()] = [rowdata]

            # check if the number of values is reached
            key = rate_done(data)

            if (key != "No"):
                # run preprocessingGPS algo
                df = get_df(data.get(key))
                print(df["id"].values)
                break
                print(df)
                df = run(df)

                # delete values from memory
                data[key] = []

                # save data in kafka topic
                save_data(df)

        except KeyboardInterrupt:
            break

    consumer.close()

    return True


def rate_done(data):
    for key in data.keys():
        if (len(data.get(key)) == 20):
            print("rate done for key " + str(key))
            return key
        else:
            return "No"


def save_data(df):
    print("save_data")
    print(df)
    print(str(df['time'][0]))
    produced_message_count = 0
    for ind in df.index:
        msg = ProducerRawDataGPS(dict(
            participant_virtual_id=str(df['participant_virtual_id'][ind]),
            time=str(df['time'][ind]),
            lat=float(df['lat'][ind]),
            lon=float(df['lon'][ind]),
            hilbert=int(df['hilbert'][ind]),
            activity=str(df['activity'][ind])
        ))
        producer.produce(topic=TOPIC_NAME_PRODUCE, key=str(uuid4()), value=msg)
        produced_message_count += 1
        producer.flush()
    print(f"Produced {produced_message_count}message")


def get_data():

    return True


def get_df(l):
    # from list of object to dataframe
    data = pd.DataFrame(
        [], columns=['id', 'participant_virtual_id', 'tablet_id', 'time', 'lat', 'lon'])
    for elm in l:
        data = data.append(elm.__dict__, ignore_index=True)
    return data


def run(df):
    # run preprocessing algo
    try:
        df = data_pre_processing_gps(df)
        return df
    except Exception as e:
        print(f'erreur :{e}')
        return pd.DataFrame()
