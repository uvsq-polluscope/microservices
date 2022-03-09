
#import locale file
from ast import Delete, Pass
from msilib.schema import Error
from dependancies.Classification_v1 import *
from object.ConsumerRawData import ConsumerRawData
from object.ProducerRawData import ProducerRawData
from uuid import uuid4

from object.helpers import todict

# import sqlalchemy
from sqlalchemy import create_engine

# import kafka
from confluent_kafka import DeserializingConsumer, SerializingProducer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer, AvroSerializer
from confluent_kafka.serialization import StringDeserializer, StringSerializer

#help fonctions
import pandas as pd
import warnings
from pandas.core.common import SettingWithCopyWarning
# import fastApi
from fastapi import FastAPI

# fastapi config
app = FastAPI()

# kafka consumer config
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_NAME_CONSUME = "ProducerRawData"

schema_registry_client = SchemaRegistryClient({"url": "http://localhost:8085"})
consumer_conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                 "key.deserializer": StringDeserializer("utf_8"),
                 "value.deserializer": AvroDeserializer(schema_str=None,  # the schema should be fetched from the registry
                                                        schema_registry_client=schema_registry_client
                                                ),
                 "group.id": "classification",
                 "auto.offset.reset": "earliest"}

consumer = DeserializingConsumer(consumer_conf)
consumer.subscribe(["rawdata"])

TOPIC_NAME_PRODUCE = "CLASSIFICATION"

# --- Producing part ---

producer_conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                 "key.serializer": StringSerializer("utf_8"),
                 "value.serializer": AvroSerializer(schema_str=ProducerRawData.schema,
                                             schema_registry_client=schema_registry_client,
                                                    to_dict=todict)}

producer = SerializingProducer(producer_conf)

#---------------------------------------------------------------------------------------------------------------------------

@app.get("/")
def runing():
    return "Hi classification microserivce is runing"


@app.get("/classification")
def classification():
        print("begin")
        # Dict contain data for each participant 
        data ={}
        # Lestener consumer kafka
        while True:
            try:
                try: 
                    msg = consumer.poll(0.0)
                except : 
                    return "plz create RAWDATA TOPIC"

                if msg is None:
                    continue
                message = msg.value()
                if message is not None:
                    # cast data to consumer object
                    rowdata = ConsumerRawData(message)
                    # check if participant_virtual_id exist in data else create it
                    if rowdata.get_participant_virtual_id() in data.keys() : 
                        data.get(rowdata.get_participant_virtual_id()).append(rowdata)
                    else : 
                        data[rowdata.get_participant_virtual_id()] = [rowdata]
                    #print(f"Consumed message: {message}")
                # check if the number of values is reached
                key = rate_done(data)

                if ( key != "No"):
                    # run preprocessing algo
                    df = run(get_df(data.get(key)), key)
                    #delete values from memory
                    data[key] = []
                    # save data in kafka topic
                    save_data(df)

            except KeyboardInterrupt:
                break


        return True


# TODO:: send this function to other file helper


def save_data(df): 

    print("begin save")
    for ind in df.index:
                #producer.poll(0.0)
                msg = ProducerRawData(dict(
                    #Handle the case with alphanumeric id_number
                    participant_virtual_id=str(df['participant_virtual_id'][ind]),
                    timestamp=str(df['timestamp'][ind]) if type(df['timestamp'][ind]) != type(None) else str(''),
                    activity=str(df['activity'][ind]),
                ))
                producer.produce(topic=TOPIC_NAME_PRODUCE,key=str(uuid4()),value=msg)
                print(f"Produced message: {msg.dict()}")
                producer.flush()

def get_df(l): 
    # from list of object to dataframe
    data = pd.DataFrame([],columns =['participant_virtual_id', 'time', 'PM25', 'PM10', 'PM1',
       'Temperature', 'Humidity', 'NO2', 'BC', 'Speed', 'activity',
       'event', 'imputed_NO2', 'imputed_temperature', 'imputed_humidity', 'imputed_PM1',
        'imputed_PM10', 'imputed_PM25', 'imputed_BC'] )
    for elm in l : 
        data = data.append(elm.__dict__, ignore_index=True)
    
    data["PM2.5"] = data["PM25"]
    data["PM1.0"] = data["PM1"]
    del data["PM25"]
    del data["PM1"]
    return True

def rate_done(data): 
    for key in data.keys():
        if (len(data.get(key)) == 60) : 
            print("rate done")
            return key
        else : 
            return "No"

def run(df,id_p): 
    
    d = classification_v1(df, id_p)
    return True