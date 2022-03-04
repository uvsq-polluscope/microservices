
#import locale file
from ast import Delete, Pass
from dependancies.preprocessign import *
from object.ConsumerRawData import ConsumerRawData


# import sqlalchemy
from sqlalchemy import create_engine

# import kafka
from confluent_kafka import DeserializingConsumer, SerializingProducer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer, AvroSerializer
from confluent_kafka.serialization import StringDeserializer, StringSerializer

#help fonctions
import pandas as pd

# import fastApi
from fastapi import FastAPI

# fastapi config
app = FastAPI()

# kafka consumer config
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_NAME = "rawdata"

schema_registry_client = SchemaRegistryClient({"url": "http://localhost:8085"})
consumer_conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                 "key.deserializer": StringDeserializer("utf_8"),
                 "value.deserializer": AvroDeserializer(schema_str=None,  # the schema should be fetched from the registry
                                                        schema_registry_client=schema_registry_client
                                                ),
                 "group.id": "test_group",
                 "auto.offset.reset": "earliest"}

consumer = DeserializingConsumer(consumer_conf)
consumer.subscribe([TOPIC_NAME])

#---------------------------------------------------------------------------------------------------------------------------


@app.get("/")
def runing():
    return "Hi preprossing microserivce is runing"


@app.get("/preprocessing")
def preprocessing():
    try: 
        # Dict contain data for each participant 
        data ={}
        # Lestener consumer kafka
        while True:
            try:
                msg = consumer.poll(1.0)
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
                    print(f"Consumed message: {message}")
                # check if the number of values is reached
                key = rate_done(data)

                if ( key != "No"):
                    # run preprocessing algo
                    df = run(get_df(data.get(key)))
                    #delete values from memory
                    data[key] = []
                    # save data in kafka topic
                    save_data(df)

            except KeyboardInterrupt:
                break

        consumer.close()

        return True
    except: 
        return "Erreur"

# TODO:: send this function to other file helper
# TODO:: add save method
# TODO:: add get_df method
              
def rate_done(data): 
    for key in data.keys():
        if (len(data.get(key)) == 60) : 
            return key
        else : 
            return "No"

def save_data(df) : 
    # save data in kafka topic
    pass

def get_df(l) : 
    # from list of object to dataframe
    pass

def run(df): 
    # run preprocessing algo
    try: 
        if 'NO2' in df.columns and len(df.dropna(subset=['NO2']))>0:
            for j in range(0,3):
                df.loc[ df.dropna(subset=['NO2'])['NO2'].first_valid_index(), 'NO2'] = np.nan
        df=data_pre_processing(df)
        return df
    except:
        print("erreur")
        return pd.DataFrame()


