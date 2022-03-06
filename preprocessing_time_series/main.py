
#import locale file
from ast import Delete, Pass
from dependancies.preprocessign import *
from object.ConsumerRawData import ConsumerRawData
from object.ProducerRawData import ProducerRawData

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

# import fastApi
from fastapi import FastAPI

# fastapi config
app = FastAPI()

# kafka consumer config
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_NAME_CONSUME = "RAWDATA"

schema_registry_client = SchemaRegistryClient({"url": "http://localhost:8085"})
consumer_conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                 "key.deserializer": StringDeserializer("utf_8"),
                 "value.deserializer": AvroDeserializer(schema_str=None,  # the schema should be fetched from the registry
                                                        schema_registry_client=schema_registry_client
                                                ),
                 "group.id": "test_group",
                 "auto.offset.reset": "earliest"}

consumer = DeserializingConsumer(consumer_conf)
consumer.subscribe([TOPIC_NAME_CONSUME])

TOPIC_NAME_PRODUCE = "PREPROCESSING"

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
# TODO:: add get_df method
              
def rate_done(data): 
    for key in data.keys():
        if (len(data.get(key)) == 60) : 
            return key
        else : 
            return "No"

def save_data(df) : 
    for ind in df.index:
                #producer.poll(0.0)
                msg = ProducerRawData(dict(
                    #Handle the case with alphanumeric id_number
                    participant_virtual_id=str(df['participant_virtual_id'][ind]),
                    time=str(df['time'][ind]) if type(original_data['time'][ind]) != type(None) else str(''),
                    PM25=float(df['PM2.5'][ind]) if type(df['PM2.5'][ind]) != type(None) else float(0.0),
                    PM10=float(df['PM10'][ind]) if type(df['PM10'][ind]) != type(None) else float(0.0),
                    PM1=float(df['PM1.0'][ind]) if type(df['PM1.0'][ind]) != type(None) else float(0.0),
                    Temperature=float(original_data['Temperature'][ind]) if type(df['Temperature'][ind]) != type(None) else float(0.0),
                    Humidity=float(df['Humidity'][ind]) if type(df['Humidity'][ind]) != type(None) else float(0.0),
                    NO2=float(df['NO2'][ind]) if type(df['NO2'][ind]) != type(None) else float(0.0),
                    BC=float(df['BC'][ind]) if type(df['BC'][ind]) != type(None) else float(0.0),
                    activity=str(df['activity'][ind]),
                    event=str(df['event'][ind]) if type(df['event'][ind]) != type(None) else str('None'),        
                    imputed_NO2=str(df['imputed NO2'][ind]) if type(df['imputed NO2'][ind]) != type(None) else bool(False),   
                    imputed_temperature=str(df['imputed temperature'][ind]) if type(df['imputed temperature'][ind]) != type(None) else                         bool(False),   
                    imputed_humidity=str(df['imputed humidity'][ind]) if type(df['imputed humidity'][ind]) != type(None) else bool(False),
                    imputed_PM1=str(df['imputed PM1.0'][ind]) if type(df['imputed PM1.0'][ind]) != type(None) else bool(False),
                    imputed_PM10=str(df['imputed PM10'][ind]) if type(df['imputed PM10'][ind]) != type(None) else bool(False),
                    imputed_PM25=str(df['imputed PM25'][ind]) if type(df['imputed PM25'][ind]) != type(None) else bool(False),
                    imputed_BC=str(df['imputed BC'][ind]) if type(df['imputed BC'][ind]) != type(None) else bool(False),
                    Speed=float(df['Speed'][ind]) if type(df['Speed'][ind]) != type(None) else float(0.0)
                ))
                producer.produce(topic=TOPIC_NAME_PRODUCE,key=str(uuid4()),value=msg)
                print(f"Produced message: {msg.dict()}")
                producer.flush()

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


