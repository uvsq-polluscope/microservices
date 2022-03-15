import pandas as pd
from uuid import uuid4

from sqlalchemy import Table, Column, MetaData, Integer, Computed
from sqlalchemy import create_engine

from confluent_kafka import DeserializingConsumer, SerializingProducer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer, AvroSerializer
from confluent_kafka.serialization import StringDeserializer, StringSerializer

from helpers import todict
from Polluscope.rawdataSMD import rawdataSMD

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_NAME = "rawdataSMD"

schema_registry_client = SchemaRegistryClient({"url": "http://localhost:8085"})

# --- Producing part ---

producer_conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                 "key.serializer": StringSerializer("utf_8"),
                 "value.serializer": AvroSerializer(schema_str=rawdataSMD.schema,  # the schema will be registered in the registry
                                                    schema_registry_client=schema_registry_client,
                                                    to_dict=todict)}

producer = SerializingProducer(producer_conf)

def app():
    db_string = "postgresql://dwaccount:password@127.0.0.1:5435/dwaccount"
    db = create_engine(db_string, echo=True)

    original_data = pd.read_sql('SELECT participant_virtual_id, time, lat, lon, hilbert, activity FROM clean_gps_with_activity limit 20', db)
    print(original_data)
    for ind in original_data.index:
        msg = rawdataSMD(dict(
            participant_virtual_id=str(original_data['participant_virtual_id'][ind]),
            time=str(original_data['time'][ind]),
            lat=float(original_data['lat'][ind]),
            lon=float(original_data['lon'][ind]),
            hilbert=int(original_data['hilbert'][ind]),
            activity=str(original_data['activity'][ind])
        ))
        print(msg)
        producer.produce(topic=TOPIC_NAME,key=str(uuid4()),value=msg)
        print(f"Produced message: {msg.dict()}")
        producer.flush()

if __name__ == "__main__":
   app()