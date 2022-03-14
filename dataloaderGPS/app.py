import pandas as pd
from uuid import uuid4

from sqlalchemy import Table, Column, MetaData, Integer, Computed
from sqlalchemy import create_engine

from confluent_kafka import DeserializingConsumer, SerializingProducer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer, AvroSerializer
from confluent_kafka.serialization import StringDeserializer, StringSerializer

from helpers import todict
from Polluscope.rawdataGPS import rawdataGPS
import os

KAFKA_BOOTSTRAP_SERVERS = os.environ['KAFKA_BOOTSTRAP_SERVERS']
TOPIC_NAME = "rawdataGPS"

schema_registry_client = SchemaRegistryClient({"url": os.environ['SCHEMA_REGISTRY_CLIENT']})

# --- Producing part ---

producer_conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                 "key.serializer": StringSerializer("utf_8"),
                 "value.serializer": AvroSerializer(schema_str=rawdataGPS.schema,  # the schema will be registered in the registry
                                                    schema_registry_client=schema_registry_client,
                                                    to_dict=todict)}

producer = SerializingProducer(producer_conf)

# This file is used to load data from database into the specified kafka topic
def app():
    db_string = os.environ["DATABASE"]
    db = create_engine(db_string, echo=True)
    # You can change the request and the LIMIT instruction as you want
    original_data = pd.read_sql('''SELECT "tabletPositionApp".id, "participant".participant_virtual_id, "tabletPositionApp".tablet_id, "tabletPositionApp".timestamp, "tabletPositionApp".lat, "tabletPositionApp".lon FROM "tablet","tabletPositionApp","campaignParticipantKit","kit","participant" WHERE "tabletPositionApp"."tablet_id"="kit"."tablet_id" and "kit"."id"="campaignParticipantKit"."kit_id" and "campaignParticipantKit"."participant_id"="participant"."id" and "tabletPositionApp"."timestamp" between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date" and "tabletPositionApp"."tablet_id"="tablet"."id" and "participant".participant_virtual_id = '9999961' LIMIT 20''', db)

    print(original_data)
    for ind in original_data.index:
        msg = rawdataGPS(dict(
            id=int(original_data['id'][ind]),
            participant_virtual_id=str(original_data['participant_virtual_id'][ind]),
            tablet_id=int(original_data['tablet_id'][ind]),
            time=str(original_data['timestamp'][ind]),
            lat=float(original_data['lat'][ind]),
            lon=float(original_data['lon'][ind]),
        ))
        producer.produce(topic=TOPIC_NAME,key=str(uuid4()),value=msg)
        print(f"Produced message: {msg.dict()}")
        producer.flush()

if __name__ == "__main__":
   app()
