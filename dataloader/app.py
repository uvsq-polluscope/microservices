import pandas as pd
from uuid import uuid4

from sqlalchemy import Table, Column, MetaData, Integer, Computed
from sqlalchemy import create_engine

from confluent_kafka import DeserializingConsumer, SerializingProducer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer, AvroSerializer
from confluent_kafka.serialization import StringDeserializer, StringSerializer

from helpers import todict
from Polluscope.rawdata import rawdata

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_NAME = "rawdata"

schema_registry_client = SchemaRegistryClient({"url": "http://localhost:8085"})

# --- Producing part ---

producer_conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                 "key.serializer": StringSerializer("utf_8"),
                 "value.serializer": AvroSerializer(schema_str=rawdata.schema,  # the schema will be registered in the registry
                                                    schema_registry_client=schema_registry_client,
                                                    to_dict=todict)}

producer = SerializingProducer(producer_conf)

id_request = "SELECT participant_id, kit_id FROM public.\"campaignParticipantKit\";"

def increment_ids(kit_id, participant_id):
    request = '''
            select distinct(res.*)
            from (
            select r1."participant_virtual_id", r1."time", r1."PM2.5", r2."PM10", r3."PM1.0", r4."Temperature", r5."Humidity", r6."NO2", r7."BC", r11."vitesse(m/s)", r8."activity"
            , r9."event"
            from (
            select  "participant"."participant_virtual_id",
                "canarinMeasure"."timestamp" AS "time",
            "canarinMeasure"."value_num" AS "PM2.5",
            "kit"."id" as "kit_id", "participant"."id" as "participant_id"
            from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
            where "canarinMeasure"."canarin_id"="kit"."canarin_id"
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "canarinMeasure"."timestamp"
            between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
            and "canarinMeasure"."canarin_id"="canarin"."id"
            and "canarinMeasure"."type_id" = 7
            ) as r1
            Left join (
            select  "canarinMeasure"."timestamp" AS "time",
            "canarinMeasure"."value_num" AS "PM10",
            "kit"."id" as "kit_id", "participant"."id" as "participant_id"
            from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
            where "canarinMeasure"."canarin_id"="kit"."canarin_id"
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "canarinMeasure"."timestamp"
            between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
            and "canarinMeasure"."canarin_id"="canarin"."id"
            and "canarinMeasure"."type_id" = 8
            ) as r2 on date_trunc('minute',r1."time")=date_trunc('minute',r2."time")
            and "r2"."kit_id"=r1."kit_id" and "r2"."participant_id"=r1."participant_id" 
            Left join (
            select  "canarinMeasure"."timestamp" AS "time",
            "canarinMeasure"."value_num" AS "PM1.0",
                "kit"."id" as "kit_id", "participant"."id" as "participant_id"
            from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
            where "canarinMeasure"."canarin_id"="kit"."canarin_id"
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "canarinMeasure"."timestamp"
            between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
            and "canarinMeasure"."canarin_id"="canarin"."id"
            and "canarinMeasure"."type_id" = 9
            ) as r3 on date_trunc('minute',r1."time")=date_trunc('minute',r3."time")
            and "r3"."kit_id"=r1."kit_id" and "r3"."participant_id"=r1."participant_id" 
            Left join (
            select  "canarinMeasure"."timestamp" AS "time",
            "canarinMeasure"."value_num" AS "Temperature",
            "kit"."id" as "kit_id", "participant"."id" as "participant_id"
            from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
            where "canarinMeasure"."canarin_id"="kit"."canarin_id"
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "canarinMeasure"."timestamp"
            between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
            and "canarinMeasure"."canarin_id"="canarin"."id"
            and "canarinMeasure"."type_id" = 4
            ) as r4 on date_trunc('minute',r1."time")=date_trunc('minute',r4."time")
            and "r4"."kit_id"=r1."kit_id" and "r4"."participant_id"=r1."participant_id" 
            Left join (
            select  "canarinMeasure"."timestamp" AS "time",
            "canarinMeasure"."value_num" AS "Humidity",
                "kit"."id" as "kit_id", "participant"."id" as "participant_id"
            from "canarin","canarinMeasure","campaignParticipantKit","kit","participant"
            where "canarinMeasure"."canarin_id"="kit"."canarin_id"
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "canarinMeasure"."timestamp"
            between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
            and "canarinMeasure"."canarin_id"="canarin"."id"
            and "canarinMeasure"."type_id" = 5
            ) as r5 on date_trunc('minute',r1."time")=date_trunc('minute',r5."time")
            and "r5"."kit_id"=r1."kit_id" and "r5"."participant_id"=r1."participant_id" 
            left join
            (select  "cairsensMeasure"."timestamp" AS "time",
            "cairsensMeasure"."level" AS "NO2",
            "kit"."id" as "kit_id", "participant"."id" as "participant_id"
            from "cairsens","cairsensMeasure","campaignParticipantKit","kit","participant"
            where "cairsensMeasure"."cairsens_id"="kit"."cairsens_id"
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "cairsensMeasure"."timestamp"
            between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
            and "cairsensMeasure"."cairsens_id"="cairsens"."id"
            )as r6 on date_trunc('minute',r1."time")=date_trunc('minute',r6."time")
            and "r6"."kit_id"=r1."kit_id" and "r6"."participant_id"=r1."participant_id" 
            left join
            (
            select  "ae51Measure"."timestamp" AS "time",
            "ae51Measure"."bc" AS "BC",
                "kit"."id" as "kit_id", "participant"."id" as "participant_id"
            from "ae51","ae51Measure","campaignParticipantKit","kit","participant"
            where
            "ae51Measure"."ae51_id"="kit"."ae51_id"
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "ae51Measure"."timestamp"
            between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
            )as r7 on date_trunc('minute',r1."time")=date_trunc('minute',r7."time")
            and "r7"."kit_id"=r1."kit_id" and "r7"."participant_id"=r1."participant_id" 

            left join
            (
            select t1."time",
            st_distancesphere( st_point(t1.lon,t1.lat),st_point(t2.lon,t2.lat))/60 "vitesse(m/s)"
            from(
            select DISTINCT ON (res1."time")"time","lat","lon"
            from
            (select date_trunc('minute', "timestamp") AS "time",
            "tabletPositionApp"."lat",
            "tabletPositionApp"."lon"
            from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
            where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
            and "tabletPositionApp"."timestamp"
            between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
            and "tabletPositionApp"."tablet_id"="tablet"."id"
            ) as res1
            ) as t1, (
            select DISTINCT ON (res1."time")"time","lat","lon"
            from
            (select date_trunc('minute', "timestamp") AS "time",
            "tabletPositionApp"."lat",
            "tabletPositionApp"."lon"
            from "tablet","tabletPositionApp","campaignParticipantKit","kit","participant"
            where "tabletPositionApp"."tablet_id"="kit"."tablet_id"
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "kit"."id"='''+str(kit_id)+''' and "participant"."id"='''+str(participant_id)+'''
            and "tabletPositionApp"."timestamp"
            between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
            and "tabletPositionApp"."tablet_id"="tablet"."id"
            ) as res1
            ) as t2
            where t2."time"=t1."time"+ interval '1 minutes' 
            ) as r11 on date_trunc('minute',r1."time")=date_trunc('minute',r11."time")

            left join
            (
            select "tabletActivityApp"."timestamp" AS "time",
            "tabletActivityApp"."activity", 
            lead("timestamp") over (order by "tabletActivityApp".id asc) as next_row,
                "kit"."id" as "kit_id", "participant"."id" as "participant_id"
            from "tablet","tabletActivityApp","campaignParticipantKit","kit","participant"
            where "tabletActivityApp"."tablet_id"="kit"."tablet_id"
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "tabletActivityApp"."timestamp"
            between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
            and "tabletActivityApp"."tablet_id"="tablet"."id"
            ) as r8 on date_trunc('minute',r1."time") between r8."time" and r8.next_row
            and "r8"."kit_id"=r1."kit_id" and "r8"."participant_id"=r1."participant_id" 
            left join
            (
            select "tabletEventApp"."timestamp" AS "time",
            "tabletEventApp"."event",
                "kit"."id" as "kit_id", "participant"."id" as "participant_id"
            from "tablet","tabletEventApp","campaignParticipantKit","kit","participant"
            where "tabletEventApp"."tablet_id"="kit"."tablet_id"
            and "kit"."id"="campaignParticipantKit"."kit_id"
            and "campaignParticipantKit"."participant_id"="participant"."id"
            and "tabletEventApp"."timestamp"
            between "campaignParticipantKit"."start_date" and "campaignParticipantKit"."end_date"
            and "tabletEventApp"."tablet_id"="tablet"."id"
            ) as r9 on date_trunc('minute',r1."time") = date_trunc('minute',r9."time") 
            and "r9"."kit_id"=r1."kit_id" and "r9"."participant_id"=r1."participant_id" 
            )as res
            order by res."time"

            '''
    return request


def app():
        db_string = "postgresql://dwaccount:password@127.0.0.1:5435/dwaccount"
        db = create_engine(db_string, echo=True)
        ids = pd.read_sql(id_request, db)
        for i in range(0,104):
            #print(ids.participant_id[i], " ", ids.kit_id[i])
            original_data = pd.read_sql(increment_ids(ids.kit_id[i],ids.participant_id[i]), db)
            #print(original_data.columns)
            for ind in original_data.index:
                #producer.poll(0.0)
                msg = rawdata(dict(
                    #Handle the case with alphanumeric id_number
                    participant_virtual_id=str(original_data['participant_virtual_id'][ind]),
                    time=str(original_data['time'][ind]) if type(original_data['time'][ind]) != type(None) else str(''),
                    PM25=float(original_data['PM2.5'][ind]) if type(original_data['PM2.5'][ind]) != type(None) else float(0.0),
                    PM10=float(original_data['PM10'][ind]) if type(original_data['PM10'][ind]) != type(None) else float(0.0),
                    PM1=float(original_data['PM1.0'][ind]) if type(original_data['PM1.0'][ind]) != type(None) else float(0.0),
                    Temperature=float(original_data['Temperature'][ind]) if type(original_data['Temperature'][ind]) != type(None) else float(0.0),
                    Humidity=float(original_data['Humidity'][ind]) if type(original_data['Humidity'][ind]) != type(None) else float(0.0),
                    NO2=float(original_data['NO2'][ind]) if type(original_data['NO2'][ind]) != type(None) else float(0.0),
                    BC=float(original_data['BC'][ind]) if type(original_data['BC'][ind]) != type(None) else float(0.0),
                    vitesse=float(original_data['vitesse(m/s)'][ind]) if type(original_data['vitesse(m/s)'][ind]) != type(None) else float(0.0),
                    activity=str(original_data['activity'][ind]),
                    event=str(original_data['event'][ind]) if type(original_data['event'][ind]) != type(None) else str('None')                
                ))
                producer.produce(topic=TOPIC_NAME,key=str(uuid4()),value=msg)
                print(f"Produced message: {msg.dict()}")
                producer.flush()

if __name__ == "__main__":
   app()