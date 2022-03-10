# -*- coding: utf-8 -*-

""" avro python class for file: rawdata """

import json
#from helpers import default_json_serialize, todict
from typing import Union


class ProducerRawDataGPS(object):

    schema = """
    {
        "type": "record",
        "namespace": "object",
        "name": "ProducerRawDataGPS",
        "fields": [
            {
                "name": "id",
                "type": "int"
            },
            {
                "name": "tablet_id",
                "type": "int"
            },
            {
                "name": "timestamp",
                "type": "string"
            },
            {
                "name": "lat",
                "type": "float"
            },
            {
                "name": "lon",
                "type": "float"
            }
        ]
    }
    """

    def __init__(self, obj: Union[str, dict, 'rawdata1']) -> None:
        if isinstance(obj, str):
            obj = json.loads(obj)

        elif isinstance(obj, type(self)):
            obj = obj.__dict__

        elif not isinstance(obj, dict):
            raise TypeError(
                f"{type(obj)} is not in ('str', 'dict', 'rawdata1')"
            )

        self.set_id(obj.get('id', None))

        self.set_tablet_id(obj.get('tablet_id', None))

        self.set_timestamp(obj.get('timestamp', None))

        self.set_lat(obj.get('lat', None))

        self.set_lon(obj.get('lon', None))

    def dict(self):
        return todict(self)

    def set_id(self, value: int) -> None:

        try:
            self.id = int(value)
        except TypeError:
            raise TypeError(
                "field 'id' should be type int")

    def get_id(self) -> int:

        return self.id

    def set_tablet_id(self, value: int) -> None:

        try:
            self.tablet_id = int(value)
        except TypeError:
            raise TypeError(
                "field 'tablet_id' should be type int")

    def get_tablet_id(self) -> int:

        return self.tablet_id

    def set_timestamp(self, value: str) -> None:

        if isinstance(value, str):
            self.timestamp = value
        else:
            raise TypeError("field 'timestamp' should be type str")

    def get_timestamp(self) -> str:

        return self.timestamp

    def set_lat(self, value: float) -> None:

        try:
            self.lat = float(value)
        except TypeError:
            raise TypeError(
                "field 'lat' should be type float")

    def get_lat(self) -> float:

        return self.lat

    def set_lon(self, value: float) -> None:

        try:
            self.lon = float(value)
        except TypeError:
            raise TypeError(
                "field 'lon' should be type float")

    def get_lon(self) -> float:

        return self.lon

    def serialize(self) -> None:
        return json.dumps(self, default=default_json_serialize)
