# -*- coding: utf-8 -*-

""" avro python class for file: rawdataGPS """

import json
from helpers import default_json_serialize, todict
from typing import Union


class rawdataGPS(object):

    schema = """
    {
        "type": "record",
        "namespace": "Polluscope",
        "name": "rawdataGPS",
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
                "name": "time",
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

    def __init__(self, obj: Union[str, dict, 'rawdataGPS']) -> None:
        if isinstance(obj, str):
            obj = json.loads(obj)

        elif isinstance(obj, type(self)):
            obj = obj.__dict__

        elif not isinstance(obj, dict):
            raise TypeError(
                f"{type(obj)} is not in ('str', 'dict', 'rawdataGPS')"
            )

        self.set_id(obj.get('id', None))

        self.set_tablet_id(obj.get('tablet_id', None))

        self.set_time(obj.get('time', None))

        self.set_lat(obj.get('lat', None))

        self.set_lon(obj.get('lon', None))

    def dict(self):
        return todict(self)

    def set_id(self, value: int) -> None:

        if isinstance(value, int):
            self.id = value
        else:
            raise TypeError("field 'id' should be type int")

    def get_id(self) -> int:

        return self.id

    def set_tablet_id(self, value: int) -> None:

        if isinstance(value, int):
            self.tablet_id = value
        else:
            raise TypeError("field 'tablet_id' should be type int")

    def get_tablet_id(self) -> int:

        return self.tablet_id

    def set_time(self, value: str) -> None:

        if isinstance(value, str):
            self.time = value
        else:
            raise TypeError("field 'time' should be type str")

    def get_time(self) -> str:

        return self.time

    def set_lat(self, value: float) -> None:

        if isinstance(value, float):
            self.lat = value
        else:
            raise TypeError("field 'lat' should be type float")

    def get_lat(self) -> float:

        return self.lat

    def set_lon(self, value: float) -> None:

        if isinstance(value, float):
            self.lon = value
        else:
            raise TypeError("field 'lon' should be type float")

    def get_lon(self) -> float:

        return self.lon

    def serialize(self) -> None:
        return json.dumps(self, default=default_json_serialize)
