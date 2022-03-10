# -*- coding: utf-8 -*-

""" avro python class for file: Move_Stop_Detection """

import json
from helpers import default_json_serialize, todict
from typing import Union


class Move_Stop_Detection(object):

    schema = """
    {
        "type": "record",
        "namespace": "Polluscope",
        "name": "Move_Stop_Detection",
        "fields": [
            {
                "name": "participant_virtual_id",
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
            },
            {
                "name": "hilbert",
                "type": "float"
            }
        ]
    }
    """

    def __init__(self, obj: Union[str, dict, 'Move_Stop_Detection']) -> None:
        if isinstance(obj, str):
            obj = json.loads(obj)

        elif isinstance(obj, type(self)):
            obj = obj.__dict__

        elif not isinstance(obj, dict):
            raise TypeError(
                f"{type(obj)} is not in ('str', 'dict', 'Move_Stop_Detection')"
            )

        self.set_participant_virtual_id(obj.get('participant_virtual_id', None))

        self.set_time(obj.get('time', None))

        self.set_lat(obj.get('lat', None))

        self.set_lon(obj.get('lon', None))

        self.set_hilbert(obj.get('hilbert', None))

    def dict(self):
        return todict(self)

    def set_participant_virtual_id(self, value: int) -> None:

        if isinstance(value, int):
            self.participant_virtual_id = value
        else:
            raise TypeError("field 'participant_virtual_id' should be type int")

    def get_participant_virtual_id(self) -> int:

        return self.participant_virtual_id

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

    def set_hilbert(self, value: float) -> None:

        if isinstance(value, float):
            self.hilbert = value
        else:
            raise TypeError("field 'hilbert' should be type float")

    def get_hilbert(self) -> float:

        return self.hilbert

    def serialize(self) -> None:
        return json.dumps(self, default=default_json_serialize)
