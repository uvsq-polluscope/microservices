
import pandas as pd
import json
from helpers import default_json_serialize, todict
from datetime import datetime


class ConsumerRawDataGPS(object):

    def __init__(self, obj) -> None:

        self.set_id(obj.get('id', None))

        self.set_participant_virtual_id(
            obj.get('participant_virtual_id', None))

        self.set_tablet_id(obj.get('tablet_id', None))

        self.set_time(obj.get('time', None))

        self.set_lat(obj.get('lat', None))

        self.set_lon(obj.get('lon', None))

    def dict(self):
        return todict(self)

    def set_time(self, value: str) -> None:

        if isinstance(value, str):
            dateTime = datetime.strptime(
                value.replace("+00:00", ""), '%Y-%m-%d %H:%M:%S')
            self.time = dateTime
        else:
            raise TypeError("field 'timestamp' should be type str")

    def get_time(self) -> datetime:

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

    def set_tablet_id(self, value: int) -> None:

        if isinstance(value, int):
            self.tablet_id = value
        else:
            raise TypeError("field 'tablet_id' should be type int")

    def get_tablet_id(self) -> int:

        return self.tablet_id

    def set_id(self, value: int) -> None:

        if isinstance(value, int):
            self.id = value
        else:
            raise TypeError("field 'id' should be type float")

    def get_id(self) -> int:

        return self.id

    def set_participant_virtual_id(self, value: str) -> None:

        if isinstance(value, str):
            self.participant_virtual_id = value
        else:
            raise TypeError(
                "field 'participant_virtual_id' should be type str")

    def get_participant_virtual_id(self) -> str:

        return self.participant_virtual_id

    def serialize(self) -> None:
        return json.dumps(self, default=default_json_serialize)
