# -*- coding: utf-8 -*-

""" avro python class for file: rawdata """

import json
from object.helpers import default_json_serialize, todict
from typing import Union


class ConsumerClassification(object):

    schema = """
    {
        "type": "record",
        "namespace": "object",
        "name": "ProducerRawData",
        "fields": [
            {
                "name": "participant_virtual_id",
                "type": "string"
            },
            {
                "name": "time",
                "type": "string"
            },
            {
                "name": "activity",
                "type": "string"
            }
           
        ]
    }
    """

    def __init__(self, obj: Union[str, dict, 'rawdata']) -> None:
        if isinstance(obj, str):
            obj = json.loads(obj)

        elif isinstance(obj, type(self)):
            obj = obj.__dict__

        elif not isinstance(obj, dict):
            raise TypeError(
                f"{type(obj)} is not in ('str', 'dict', 'rawdata')"
            )

        self.set_participant_virtual_id(
            obj.get('participant_virtual_id', None))

        self.set_time(obj.get('time', None))

        self.set_activity(obj.get('activity', None))

    def dict(self):
        return todict(self)

    def set_participant_virtual_id(self, value: str) -> None:

        if isinstance(value, str):
            self.participant_virtual_id = value
        else:
            raise TypeError(
                "field 'participant_virtual_id' should be type int")

    def get_participant_virtual_id(self) -> str:

        return self.participant_virtual_id

    def set_time(self, value: str) -> None:

        if isinstance(value, str):
            self.time = value
        else:
            raise TypeError("field 'time' should be type str")

    def get_time(self) -> str:

        return self.time

    def set_activity(self, value: str) -> None:

        if isinstance(value, str):
            self.activity = value
        else:
            raise TypeError("field 'activity' should be type str")

    def get_activity(self) -> str:

        return self.activity

    def serialize(self) -> None:
        return json.dumps(self, default=default_json_serialize)
