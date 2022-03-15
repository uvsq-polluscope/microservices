# -*- coding: utf-8 -*-

""" avro python class for file: Move_Stop_Detection """

import json
from object.helpers import default_json_serialize, todict
from typing import Union


class stopMoveDetectionTopic(object):

    schema = """
    {
        "type": "record",
        "namespace": "object",
        "name": "stopMoveDetectionTopic",
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
            },
            {
                "name": "detected_label",
                "type": "string"
            },
            {
                "name": "userId",
                "type": "string"
            } ,
            {
                "name": "target",
                "type": "string"
            },
            {
                "name": "pred",
                "type": "string"
            },
            {
                "name": "prediction",
                "type": "string"
            }
        ]
    }
    """

    def __init__(self, obj: Union[str, dict, 'stopMoveDetectionTopic']) -> None:
        if isinstance(obj, str):
            obj = json.loads(obj)

        elif isinstance(obj, type(self)):
            obj = obj.__dict__

        elif not isinstance(obj, dict):
            raise TypeError(
                f"{type(obj)} is not in ('str', 'dict', 'stopMoveDetectionTopic')"
            )

        self.set_participant_virtual_id(obj.get('participant_virtual_id', None))

        self.set_time(obj.get('time', None))

        self.set_activity(obj.get('activity', None))

        self.set_detected_label(obj.get('detected_label', None))
        
        self.set_userId(obj.get('userId', None))

        self.set_target(obj.get('target', None))

        self.set_pred(obj.get('pred', None))

        self.set_prediction(obj.get('prediction', None))

    def dict(self):
        return todict(self)

    def set_participant_virtual_id(self, value: str) -> None:

        if isinstance(value, str):
            self.participant_virtual_id = value
        else:
            raise TypeError("field 'participant_virtual_id' should be type str")

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

    def set_detected_label(self, value: str) -> None:

        if isinstance(value, str):
            self.detected_label = value
        else:
            raise TypeError("field 'detected_label' should be type str")

    def get_detected_label(self) -> str:

        return self.detected_label

    def set_userId(self, value: str) -> None:

        if isinstance(value, str):
            self.userId = value
        else:
            raise TypeError("field 'userId' should be type string")

    def get_userId(self) -> str:

        return self.userId 

    def set_target(self, value: str) -> None:

        if isinstance(value, str):
            self.target = value
        else:
            raise TypeError("field 'target' should be type string")

    def get_pred(self) -> str:

        return self.pred

    def set_pred(self, value: str) -> None:

        if isinstance(value, str):
            self.pred = value
        else:
            raise TypeError("field 'pred' should be type str")

    def get_pred(self) -> str:

        return self.pred

    def set_prediction(self, value: str) -> None:

        if isinstance(value, str):
            self.prediction = value
        else:
            raise TypeError("field 'prediction' should be type string")

    def get_prediction(self) -> str:

        return self.prediction

    def serialize(self) -> None:
        return json.dumps(self, default=default_json_serialize)