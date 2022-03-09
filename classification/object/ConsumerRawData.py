
import pandas as pd
from datetime import datetime

class ConsumerRawData(object): 


    def __init__(self, obj) -> None:

        self.set_participant_virtual_id(obj.get('participant_virtual_id', None))

        self.set_time(obj.get('time', None))

        self.set_PM25(obj.get('PM25', None))

        self.set_PM10(obj.get('PM10', None))

        self.set_PM1(obj.get('PM1', None))

        self.set_Temperature(obj.get('Temperature', None))

        self.set_Humidity(obj.get('Humidity', None))

        self.set_NO2(obj.get('NO2', None))

        self.set_BC(obj.get('BC', None))

        self.set_activity(obj.get('activity', None))

        self.set_event(obj.get('event', None))
        
        self.set_imputed_NO2(obj.get('imputed_NO2', None))
        
        self.set_imputed_temperature(obj.get('imputed_temperature', None))
        
        self.set_imputed_humidity(obj.get('imputed_humidity', None))
        
        self.set_imputed_PM1(obj.get('imputed_PM1', None))
        
        self.set_imputed_PM10(obj.get('imputed_PM10', None))
        
        self.set_imputed_PM25(obj.get('imputed_PM25', None))
        
        self.set_imputed_BC(obj.get('imputed_BC', None))
        
        self.set_Speed(obj.get('Speed', None))


    def set_participant_virtual_id(self, value: str) -> None:

        if isinstance(value, str):
            self.participant_virtual_id = value
        else:
            raise TypeError("field 'participant_virtual_id' should be type int")

    def get_participant_virtual_id(self) -> str:

        return self.participant_virtual_id

    def set_time(self, value: str) -> None:

        if isinstance(value, str):            
            dateTime= datetime.strptime(value,'%Y-%m-%d %H:%M:%S')
            self.time = dateTime
        else:
            raise TypeError("field 'time' should be type str")

    def get_time(self) -> str:

        return self.time

    def set_PM25(self, value: float) -> None:

        if isinstance(value, float):
            self.PM25 = value
        else:
            raise TypeError("field 'PM25' should be type float")

    def get_PM25(self) -> float:

        return self.PM25

    def set_PM10(self, value: float) -> None:

        if isinstance(value, float):
            self.PM10 = value
        else:
            raise TypeError("field 'PM10' should be type float")

    def get_PM10(self) -> float:

        return self.PM10

    def set_PM1(self, value: float) -> None:

        if isinstance(value, float):
            self.PM1 = value
        else:
            raise TypeError("field 'PM1' should be type float")

    def get_PM1(self) -> float:

        return self.PM1

    def set_Temperature(self, value: float) -> None:

        if isinstance(value, float):
            self.Temperature = value
        else:
            raise TypeError("field 'Temperature' should be type float")

    def get_Temperature(self) -> float:

        return self.Temperature

    def set_Humidity(self, value: float) -> None:

        if isinstance(value, float):
            self.Humidity = value
        else:
            raise TypeError("field 'Humidity' should be type float")

    def get_Humidity(self) -> float:

        return self.Humidity

    def set_NO2(self, value: float) -> None:

        if isinstance(value, float):
            self.NO2 = value
        else:
            raise TypeError("field 'NO2' should be type float")

    def get_NO2(self) -> float:

        return self.NO2

    def set_BC(self, value: float) -> None:

        if isinstance(value, float):
            self.BC = value
        else:
            raise TypeError("field 'BC' should be type float")

    def get_BC(self) -> float:

        return self.BC
        
    def set_activity(self, value: str) -> None:

        if isinstance(value, str):
            self.activity = value
        else:
            raise TypeError("field 'activity' should be type str")

    def get_activity(self) -> str:

        return self.activity

    def set_event(self, value: str) -> None:
        print("--------------------")
        print(isinstance(value, bool))
        print(value)
        print(type(value))
        if isinstance(value, str):
            self.event = value
        else:
            raise TypeError("field 'event' should be type str")

    def get_event(self) -> str:

        return self.event
    
    def set_imputed_NO2(self, value: bool) -> None:
    

        if isinstance(value, bool):
            self.imputed_NO2 = value
        else:
            raise TypeError("field 'imputed_NO2' should be type bool")

    def get_imputed_NO2(self) -> bool:
        
        return self.imputed_NO2
    def set_imputed_temperature(self, value: bool) -> None:

        if isinstance(value, bool):
            self.imputed_temperature = value
        else:
            raise TypeError("field 'imputed_NO2' should be type bool")

    def get_imputed_temperature(self) -> bool:

        return self.imputed_temperature

    def set_imputed_humidity(self, value: bool) -> None:

        if isinstance(value, bool):
            self.imputed_humidity = value
        else:
            raise TypeError("field 'imputed_humidity' should be type bool")

    def get_imputed_humidity(self) -> bool:

        return self.imputed_humidity
    
    def set_imputed_PM1(self, value: bool) -> None:

        if isinstance(value, bool):
            self.imputed_PM1 = value
        else:
            raise TypeError("field 'imputed_humidity' should be type bool")

    def get_imputed_PM1(self) -> bool:

        return self.imputed_PM1
    
    def set_imputed_PM10(self, value: bool) -> None:

        if isinstance(value, bool):
            self.imputed_PM10 = value
        else:
            raise TypeError("field 'imputed_PM10' should be type bool")

    def get_imputed_PM10(self) -> bool:

        return self.imputed_PM10

    def set_imputed_PM25(self, value: bool) -> None:

        if isinstance(value, bool):
            self.imputed_PM25 = value
        else:
            raise TypeError("field 'imputed_PM25' should be type str")

    def get_imputed_PM25(self) -> bool:

        return self.imputed_PM25
    
    def set_imputed_BC(self, value: bool) -> None:

        if isinstance(value, bool):
            self.imputed_BC = value
        else:
            raise TypeError("field 'imputed_BC' should be type str")

    def get_imputed_BC(self) -> bool:

        return self.imputed_BC

    def set_Speed(self, value: float) -> None:

        if isinstance(value, float):
            self.Speed = value
        else:
            raise TypeError("field 'Speed' should be type str")

    def get_Speed(self) -> float:

        return self.Speed

