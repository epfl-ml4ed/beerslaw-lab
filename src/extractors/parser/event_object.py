import numpy as np
import datetime

class Event:
    """Creates an event dataframe where each row contains information about an events
    """
    def __init__(self, row: np.array):
        self._message_index = row['message_index']
        
        self._timestamp = row['timestamp']
        self._year = row['year']
        self._month = row['month']
        self._day = row['day']
        self._hour = row['hour']
        self._minute = row['minute']
        self._second = row['second']
        
        self._phetio_id = row['phetio_id']
        self._name = row['event_name']
        self._type = row['event_type']
        self._component = row['event_component']
        self._params = row['event_params']
        
        if 'event_children' in row:
            self._children = row['event_children']
        
    def get_message_index(self) -> float:
        return self._message_index
    
    def get_timestamp(self) -> datetime.datetime:
        return self._timestamp
    
    def get_year(self) -> int:
        return self._year
    
    def get_month(self) -> int:
        return self._month
    
    def get_day(self) -> int:
        return self._day
    
    def get_hour(self) -> int:
        return self._hour
    
    def get_minute(self) -> int:
        return self._minute
    
    def get_second(self) -> int:
        return self._second
    
    def get_phetio_id(self) -> str:
        return self._phetio_id
    
    def get_name(self) -> str:
        return self._name
    
    def get_type(self) -> str:
        return self._type
    
    def get_component(self) -> str:
        return self._component
    
    def get_params(self) -> dict:
        return self._params
    
    def get_children(self) -> dict:
        return self._children
    
    
        

        
        