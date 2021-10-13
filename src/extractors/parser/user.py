import json
import pickle

import pandas as pd
import numpy as np

import datetime
class User:
    def __init__(self, path: str):
        with open(path, 'r') as fp:
            self._user_log = json.load(fp)
            
        self._session_id = self._user_log['session']['session_id']
        self._learner_id = float(self._user_log['session']['learner_id'].replace('NaN', ''))
        self._event_df = pd.DataFrame()
        
    def createEventDataFrame(self):
        event_df = []
        for event in self._user_log['events']:
            timestamp = event['timestamp']
            timestamp = datetime.datetime.fromtimestamp(timestamp / 1e3)
            year = timestamp.year
            month = timestamp.month
            day = timestamp.day
            hour = timestamp.hour
            minute = timestamp.minute
            second = timestamp.second
            
            event_name = event['event'].replace('capacitorLabBasics.', '')
            event_name = event_name.replace('capacitanceScreen.', '')
            
            event_type = event['data']['eventType']
            if 'parameters' in event['data']:
                if 'method' in event['data']['parameters']:
                    method_name = event['data']['parameters']['method']
                else:
                    method_name = 'null'
            # parameters_event = event['data']['eventType']
            if 'phetioID' in event['data']:
                phetio_id = event['data']['phetioID']
            else:
                phetio_id = 'null'
            

            data = event['data']
            
            event_df.append([event_name, event_type, method_name, phetio_id, timestamp, year, month, day, hour, minute, second, data])
            
        event_df = pd.DataFrame(event_df)
        event_df.columns = ['event_name', 'event_type', 'method', 'phetio_id', 'timestamp', 'year', 'month', 'day', 'hour', 'minute', 'second', 'data']

        event_df = event_df.sort_values(['year', 'month', 'day', 'hour', 'minute', 'second'])
        
        self._event_df = event_df

    def save(self, version=''):
        name = str(self._session_id) + '_' + str(self._learner_id) + version + '_UserObject.pkl'
        name = '../Objects/users/' + name
        with open(name, 'wb') as fp:
            pickle.dump(self, fp)
        
            
        