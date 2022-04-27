import json
import pickle
import datetime
from typing import Tuple

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from extractors.parser.simulation_object import SimObjects
from extractors.parser.checkbox_object import Checkbox
from extractors.parser.value_object import SimCharacteristics
from extractors.parser.event_object import Event
from extractors.parser.user import User

class Simulation:
    """This class reads in the logs of the files via __init__, and then parses the simulation using parse_simulation.
    """

    def __init__(self, paths: list):
        """Loads the log files located in the elements of path, and initialises the simulation

        Args:
            paths (list<str>): list containing the paths of the logs to parse for one user, and one interaction
        """
        # Loading simulation
        self.__load_logs(paths)

        # Initialise simulation parameters
        self.__initialise_simulation_parameters()

    def __load_logs(self, paths: list):
        """ Turns the logs into a dataframe where each row describes an event
        Args:
            paths (list<str>): [list of the paths of the logs to create the simulation from]
        """
        event_df = {}
        for path in paths:
            with open(path, 'r') as fp:
                logs = json.load(fp)
                self._simulation_id = logs['session']['session_id']
                self._learner_id = float(logs['session']['learner_id'].replace('NaN', ''))

                event_df[logs['events'][0]['timestamp']] = []

            old_timestamp = logs['events'][0]['timestamp']
            for event in logs['events']:
                timestamp = event['timestamp']
                assert old_timestamp <= timestamp
                old_timestamp = timestamp

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
                
                event_df[logs['events'][0]['timestamp']].append([event_name, event_type, method_name, phetio_id, timestamp, year, month, day, hour, minute, second, data])

        timestamps = np.sort(list(event_df.keys()))
        df = []
        for k in timestamps:
            df = df + event_df[k]
        df = pd.DataFrame(df)
        df.columns = ['event_name', 'event_type', 'method', 'phetio_id', 'timestamp', 'year', 'month', 'day', 'hour', 'minute', 'second', 'data']
        df = df.sort_values(['year', 'month', 'day', 'hour', 'minute', 'second'])
        self._event_df = df

    def __initialise_simulation_parameters(self):
        self._started = 0
        self._start_timestamp = -1
        self._timestamps_restarts = []
        
        # checkboxes
        self._checkbox_capacitance = Checkbox('checkbox_capacitance', self._simulation_id, 1) #
        self._checkbox_topplate = Checkbox('checkbox_topplate', self._simulation_id, 0) #
        self._checkbox_storedenergy = Checkbox('checkbox_storedenergy', self._simulation_id, 0) #
        self._checkbox_platecharges = Checkbox('checkbox_platecharches', self._simulation_id, 1) #
        self._checkbox_bargraphs = Checkbox('checkbox_bargraphs', self._simulation_id, 1) #
        self._checkbox_electricfields = Checkbox('checkbox_electricfields', self._simulation_id, 0) 
        #
        self._checkbox_currentdirection = Checkbox('checkbox_currentDirection', self._simulation_id, 1) #
        
        # voltmeter
        self._voltmeter = SimObjects('voltmeter', self._simulation_id, {'x':0, 'y':0, 'z':0}, visible=False) #
        self._positive_probe = SimObjects('positive_probe', self._simulation_id, {'x':0, 'y':0, 'z':0})
        self._negative_probe = SimObjects('negative_probe', self._simulation_id, {'x':0, 'y':0, 'z':0})
        
        # characteristics
        self._voltage = SimObjects('voltage', self._simulation_id, 0.0) #
        self._plate_separation = SimObjects('plate_separation', self._simulation_id, 0.006) #
        self._plate_area = SimObjects('plate_area', self._simulation_id, 0.0002) #
        self._circuit = SimCharacteristics('closed_circuit', self._simulation_id, 'BATTERY_CONNECTED') #
        self._current = SimCharacteristics('current', self._simulation_id, 0) #
        
        # circuit
        self._top_openconnection_node = SimObjects('top_openconnection_node', self._simulation_id, {'x':0, 'y':0})
        self._bottom_openconnection_node = SimObjects('bottom_openconnection_node', self._simulation_id, {'x':0, 'y':0})
        self._top_battery_node = SimObjects('top_battery_node', self._simulation_id, {'x':0, 'y':0})
        self._bottom_battery_node = SimObjects('bottom_battery_node', self._simulation_id, {'x':0, 'y':0})
        self._top_node = SimObjects('top_node', self._simulation_id, {'x':0, 'y':0})
        self._bottom_node = SimObjects('bottom_node', self._simulation_id, {'x':0, 'y':0})
        
        # plate characteristics
        self._plate_voltage = SimCharacteristics('plate_voltage', self._simulation_id, 0) #
        self._plate_charge = SimCharacteristics('plate_charge', self._simulation_id, 0) #
        
        # measures
        self._stored_energy = SimCharacteristics('stored_energy', self._simulation_id, 0)#
        self._capacitance = SimCharacteristics('capacitance', self._simulation_id, 0.3)#
        
        # inactivity
        self._inactivity = Checkbox('activity', self._simulation_id, 0) 
        self._inactivity_start = 0
        
        # attentive
        self._attentive = Checkbox('attentive', self._simulation_id, 1)
        
        self._last_timestamp = -1
        self._last_event = ''

        self._timeline = []
        self._timestamps = []
        
#### Getters
    def get_learner_id(self) -> str:
        return self._learner_id
    
    def get_simulation_id(self) -> str:
        return self._simulation_id
    
    def get_checkbox_capacitance(self) -> Checkbox:
        return self._checkbox_capacitance
    
    def get_checkbox_topplate(self) -> Checkbox:
        return self._checkbox_topplate
    
    def get_checkbox_storedenergy(self) -> Checkbox:
        return self._checkbox_storedenergy
    
    def get_checkbox_platecharges(self) -> Checkbox:
        return self._checkbox_platecharges
    
    def get_checkbox_bargraphs(self) -> Checkbox:
        return self._checkbox_bargraphs
    
    def get_checkbox_electricfields(self) -> Checkbox:
        return self._checkbox_electricfields
    
    def get_checkbox_currentdirection(self) -> Checkbox:
        return self._checkbox_currentdirection
    
    def get_attentive(self) -> Checkbox:
        return self._attentive
    
    def get_inactivity(self) -> Checkbox:
        return self._inactivity
    
    def get_voltmeter(self) -> SimObjects:
        return self._voltmeter
    
    def get_positive_probe(self) -> SimObjects:
        return self._positive_probe
    
    def get_negative_probe(self) -> SimObjects:
        return self._negative_probe
    
    def get_voltage(self) -> SimCharacteristics:
        return self._voltage
    
    def get_plate_separation(self) -> SimCharacteristics:
        return self._plate_separation
    
    def get_plate_area(self) -> SimCharacteristics:
        return self._plate_area
    
    def get_circuit(self) -> SimCharacteristics:
        return self._circuit
    
    def get_current(self) -> SimCharacteristics:
        return self._current
    
    def get_top_openconnection_node(self) -> SimObjects:
        return self._top_openconnection_node
    
    def get_bottom_openconnection_node(self) -> SimObjects:
        return self._bottom_openconnection_node
    
    def get_top_battery_node(self) -> SimObjects:
        return self._negati_top_battery_nodeve_probe
    
    def get_bottom_battery_node(self) -> SimObjects:
        return self._bottom_battery_node
    
    def get_top_node(self) -> SimObjects:
        return self._top_node
    
    def get_bottom_node(self) -> SimObjects:
        return self._bottom_node
    
    def get_plate_voltage(self) -> SimCharacteristics:
        return self._plate_voltage
    
    def get_plate_charge(self) -> SimCharacteristics:
        return self._plate_charge
    
    def get_stored_energy(self) -> SimCharacteristics:
        return self._stored_energy
    
    def get_capacitance(self) -> SimCharacteristics:
        return self._capacitance
    
    def get_timestamps_restarts(self) -> list:
        return [x for x in self._timestamps_restarts]
########### Parsing
    def set_permutation(self, permutation: str):
        self._permutation = permutation
        
    def get_permutation(self) -> str:
        return self._permutation
    
    def get_last_timestamp(self) -> float:
        return self._last_timestamp
    
    def parse_simulation(self):
        """This function parses the log files that were loaded at initiation, and creates the simulation object.
        """
        for i in range(len(self._event_df)):
            event = Event(self._event_df.iloc[i])
            self._filter_event(event)
        self.__close_simulation()
        
    def _debug_event(self, event: Event):
        if event.name == 'view.resetAllButton.isFiringProperty.changed':
            print(event.get_data())
            
    def _filter_event(self, event: Event):
        event_name = event.get_name()
        # print(event_name)
        # if Event_name != 'phetio.stepSimulation' and Event_name != 'phetio.inputEvent' and Event_name != 'phetio.get_state()':
        #     # print('          ', Event_name)
        if self._start_timestamp != -1:
            self._last_timestamp = self.__get_timestamp(event.get_timestamp())
            
        if event_name == 'simStarted':
            self.__start_simulation(event)
            
        elif event_name == 'phetio.state':
            pass 
            
        elif event_name == 'phetio.inputEvent':
            self.inaction(event)
        
        elif event_name == 'phetio.get_state()':
            return 'None'
        
        elif event_name == 'view.barMeterPanel.capacitanceCheckbox.toggled':
            self.__capacitance_checkbox(event)
            
        elif event_name == 'view.barMeterPanel.topPlateChargeCheckbox.toggled':
            self.__topplate_checkbox(event)
            
        elif event_name == 'view.barMeterPanel.storedEnergyCheckbox.toggled':
            self.__storedenergy_checkbox(event)
            
        elif event_name == 'view.viewControlPanel.plateChargesCheckbox.toggled':
            self.__platecharges_checkbox(event)
            
        elif event_name == 'view.viewControlPanel.barGraphsCheckbox.toggled':
            self.__bargraphs_checkbox(event)
            
        elif event_name == 'view.viewControlPanel.electricFieldCheckbox.toggled':
            self.__electricfield_checkbox(event)
            
        elif event_name == 'view.viewControlPanel.currentCheckbox.toggled':
            self.__currentdirection_checkbox(event)
            
        ############## Voltmeter
        # Dragging of the voltmeter
        elif event_name == 'view.voltmeterNode.bodyNode.dragHandler.dragStarted':
            self.__voltmeter_startdragging(event)
            
        elif event_name == 'view.voltmeterNode.bodyNode.dragHandler.dragged':
            self.__voltmeter_dragging(event)

        elif event_name == 'view.voltmeterNode.bodyNode.dragHandler.dragEnded':
            self.__voltmeter_stopdragging(event)
            
        elif event_name == 'view.voltmeterToolbox.dragged':
            self.__voltmeter_startdragging(event)
        
        # Draging of the top probe
        elif event_name == 'view.voltmeterNode.positiveProbeNode.dragHandler.dragStarted':
            self.__positiveprobe_start_dragging(event)
            
        elif event_name == 'view.voltmeterNode.positiveProbeNode.dragHandler.dragged':
            self.__positiveprobe_dragging(event)
            
        elif event_name == 'view.voltmeterNode.positiveProbeNode.dragHandler.dragEnded':
            self.__positiveprobe_enddragging(event)
        
        elif event_name == 'view.voltmeterNode.negativeProbeNode.dragHandler.dragStarted':
            self.__negativeprobe_startdragging(event)
            
        elif event_name == 'view.voltmeterNode.negativeProbeNode.dragHandler.dragged':
            self.__negativeprobe_dragging(event)
            
        elif event_name == 'view.voltmeterNode.negativeProbeNode.dragHandler.dragEnded':
            self.__negativeprobe_enddragging(event)
            
            
        ############## Voltage slider
        # Dragging of the battery slider

        elif event_name == 'view.capacitanceCircuitNode.batteryNode.sliderNode.track.trackInputListener.dragStarted':
            self.__voltage_starttracking(event)

        elif event_name == 'view.capacitanceCircuitNode.batteryNode.sliderNode.track.trackInputListener.dragged':
            self.__voltage_tracking(event)

        elif event_name == 'view.capacitanceCircuitNode.batteryNode.sliderNode.track.trackInputListener.dragEnded':
            self.__voltage_enddragging(event)
            
        elif event_name == 'view.capacitanceCircuitNode.batteryNode.sliderNode.thumbInputListener.dragStarted':
            self.__voltage_startdragging(event)
            
        elif event_name == 'view.capacitanceCircuitNode.batteryNode.sliderNode.thumbInputListener.dragged':
            self.__voltage_dragging(event)
            
        elif event_name == 'view.capacitanceCircuitNode.batteryNode.sliderNode.thumbInputListener.dragEnded':
            self.__voltage_enddragging(event)
            
            
        # # ############## Circuit manipulations
        # # top interactions
        elif event_name == 'view.capacitanceCircuitNode.topSwitchNode.openConnectionNode.pressListener.press':
            self.__topnode_open_pressed(event)
            
        elif event_name == 'view.capacitanceCircuitNode.topSwitchNode.openConnectionNode.pressListener.drag':
            self.__topnode_open_dragged(event)
            
        elif event_name == 'view.capacitanceCircuitNode.topSwitchNode.openConnectionNode.pressListener.release':
            self.__topnode_open_release(event)
            
        elif event_name == 'view.capacitanceCircuitNode.topSwitchNode.batteryConnectionNode.pressListener.press':
            self.__topnode_battery_pressed(event)
            
        elif event_name == 'view.capacitanceCircuitNode.topSwitchNode.batteryConnectionNode.pressListener.drag':
            self.__topnode_battery_dragged(event)
            
        elif event_name == 'view.capacitanceCircuitNode.topSwitchNode.batteryConnectionNode.pressListener.release':
            self.__topnode_battery_released(event)
            
        elif event_name == 'view.capacitanceCircuitNode.topSwitchNode.dragHandler.drag':
            self.__topnode_dragging(event)
        
        elif event_name == 'view.capacitanceCircuitNode.topSwitchNode.dragHandler.release':
            self.__topnode_release(event)
            
        # bottom interactions        
        elif event_name == 'view.capacitanceCircuitNode.bottomSwitchNode.openConnectionNode.pressListener.press':
            self.__bottomnode_open_pressed(event)
            
        elif event_name == 'view.capacitanceCircuitNode.bottomSwitchNode.openConnectionNode.pressListener.drag':
            self.__bottomnode_open_presseddrag(event)
            
        elif event_name == 'view.capacitanceCircuitNode.bottomSwitchNode.openConnectionNode.pressListener.release':
            self.__bottomnode_open_release(event)
        
        elif event_name == 'view.capacitanceCircuitNode.bottomSwitchNode.batteryConnectionNode.pressListener.press':
            self.__bottomnode_battery_pressed(event)
            
        elif event_name == 'view.capacitanceCircuitNode.bottomSwitchNode.batteryConnectionNode.pressListener.drag':
            self.__bottomnode_battery_drag(event)
            
        elif event_name == 'view.capacitanceCircuitNode.bottomSwitchNode.batteryConnectionNode.pressListener.release':
            self.__bottomnode_battery_release(event)
            
        elif event_name == 'view.capacitanceCircuitNode.bottomSwitchNode.dragHandler.drag':
            self.__bottomnode_dragging(event)
            
        elif event_name == 'view.capacitanceCircuitNode.bottomSwitchNode.dragHandler.release':
            self.__bottomnode_release(event)
            
        elif event_name == 'model.circuit.circuitConnectionProperty.changed':
            self.__circuit_propertychange(event)
            
        elif event_name == 'model.circuit.currentAmplitudeProperty.changed':
            self.__circuit_getamplitude(event)
            
        # ############## Plate Separation
        elif event_name == 'view.capacitanceCircuitNode.plateSeparationDragHandleNode.dragHandler.start':
            self.__plateseparation_startdragging(event)
            
        elif event_name == 'view.capacitanceCircuitNode.plateSeparationDragHandleNode.dragHandler.drag':
            self.__plateseparation_dragging(event)
            
        elif event_name == 'view.capacitanceCircuitNode.plateSeparationDragHandleNode.dragHandler.release':
            self.__plateseparation_stopdragging(event)
            
        elif event_name == 'view.capacitanceCircuitNode.plateSeparationDragHandleNode.dragHandler.press':
        #     # print(event.get_data())
        #     Coordinates of the slider, not that important
            
        #     self.plateSeparationPressing(event)
        #     area = event.get_data()['parameters']['x'] * event.get_data()['parameters']['y']
        #     self._plate_separation.is_dragging(event.get_data()['parameters'])
        #     # print(event.get_data())
            pass
            # return 'None'
            
        # ############## Plate Area
        elif event_name == 'view.capacitanceCircuitNode.plateAreaDragHandleNode.inputListener.start':
            self.__platearea_startdragging(event)
            
        elif event_name == 'view.capacitanceCircuitNode.plateAreaDragHandleNode.inputListener.drag':
            self.__platearea_dragging(event)
            
        elif event_name == 'view.capacitanceCircuitNode.plateAreaDragHandleNode.inputListener.release':
            self.__platearea_stopdragging(event)
        
        elif event_name == 'view.capacitanceCircuitNode.plateAreaDragHandleNode.inputListener.press':
            # Gives coordinate of the drag handler press -> not interesting ?
            # return 'None'
            pass
        
        elif event_name == 'view.resetAllButton.fired':
            self.__reset_simulation(event)
            
        elif event_name == 'capacitorLabBasics.browserTabVisibleProperty.changed':
            self.__tab_change(event)
            
        elif event_name == 'browserTabVisibleProperty.changed':
            self.__tab_change(event)
            
        elif event_name == 'phetio.stepSimulation':
            if self._last_event != 'phetio.stepSimulation':
                self.inaction(event)
            
        else:
            print('Event not found: ' + event_name + '          ' + event.get_type())
            # print(event.get_data())
            print()
            
        self._last_event = event_name
            
 ##########################################################################################################################
 ############# Button - Action Functions
    def __get_timestamp(self, timestamp:datetime):
        if self._start_timestamp == -1:
            print('Time stamp not initialised')
        return ((timestamp - self._start_timestamp).total_seconds())
            
    def __start_simulation(self, event: Event):
        if self._start_timestamp == -1:
            self._timeline.append('start_simulation')
            self._timestamps.append(0)
            self._started = 1
            self._start_timestamp = event.get_timestamp()
            self._timestamps_restarts.append(0)
        else:
            # Used in the case where several simulations have been opened for one user
            self._timeline.append('start_multiple_simulation')
            timestamp = self.__get_timestamp(event.get_timestamp())
            self._timestamps.append(timestamp)
            self._timestamps_restarts.append(timestamp)
            self._reset_simulation(event)
        
    ###########  Toggle each of the Checkboxes
    def __capacitance_checkbox(self, event: Event):
        self._timeline.append('checkbox_capacitance')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._checkbox_capacitance.check_switch(int(event.get_data()['parameters']['newValue']), timestamp)
        
    def __topplate_checkbox(self, event: Event):
        self._timeline.append('checkbox_topplate')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._checkbox_topplate.check_switch(int(event.get_data()['parameters']['newValue']), timestamp)
              
    def __storedenergy_checkbox(self, event: Event):
        self._timeline.append('checkbox_storedenergy')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._checkbox_storedenergy.check_switch(int(event.get_data()['parameters']['newValue']), timestamp)
        
    def __platecharges_checkbox(self, event: Event):
        self._timeline.append('checkbox_platecharges')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._checkbox_platecharges.check_switch(int(event.get_data()['parameters']['newValue']), timestamp)
            
    def __bargraphs_checkbox(self, event: Event):
        self._timeline.append('checkbox_bargraphs')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._checkbox_bargraphs.check_switch(int(event.get_data()['parameters']['newValue']), timestamp)
            
    def __electricfield_checkbox(self, event: Event):
        self._timeline.append('checkbox_electricField')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._checkbox_electricfields.check_switch(int(event.get_data()['parameters']['newValue']), timestamp)
            
    def __currentdirection_checkbox(self, event: Event):
        self._timeline.append('checkbox_currentdirection')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._checkbox_currentdirection.check_switch(int(event.get_data()['parameters']['newValue']), timestamp)
        
    ########## Voltmeter
    
    # body dragging
    def __voltmeter_startdragging(self, event: Event):
        self._timeline.append('voltmeter_startdragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        start_dragging = True
        if 'parameters' in event.get_data():
            self._voltmeter.start_dragging(event.get_data()['parameters'], timestamp)
            start_dragging = True
        for child in event.get_data()['children']:
            if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.view.voltmeterNode.bodyNode.dragHandler' and child['event'] == 'dragStarted':
                start_dragging = True
            if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeterVisibleProperty':
                self._voltmeter.check_visibility_switch(int(child['parameters']['newValue']), timestamp)
                if 'children' in child['phetioID']:
                    for grandchild in child['phetioID']['children']:
                        if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.measuredVoltageProperty':
                            self._voltage.check_state(grandchild['parameters']['newValue'], timestamp)
                            if start_draging:
                                # print('position voltmeter: ', grandchild['parameters']['newValue'])
                                self._voltmeter.start_dragging(grandchild['parameters']['newValue'], timestamp)
            if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.bodyLocationProperty':
                self._voltmeter.start_dragging(child['parameters']['newValue'], timestamp)
                             
    def __voltmeter_dragging(self, event: Event):
        self._timeline.append('voltmeter_dragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        if 'children' in event.get_data():
            for child in event.get_data()['children']:
                if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.bodyLocationProperty':
                    self._voltmeter.is_dragging(child['parameters']['newValue'], self.__get_timestamp(event.get_timestamp()))
                            
    def __voltmeter_stopdragging(self, event: Event):
        self._timeline.append('voltmeter_stopdragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._voltmeter.stop_dragging(0, timestamp)
        if 'children' in event.get_data():
            for child in event.get_data()['children']:
                if 'children' in child:
                    for grandchild in child['children']:
                        if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeterVisibleProperty':
                            self._voltmeter.check_visibility_switch(int(grandchild['parameters']['newValue']), timestamp)
                        if 'children' in grandchild:
                            for grandgrandchild in grandchild['children']:
                                self._voltage.check_state(grandgrandchild['parameters']['newValue'], timestamp)
                 
    def __positiveprobe_start_dragging(self, event: Event):
        self._timeline.append('voltmeter_positiveprobe_startdragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._voltmeter.check_switch(1, timestamp)
        self._positive_probe.start_dragging(self._voltage.get_state(), timestamp)
        
    def __positiveprobe_dragging(self, event: Event):
        self._timeline.append('voltmeter_positiveprobe_dragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        # self._voltmeter.check_switch(1, timestamp)
        if not self._positive_probe.is_active():
            print('Error in the simulation, no dragging started')
        else:
            if 'children' in event.get_data():
                for child in event.get_data()['children']:
                    if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.positiveProbeLocationProperty' and 'children' in child:
                        for grandchild in child['children']:
                            # Don't know what target is
                            if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.positiveProbeTargetProperty':
                                target = grandchild['parameters']['newValue']
                                self._positive_probe.is_dragging(target, timestamp)
                            if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.measuredVoltageProperty':
                                measured_voltage = grandchild['parameters']['newValue']
                                self._voltage.check_state(measured_voltage, timestamp)
                                
    def __positiveprobe_enddragging(self, event: Event):
        self._timeline.append('voltmeter_positiveprobe_stopdragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._voltmeter.check_switch(0, timestamp)
        self._positive_probe.stop_dragging(self._positive_probe.get_state(), timestamp)
        
    # Bottom node dragging
    def __negativeprobe_startdragging(self, event: Event):
        self._timeline.append('voltmeter_negativeprobe_startdragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._voltmeter.check_switch(1, timestamp)
        self._negative_probe.start_dragging(self._negative_probe.get_state(), timestamp)
        
    def __negativeprobe_dragging(self, event: Event):
        self._timeline.append('voltmeter_negativeprobe_dragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._voltmeter.check_switch(1, timestamp)
        if not self._negative_probe.is_active():
            print('Error in the simulation, no dragging started')
        else:
            if 'children' in event.get_data():
                for child in event.get_data()['children']:
                    if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.negativeProbeLocationProperty':
                        if 'children' in child:
                            for grandchild in child['children']:
                                if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.negativeProbeTargetProperty':
                                    self._negative_probe.is_dragging(grandchild['parameters']['newValue'], timestamp)
                                if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.measuredVoltageProperty':
                                    measured_voltage = grandchild['parameters']['newValue']
                                    self._voltage.check_state(measured_voltage, timestamp)
                                    
    def __negativeprobe_enddragging(self, event: Event):
        self._timeline.append('voltmeter_negativeprobe_stopdragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._voltmeter.check_switch(0, timestamp)
        self._negative_probe.stop_dragging(self._voltage.get_state(), timestamp)
        
    # Battery Slider start Dragging
    # def voltageProbeDragging(self, event):
    #     self._timeline('voltage_dragging')
    #     self.events_sequence.append('Voltage Probe Dragging')
    #     self._voltage.dragging(self._voltage.get_state(), self.__get_timestamp(event.get_timestamp()))
        
    def __voltage_startdragging(self, event: Event):
        self._timeline.append('voltage_startdragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._voltage.start_dragging(self._voltage.get_state(), timestamp)

    def __voltage_starttracking(self, event: Event):
        self._timeline.append('voltage_startdragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._voltage.start_dragging(self._voltage.get_state(), timestamp)
        if 'children' in event.get_data():
            for child in event.get_data()['children']:
                if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.battery.voltageProperty' and 'children' in child:
                    for grandchild in child['children']:
                        if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateVoltageProperty':
                            self._plate_voltage.set_state(grandchild['parameters']['newValue'], timestamp)
                        if 'children' in grandchild:
                            for grandgrandchild in grandchild['children']:
                                if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateChargeProperty':
                                    self._plate_charge.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                                if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.storedEnergyProperty':
                                    self._stored_energy.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                                # if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.measuredVoltageProperty':
                                #     self._voltage.start_dragging(grandgrandchild['parameters']['newValue'], timestamp)

    def __voltage_tracking(self, event: Event):
        self._timeline.append('voltage_dragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        
        if 'children' in event.get_data():
            for child in event.get_data()['children']:
                if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.battery.voltageProperty' and 'children' in child:
                    for grandchild in child['children']:
                        if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateVoltageProperty':
                            self._plate_voltage.set_state(grandchild['parameters']['newValue'], timestamp)
                        if 'children' in grandchild['children']:
                            for grandgrandchild in grandchild['children']:
                                if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateChargeProperty':
                                    self._plate_charge.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                                if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.storedEnergyProperty':
                                    self._stored_energy.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                                if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.measuredVoltageProperty':
                                    self._voltage.is_dragging(grandgrandchild['parameters']['newValue'], timestamp)
                        else:
                            self._voltage.is_dragging(-1,  timestamp)
        else:
            self._voltage.is_dragging(-1,  timestamp)
            
    def __voltage_dragging(self, event: Event):
        self._timeline.append('voltage_dragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        if 'children' in event.get_data():
                for child in event.get_data()['children']:
                    if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.battery.voltageProperty':
                        self._voltage.is_dragging(child['parameters']['newValue'], timestamp)
                        
                    if 'children' in child:
                        for grandchild in child['children']:
                            if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateVoltageProperty':
                                self._plate_voltage.set_state(grandchild['parameters']['newValue'], timestamp)
                                
                        if 'children' in grandchild:
                            for grandgrandchild in grandchild['children']:
                                if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateChargeProperty':
                                    self._plate_charge.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                                if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.storedEnergyProperty':
                                    self._stored_energy.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                                if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.measuredVoltageProperty':
                                    measured_voltage = grandgrandchild['parameters']['newValue']
                                    self._voltage.is_dragging(measured_voltage, timestamp)
        
    def __voltage_enddragging(self, event: Event):
        self._timeline.append('voltage_stopdragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._voltage.stop_dragging(self._voltage.get_state(), timestamp)
        
    ########## Circuit
    # Top Node
    def __topnode_open_pressed(self, event: Event):
        self._timeline.append('circuit_topnodeopen_pressed')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._top_openconnection_node.start_dragging(self._top_openconnection_node.get_state(), timestamp)
        if 'children' in event.get_data():
            for child in event.get_data()['children']:
                if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.circuitConnectionProperty':
                    self._circuit.set_state_circuit(child['parameters']['newValue'], timestamp)
                    if 'children' in child['children']:
                        for grandchild in child['children']:
                            if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateVoltageProperty':
                                self._plate_voltage.set_state(grandchild['parameters']['newValue'], timestamp)
                            elif grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.storedEnergyProperty':
                                self._stored_energy.set_state(grandchild['parameters']['newValue'], timestamp)
                                
                elif child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.view.capacitanceCircuitNode.topSwitchNode.dragHandler':
                    if 'children' in child:
                        for grandchild in child['children']:
                            if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.circuitConnectionProperty':
                                self._circuit.set_state_circuit(grandchild['parameters']['newValue'], timestamp)
           
    def __topnode_open_dragged(self, event: Event):
        self._timeline.append('circuit_topnodeopen_dragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._top_openconnection_node.is_dragging(event.get_data()['parameters'], timestamp)
                     
    def __topnode_open_release(self, event: Event):
        self._timeline.append('circuit_topnodeopen_release')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._top_openconnection_node.stop_dragging(self._top_openconnection_node.get_state(), timestamp)
                            
    def __topnode_battery_pressed(self, event: Event):
        self._timeline.append('circuit_topenodebattery_pressed')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._top_battery_node.start_dragging(1, self.__get_timestamp(event.get_timestamp()))
        if 'children' in event.get_data():
            for child in event.get_data()['children']:
                if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.circuitConnectionProperty':
                    self._circuit.set_state_circuit(child['parameters']['newValue'], timestamp)
                    if 'children' in child:
                        for grandchild in child['children']:
                            if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateVoltageProperty':
                                self._voltage.check_state(grandchild['parameters']['newValue'], timestamp)
                                if 'children' in grandchild:
                                    for grandgrandchild in grandchild['children']:
                                        if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateChargeProperty':
                                            self._plate_charge.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                                        elif grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.storedEnergyProperty':
                                            self._stored_energy.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                            elif grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.topCircuitSwitch.angleProperty':
                                if 'children' in grandchild:
                                    for grandgrandchild in grandchild['children']:
                                        if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.measuredVoltageProperty':
                                            self._voltage.check_state(grandgrandchild['parameters']['newValue'], timestamp)
                                            
                # if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.view.capacitanceCircuitNode.topSwitchNode.dragHandler':
                #     self._top_node.is_dragging(child['parameters'], self.__get_timestamp(event.get_timestamp()))
                #     if 'children' in child:
                #         for grandchild in child['children']:
                #             if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.circuitConnectionProperty':
                #                 self._circuit.set_state_circuit(grandchild['parameters']['newValue'], self.__get_timestamp(event.get_timestamp()))
                    
    def __topnode_battery_dragged(self, event: Event):      
        self._timeline.append('circuit_topnodebattery_dragged')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)  
        self._top_battery_node.is_dragging(event.get_data()['parameters'], timestamp)                            
                    
    def __topnode_battery_released(self, event: Event):
        self._timeline.append('circuit_topnodebattery_released')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._top_battery_node.stop_dragging(self._top_battery_node.get_state(), timestamp)
    
    def __topnode_dragging(self, event: Event):
        self._timeline.append('circuit_topnode_dragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        if 'children' in event.get_data():
            for child in event.get_data()['children']:
                if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.topCircuitSwitch.angleProperty':
                    if 'children' in child:
                        for grandchild in child['children']:
                            if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.topCircuitSwitch.switchSegment.endPointProperty':
                                self._top_node.is_dragging(grandchild['parameters']['newValue'], timestamp)
                            # if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.bottomCircuitSwitch.angleProperty' and 'children' in grandchild:
                            #     for grandgrandchild in grandchild['children']:
                            #         if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.bottomCircuitSwitch.switchSegment.endPointProperty':
                            #             self._bottom_node.start_dragging(grandgrandchild['parameters']['newValue'], self.__get_timestamp(event.get_timestamp()))   
                            else:
                                self._top_node.is_dragging(-1, timestamp)
        else:
            self._top_node.is_dragging(-1, timestamp)
                                
                                     
    def __topnode_release(self, event: Event):
        self._timeline.append('circuit_topnode_release')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._top_node.stop_dragging(self._top_node.get_state(), timestamp)
        # self._bottom_node.stop_dragging(self._bottom_node.get_state(), self.__get_timestamp(event.get_timestamp()))  
                                 
    # Bottom node          
    def __bottomnode_open_pressed(self, event: Event):
        self._timeline.append('circuit_bottomnodeopen_pressed')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        if 'children' in event.get_data():
            for child in event.get_data()['children']:
                if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.circuitConnectionProperty':
                    self._circuit.set_state_circuit(child['parameters']['newValue'], timestamp)
                    self._bottom_openconnection_node.start_dragging(self._bottom_openconnection_node.get_state(), timestamp)
                    
    def __bottomnode_open_presseddrag(self, event: Event):
        self._timeline.append('circuit_bottomnodeopen_drag')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._bottom_openconnection_node.is_dragging(event.get_data()['parameters'], timestamp)
    
    def __bottomnode_open_release(self, event: Event):
        self._timeline.append('circuit_bottomnodeopen_release')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._bottom_openconnection_node.stop_dragging(self._bottom_openconnection_node.get_lastvalue_drags(), timestamp)
        
    def __bottomnode_battery_pressed(self, event: Event):
        self._timeline.append('circuit_bottomnodebattery_pressed')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._bottom_battery_node.start_dragging(self._bottom_battery_node.get_state(), timestamp)
        if 'children' in event.get_data():
            for child in event.get_data()['children']:
                if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.circuitConnectionProperty':
                    self._circuit.set_state_circuit(child['parameters']['newValue'], timestamp)
                    if 'children' in child:
                        for grandchild in child['children']:
                            if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateVoltageProperty':
                                self._plate_voltage.set_state(grandchild['parameters']['newValue'], timestamp)
                                if 'children' in event.get_data()['children']:
                                    for grandgrandchild in grandchild['children']:
                                        if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateChargeProperty':
                                            self._plate_charge.set_state(grandgranchild['parameters']['newValue'], timestamp)
                                        if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.storedEnergyProperty':
                                            self._stored_energy.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                            if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.topCircuitSwitch.angleProperty':
                                if 'children' in grandchild:
                                    for grandgrandchild in grandchild['children']:
                                        if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.measuredVoltageProperty':
                                            self._voltage.check_state(grandgrandchild['parameters']['newValue'], timestamp)
                if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.view.capacitanceCircuitNode.bottomSwitchNode.dragHandler':
                    if 'children' in child:
                        for grandchild in child['children']:
                            self._circuit.set_state_circuit(grandchild['parameters']['newValue'], timestamp)
                            
    def __bottomnode_battery_drag(self, event: Event):
        self._timeline.append('circuit_bottomnodebattery_drag')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._bottom_battery_node.is_dragging(self._bottom_battery_node.get_lastvalue_drags(), timestamp)
        
    def __bottomnode_battery_release(self, event:Event):
        self._timeline.append('circuit_bottomnodebattery_release')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._bottom_battery_node.stop_dragging(self._bottom_battery_node.get_lastvalue_drags(), timestamp)
        
    def __bottomnode_dragging(self, event: Event):
        self._timeline.append('circuit_bottomnode_drag')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        if 'children' in event.get_data():
            for child in event.get_data()['children']:
                # if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.bottomCircuitSwitch.angleProperty':
                    # # print('No need to record angle ?')
                if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.bottomCircuitSwitch.switchSegment.endPointProperty':
                    # print('bottomnodedragging ', child['parameters']['newValue'])
                    self._bottom_node.is_dragging(child['parameters']['newValue'], timestamp)
                elif child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.topCircuitSwitch.switchSegment.endPointProperty':
                    print('ho')
                    self._bottom_node.is_dragging(child['parameters']['newValue'], timestamp)
                
                else:
                    self._bottom_node.is_dragging(-1, timestamp)
                    
    def __bottomnode_release(self, event: Event):
        self._timeline.append('circuit_bottomnode_release')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._bottom_node.stop_dragging(self._bottom_node.get_lastvalue_drags(), timestamp)
        
    # Circuit change
    def __circuit_propertychange(self, event: Event):
        self._timeline.append('circuit_propertychange')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._circuit.set_state_circuit(event.get_data()['parameters']['newValue'], timestamp)
        if 'children' in event.get_data():
            for child in event.get_data()['children']:
                if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateVoltageProperty':
                    self._plate_voltage.set_state(child['parameters']['newValue'], timestamp)
                    if 'children' in child:
                        for grandchild in child['children']:
                            if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateChargeProperty':
                                self._plate_charge.set_state(grandchild['parameters']['newValue'], timestamp)
                            if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.storedEnergyProperty':
                                self._stored_energy.set_state(grandchild['parameters']['newValue'], timestamp)
    
    def __circuit_getamplitude(self, event: Event):
        self._timeline.append('circuit_amplitudecheck')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._current.set_state(event.get_data()['parameters']['newValue'], timestamp)
                                
    ########## Plate separation
    # Start dragging
    def __plateseparation_startdragging(self, event: Event):
        self._timeline.append('plateseparation_startdragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._plate_separation.start_dragging(self._plate_separation.get_lastvalue_drags(), timestamp)
        
    # Dragging
    def __plateseparation_dragging(self, event: Event):
        self._timeline.append('plateseparation_dragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        if 'children' in event.get_data():
            for child in event.get_data()['children']:
                if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateSeparationProperty':
                    self._plate_separation.is_dragging(child['parameters']['newValue'], timestamp)
                    if 'children' in child:
                        if (child['parameters']['newValue'] > 1):
                            print('CHILD BIGGER', child['parameters']['newValue'])
                        for grandchild in child['children']:
                            if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.capacitanceProperty':
                                self._capacitance.set_state(grandchild['parameters']['newValue'], timestamp)
                                if 'children' in grandchild:
                                    for grandgrandchild in grandchild['children']:
                                        if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateChargeProperty':
                                            self._plate_charge.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                                        elif grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.storedEnergyProperty':
                                            self._stored_energy.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                            if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateVoltageProperty':
                                self._plate_voltage.set_state(grandchild['parameters']['newValue'], timestamp)
                                if 'children' in grandchild:
                                    for grandgrandchild in grandchild['children']:
                                        if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.storedEnergyProperty':
                                            self._stored_energy.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                                        elif grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.measuredVoltageProperty':
                                            self._voltage.check_state(grandgrandchild['parameters']['newValue'], timestamp)
    
    # End Dragging    
    def __plateseparation_stopdragging(self, event: Event):
        self._timeline.append('plateseparation_stopdragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._plate_separation.stop_dragging(self._plate_separation.get_lastvalue_drags(), timestamp)
        
    ########## Plate Area
    # Start Dragging
    def __platearea_startdragging(self, event: Event):
        self._timeline.append('platearea_startdragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._plate_area.start_dragging(self._plate_area.get_state(), timestamp)
        
    # Dragging
    def __platearea_dragging(self, event: Event):
        self._timeline.append('platearea_dragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        if 'children' in event.get_data():
            for child in event.get_data()['children']:
                if child['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateSizeProperty':
                    length = child['parameters']['newValue']['maxX']
                    width = child['parameters']['newValue']['maxZ']
                    self._plate_area.is_dragging(length * width, timestamp)
                    if 'children' in child:
                        for grandchild in child['children']:
                            if grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.capacitanceProperty':
                                self._capacitance.set_state(grandchild['parameters']['newValue'], timestamp)
                                if 'children' in grandchild:
                                    for grandgrandchild in grandchild['children']:
                                        if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateChargeProperty':
                                            self._plate_charge.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                                        elif grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.storedEnergyProperty':
                                            self._stored_energy.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                            elif grandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateVoltageProperty':
                                self._plate_voltage.set_state(grandchild['parameters']['newValue'], timestamp)
                                if 'children' in grandchild:
                                    for grandgrandchild in grandchild['children']:
                                        if grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.plateChargeProperty':
                                            self._plate_charge.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                                        elif grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.circuit.capacitor.storedEnergyProperty':
                                            self._stored_energy.set_state(grandgrandchild['parameters']['newValue'], timestamp)
                                        elif grandgrandchild['phetioID'] == 'capacitorLabBasics.capacitanceScreen.model.voltmeter.measuredVoltageProperty':
                                            self._voltage.check_state(grandgrandchild['parameters']['newValue'], timestamp)
    
    # Stop Dragging
    def __platearea_stopdragging(self, event: Event):
        self._timeline.append('platearea_stopdragging')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._plate_area.stop_dragging(self._plate_area.get_state(), timestamp)
        
    ########## Simulation reset
    def __reset_simulation(self, event: Event):
        self._timeline.append('reset_simulation')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._timestamps_restarts.append(timestamp)
        
        self._checkbox_capacitance.reset(timestamp) 
        self._checkbox_topplate.reset(timestamp)
        self._checkbox_storedenergy.reset(timestamp)
        self._checkbox_platecharges.reset(timestamp)
        self._checkbox_bargraphs.reset(timestamp)
        self._checkbox_electricfields.reset(timestamp)
        self._checkbox_currentdirection.reset(timestamp)
        
        self._voltmeter.reset(timestamp)
        self._positive_probe.reset(timestamp)
        self._negative_probe.reset(timestamp)
        
        self._voltage.reset(timestamp)
        self._plate_separation.reset(timestamp)
        self._plate_area.reset(timestamp)
        self._circuit.reset(timestamp)
        self._top_openconnection_node.reset(timestamp)
        self._bottom_openconnection_node.reset(timestamp)
        self._top_battery_node.reset(timestamp)
        self._bottom_battery_node.reset(timestamp)
        self._top_node.reset(timestamp)
        self._bottom_node.reset(timestamp)
        
        self._plate_voltage.reset(timestamp)
        self._plate_charge.reset(timestamp)
        
        self._stored_energy.reset(timestamp)
        self._capacitance.reset(timestamp)
        
        self._inactivity.reset(timestamp)
        self._inactivity_start = 0
        
    ########## Tab change
    def __tab_change(self, event: Event):
        self._timeline.append('tab_change')
        timestamp = self.__get_timestamp(event.get_timestamp())
        self._timestamps.append(timestamp)
        self._attentive.check_switch(event.get_data()['parameters']['newValue'], timestamp)
        
    ########## Inaction
    def inaction(self, event: Event):
        pass
        # self._timeline.append('inaction')
        # self._timestamps.append(self.__get_timestamp(event.get_timestamp()))
        # if not self._inactivity_start:
        #     self._inactivity_start = 1
        # else:
        #     self._inactivity.switch(self.__get_timestamp(event.get_timestamp()))
            
            
    ########## CLose Simulation
    def __close_simulation(self):
        self._timeline.append('close_simulation')
        self._timestamps.append(self._last_timestamp)
        timestamp = self._last_timestamp
        self._checkbox_capacitance.close(timestamp)
        self._checkbox_topplate.close(timestamp)
        self._checkbox_storedenergy.close(timestamp)
        self._checkbox_platecharges.close(timestamp)
        self._checkbox_bargraphs.close(timestamp)
        self._checkbox_electricfields.close(timestamp)
        self._checkbox_currentdirection.close(timestamp)
        
        self._voltmeter.close(timestamp)
        self._positive_probe.close(timestamp)
        self._negative_probe.close(timestamp)
        self._voltage.close(timestamp)
        self._plate_separation.close(timestamp)
        self._plate_area.close(timestamp)
        self._circuit.close_circuit(timestamp)
        self._current.close(timestamp)
        self._top_openconnection_node.close(timestamp)
        self._bottom_openconnection_node.close(timestamp)
        self._top_battery_node.close(timestamp)
        self._bottom_battery_node.close(timestamp)
        self._top_node.close(timestamp)
        self._bottom_node.close(timestamp)
        self._plate_voltage.close(timestamp)
        self._plate_charge.close(timestamp)
        self._stored_energy.close(timestamp)
        self._capacitance.close(timestamp)
        self._inactivity.close(timestamp)
        self._attentive.close(timestamp)

    def save(self, version='') -> str:
        path = '../data/parsed simulations/perm' + self._permutation + '_lid' + str(self._learner_id) + '_sid' + self._simulation_id + 'v' + version + '_simulation.pkl'
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)

        return path
        
    
        
                
                
        
                
                
        
        
        
        
        
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
          