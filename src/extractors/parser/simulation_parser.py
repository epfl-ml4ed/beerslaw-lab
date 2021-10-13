import logging
import dill
import json
import pickle

import re
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
        self.nevents = 0
        self._load_logs(paths)
        
        if len(self._event_df) != 0:
            self.nevents = len(self._event_df)
            # Initialise simulation parameters
            self._initialise_simulation_parameters()
            
            # Initialise map of event - functions
            self._load_eventmap()
            self._load_variablesmap()
            
            self.debug = []
        
    ##############################
    # Initialisations    
    #
    def _process_children(self, event: Event, event_df: list, debug=False) -> list:
        message_index = event.get_message_index()
        timestamp = event.get_timestamp()
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        hour = timestamp.hour
        minute = timestamp.minute
        second = timestamp.second
        
        phetio_id = event.get_phetio_id()
        event_name = event.get_name()
        event_type = event.get_type()
        event_component = event.get_component()
        if len(event.get_params().keys()) != 0:
            event_params = event.get_params()
        else:
            event_params = {}
            
        new_event = [
                message_index, timestamp, year, month, day, hour, minute, second, 
                phetio_id, event_name, event_type, event_component, event_params, {}
        ]
        event_df.append(new_event)
        
        
        
        if len(event.get_children()) != 0:
            for child in event.get_children():
                child_event = self._process_event(child)
                child_event = Event(self._process_child_list(child_event))
                self._process_children(child_event, event_df)
                
        return event_df
    
    def _process_child_list(self, child:list) -> dict:
        dic = {
            'message_index': child[0],
            'timestamp': child[1],
            'year': child[2],
            'month': child[3],
            'day': child[4],
            'hour': child[5],
            'minute': child[6],
            'second': child[7],
            'phetio_id': child[8],
            'event_name': child[9],
            'event_type': child[10],
            'event_component': child[11],
            'event_params': child[12],
            'event_children': child[13]
        }
        return dic
    
    def _process_event(self, event:dict) -> list:
        message_index = event['messageIndex']
        timestamp = event['time']
        timestamp = datetime.datetime.fromtimestamp(timestamp/1e3)
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        hour = timestamp.hour
        minute = timestamp.minute
        second = timestamp.second
        
        phetio_id = event['phetioID']
        event_name = event['event']
        event_type = event['eventType']
        event_component = event['componentType']
        if 'parameters' in event:
            event_params = event['parameters']
        else:
            event_params = {}
            
        if 'children' in event:
            event_children = event['children']
        else:
            event_children = {}
            
        new_event = [
                message_index, timestamp, year, month, day, hour, minute, second, 
                phetio_id, event_name, event_type, event_component, event_params, event_children
        ]
        return new_event
    
    def _load_logs(self, path: str):
        """ Turns the logs into a dataframe where each row describes an event
        Args:
            paths (list<str>): [list of the paths of the logs to create the simulation from]
        """
        event_df = []
        
        task = int(path[-5])
        self._task = str(task)
        self._learner_id = path.split('\\')[-1].split('-')[0]
        self._simulation_id = self._learner_id + '_' + str(task)
        self._permutation = ''
        
        file = open(path, 'r')
        f = file.readlines()
        f = ''.join(f)
        f = re.sub('}\n{', '},\n{', f)
        f = '[' + f + ']'
        logs = json.loads(f)
        logs.sort(key=lambda x: (x['time'], x['messageIndex']))
        
        for event in logs:
            e_df = self._process_event(event)
            event_df.append(e_df)
            
        df = pd.DataFrame(event_df)
        if len(df) != 0:
            df.columns = [
                'message_index', 'timestamp', 'year', 'month', 'day', 'hour', 'minute', 'second',
                'phetio_id', 'event_name', 'event_type', 'event_component', 'event_params', 'event_children'
            ]
            df = df.sort_values(['year', 'month', 'day', 'hour', 'minute', 'second', 'message_index'])
        self._event_df = df

    def _initialise_simulation_parameters(self):
        self._started = 0
        self._start_timestamp = -1
        self._timestamps_restarts = []
        
        # characteristics
        self._wavelength = SimCharacteristics('wavelength', self._simulation_id, 508)
        self._wavelength_variable = Checkbox('wl preset or variable', self._simulation_id, False) # F for preset, T for variable
        self._wavelength_display = SimCharacteristics('wavelength', self._simulation_id, '508 nm')
        self._width = SimCharacteristics('width', self._simulation_id, 1)
        self._concentration = SimCharacteristics('concentration', self._simulation_id, 0.1)
        self._solution = SimCharacteristics('solution', self._simulation_id, 'drinkMix')
        self._ruler_position = SimCharacteristics('ruler position', self._simulation_id, (0, 0))
        
        
        # measurements displays and tools
        self._measure = SimCharacteristics('measure recorded', self._simulation_id, '-') # value of the absorbance or transmittance
        self._measure_display = SimCharacteristics('measure shown', self._simulation_id, '-') # value of the absorbance or transmittance
        self._metric = SimCharacteristics('metric observed', self._simulation_id, 'transmittance') # transmittance or absorbance
        self._magnifier_position = SimCharacteristics('magnifier position', self._simulation_id, (434, 131)) # absorbance or transmittance
        
        # Measures of transmittance
        self._checkbox_transmittance = Checkbox('checkbox_transmittance', self._simulation_id, 1)
        self._checkbox_absorbance = Checkbox('checkbox_absorbance', self._simulation_id, 0)
        self._magnifier = SimObjects('measuring_tool', self._simulation_id, '-')
        
        # Laser
        self._laser = Checkbox('laser', self._simulation_id, False)
        self._light = Checkbox('light', self._simulation_id, False)
        self._wl_preset = Checkbox('checkbox_wl_preset', self._simulation_id, True)
        self._wl_variable = Checkbox('checkbox_wl_variable', self._simulation_id, False)
        self._wl_slider_minus = SimObjects('wl_slider_minus', self._simulation_id, -1)
        self._wl_slider_plus = SimObjects('wl_slider_plus', self._simulation_id, -1)
        self._wl_slider = SimObjects('wl_slider', self._simulation_id, 508) # will record interactions from slider minus with the fired events
        
        # solution box
        self._solution_menu = SimObjects('solution_menu', self._simulation_id, 'drinkMix')
        self._concentration_slider_minus = SimObjects('concentration_slider_minus', self._simulation_id, 100)
        self._concentration_slider_plus = SimObjects('concentration_slider_plus', self._simulation_id, 100)
        self._concentration_slider = SimObjects('concentration_slider', self._simulation_id, 100) # will record interactions from slider minus with the fired events
        
        # width
        self._flask = SimObjects('flask', self._simulation_id, 1)
        self._ruler = SimObjects('ruler', self._simulation_id, 0)
        
        # pdf
        self._pdf = Checkbox('pdf', self._simulation_id, 0)
        
        # Other Simulation
        self._concentration_lab_state = Checkbox('concentrationlab', self._simulation_id, 0)
        self._concentration_actions = SimObjects('concentrationlab_actions', self._simulation_id, 0)
        self._chemlab_state = Checkbox('chemlab', self._simulation_id, 0)
        self._menu_state = Checkbox('menu', self._simulation_id, 1)
        
        self._last_timestamp = -1
        self._last_event = ''

        self._timeline = []
        self._timestamps = []
        
    def _load_eventmap(self):
        self._event_map = {
            'beersLawLab.sim': {
              'simStarted': self._no_update  
            },
            'beersLawLab.homeScreen.view.beersLawScreenSmallButton': {
                'fired': self._no_update
            },
            'beersLawLab.homeScreen.view.beersLawScreenLargeButton': {
                'fired': self._start_chemlab
            },
            'beersLawLab.navigationBar.beersLawScreenButton': {
                'fired': self._start_chemlab
            },
            'beersLawLab.simIFrameAPI': {
                'invoked': self._no_update
            },
            'beersLawLab.sim.screenIndexProperty':{
                'changed': self._no_update
            },
            'beersLawLab.beersLawScreen.view.detectorNode.probeNode.movableDragHandler': {
                'dragStarted': self._magnifier_dragstart,
                'dragEnded': self._magnifier_enddrag,
                'dragged': self._magnifier_drag
            },
            'beersLawLab.beersLawScreen.view.cuvetteNode.cuvetteDragHandler': {
                'dragStarted': self._flask_dragstart,
                'dragEnded': self._flask_enddrag,
                'dragged': self._flask_drag
            },
            'beersLawLab.beersLawScreen.view.lightNode.button': {
                'toggled': self._laser_toggle
            },
            'beersLawLab.wrapper': {
                'showPDF': self._pdf_show,
                'hidePDF': self._pdf_hide
            },
            'beersLawLab.beersLawScreen.view.wavelengthControls.variableWavelengthRadioButton': {
                'fired': self._wlvariable_toggle
            },
            'beersLawLab.beersLawScreen.view.wavelengthControls.presetWavelengthRadioButton': {
                'fired': self._wlpreset_toggle
            },
            'beersLawLab.beersLawScreen.view.wavelengthControls.wavelengthSlider.thumbInputListener': {
                'dragStarted': self._wlslider_startdrag,
                'dragEnded': self._wlslider_enddrag,
                'dragged': self._wlslider_drag
            },
            'beersLawLab.beersLawScreen.view.wavelengthControls.wavelengthSlider.trackInputListener': {
                'dragStarted': self._wlslider_touch,
                'dragEnded': self._wlslider_untouch,
                'dragged': self._wlslider_drag
            },
            'beersLawLab.beersLawScreen.view.wavelengthControls.wavelengthSlider.minusButton': {
                'fired': self._wlslider_minus
            },
            'beersLawLab.beersLawScreen.view.wavelengthControls.wavelengthSlider.plusButton': {
                'fired': self._wlslider_plus
            },
            'beersLawLab.beersLawScreen.view.solutionControls.concentrationControl.slider.thumb.dragHandler': {
                'dragStarted': self._concentration_startdrag,
                'dragEnded': self._concentration_enddrag,
                'dragged': self._concentration_drag
            },
            'beersLawLab.beersLawScreen.view.solutionControls.concentrationControl.slider.track.inputListener': {
                'dragStarted': self._concentration_touch,
                'dragEnded': self._concentration_untouch,
                'dragged': self._concentration_drag
            },
            'beersLawLab.beersLawScreen.view.solutionControls.concentrationControl.slider.minusButton': {
                'fired': self._concentration_minus
            },
            'beersLawLab.beersLawScreen.view.solutionControls.concentrationControl.slider.plusButton': {
                'fired': self._concentration_plus
            },
            'beersLawLab.beersLawScreen.view.solutionControls.comboBox': {
                'popupShown': self._solution_shown,
                'popupHidden': self._solution_hidden,
                'fired': self._solution_select
            },
            'beersLawLab.beersLawScreen.view.rulerNode.movableDragHandler':{
                'dragStarted': self._ruler_startdrag,
                'dragEnded': self._ruler_stopdrag,
                'dragged': self._ruler_drag
            },
            'beersLawLab.beersLawScreen.view.detectorNode.bodyNode.absorbanceRadioButton': {
                'fired': self._absorbance_click
            },
            'beersLawLab.beersLawScreen.view.detectorNode.bodyNode.transmittanceRadioButton': {
                'fired': self._transmittance_click
            },
            'beersLawLab.beersLawScreen.view.resetAllButton':{
                'fired': self._reset
            },
            'beersLawLab.navigationBar.concentrationScreenButton': {
                'fired': self._concentrationlab_activate
            },
            'concentration_lab': {
                'dragStarted': self._concentrationlab_startdrag,
                'dragged': self._concentrationlab_drag,
                'dragEnded': self._concentrationlab_stopdrag,
                'fired': self._concentrationlab_fire,
                'changed': self._concentrationlab_fire,
                'released': self._concentrationlab_fire,
                'popupShown': self._concentrationlab_fire,
                'popupHidden': self._concentrationlab_fire,
                'pressed': self._concentrationlab_fire,
                'endTapToDispense': self._concentrationlab_fire
            },
            'beersLawLab.homeScreen.view.phetButton': {
                'fired': self._visit_menu
            },
            'beersLawLab.navigationBar.homeButton': {
                'fired': self._visit_menu
            },
            'beersLawLab.navigationBar.phetButton': {
                'fired': self._look_menu
            },
            'beersLawLab.sim.barrierRectangle': {
                'fired': self._unlook_menu
            },
            'beersLawLab.homeScreen.view.phetButton.phetMenu.aboutButton':{
                'fired': self._no_update
            },
            'beersLawLab.navigationBar.phetButton.phetMenu.screenshotMenuItem': {
                'fired': self._no_update
            },
            'beersLawLab.navigationBar.phetButton.phetMenu.aboutButton': {
                'fired': self._no_update
            }
        }
      
    def _load_variablesmap(self):
        self._variablesmap = {
            'beersLawLab.beersLawScreen.model.detector.probe.locationProperty':{
                'changed': self._update_magnifier_position
            },
            'beersLawLab.beersLawScreen.model.detector.valueProperty':{
                'changed': self._update_measure
            },
            'beersLawLab.beersLawScreen.model.detector.modeProperty':{
                'changed': self._update_metric
            },
            'beersLawLab.beersLawScreen.model.cuvette.widthProperty':{
                'changed': self._update_flask_width
            },
            'beersLawLab.beersLawScreen.model.light.onProperty': {
                'changed': self._update_light
            },
            'beersLawLab.beersLawScreen.model.light.wavelengthProperty':{
                'changed': self._update_laser_wavelength
            },
            'beersLawLab.beersLawScreen.model.solutionProperty': {
                'changed': self._update_solution
            },
            'beersLawLab.beersLawScreen.model.ruler.locationProperty':{
                'changed':  self._update_ruler
            },
            'beersLawLab.beersLawScreen.view.detectorNode.bodyNode.valueNode':{
                'changed': self._update_measure_display,
                'textChanged': self._update_measure_display
            },
            'beersLawLab.beersLawScreen.view.wavelengthControls.variableWavelengthProperty': {
                'changed': self._update_wavelength_variable
            },
            'beersLawLab.beersLawScreen.view.wavelengthControls.valueDisplay': {
                'textChanged': self._update_laser_wavelength_display
            },
            'beersLawLab.beersLawScreen.solutions.drinkMix.concentrationProperty':{
                'changed': self._update_concentration
            },
            'beersLawLab.beersLawScreen.solutions.copperSulfate.concentrationProperty':{
                'changed': self._update_concentration
            },
            'beersLawLab.beersLawScreen.solutions.cobaltIINitrate.concentrationProperty':{
                'changed': self._update_concentration
            },
            'beersLawLab.beersLawScreen.solutions.cobaltChloride.concentrationProperty':{
                'changed': self._update_concentration
            },
            'beersLawLab.beersLawScreen.solutions.potassiumDichromate.concentrationProperty':{
                'changed': self._update_concentration
            },
            'beersLawLab.beersLawScreen.solutions.potassiumChromate.concentrationProperty':{
                'changed': self._update_concentration
            },
            'beersLawLab.beersLawScreen.solutions.nickelIIChloride.concentrationProperty':{
                'changed': self._update_concentration
            },
            'beersLawLab.beersLawScreen.solutions.potassiumPermanganate.concentrationProperty':{
                'changed': self._update_concentration
            }
        }
        
        
    ##############################
    # Getters    
    #
    def get_bugs(self) -> str:
        """Records all the times where some assertion that should have existed did not in the data

        Returns:
            str: [description]
        """
    def get_learner_id(self) -> str:
        return self._learner_id
    
    def get_simulation_id(self) -> str:
        return self._simulation_id
    
    def get_checkbox_transmittance(self) -> Tuple[dict, dict]:
        """Returns the on values of transmittance checkbox
        
        Returns:
            Tuple[dict, dict]
                on{begin, end}:  beginning and end timestamps of the period when the radio box is set on transmittance
                off{begin, end}: beginning and end timestamps of the period when the radio box is set on absorbance
        """
        return  self._checkbox_transmittance.get_switch_on(), self._wavelength_variable.get_switch_off()
    
    def get_wavelength(self) -> Tuple[list, list]:
        """Looks into the values of the wavelength throughout the simulation
        Returns:
            Tuple[list0, list1]: list0: values of the wavelength of the simulation
                                 list1: timesteps of when those values changed
        """
        return self._wavelength.get_values(), self._wavelength.get_timesteps()
    
    def get_wavelength_radiobox(self) -> Tuple[dict, dict]:
        """Looks into the whether the button 'preset' or 'variable'

        Returns:
            Tuple[dict1, dict2]: 
            on{begin, end}:  beginning and end timestamps of the period when the radio box is set on variable
            off{begin, end}: beginning and end timestamps of the period when the radio box is set on preset
        """
        return self._wavelength_variable.get_switch_on(), self._wavelength_variable.get_switch_off()
    
    def get_wavelength_displayed(self) -> Tuple[list, list]:
        """Looks into the values of the wavelength throughout the simulation as displayed on the probe
        Returns:
            Tuple[list0, list1]: list0: values of the displayed wavelength of the simulation
                                 list1: timesteps of when those values changed
        """
        return self._wavelength.get_values(), self._wavelength.get_timesteps()
    
    def get_width(self) -> Tuple[list, list]:
        """Looks into the values of the width of the flask throughout the simulation 
        Returns:
            Tuple[list0, list]: list0: values of the width of the flask
                                list1: timesteps of when those values changed
        """
        return self._width.get_values(), self._width.get_timesteps()
    
    def get_concentration(self) -> Tuple[list, list]:
        """Looks into the values of the concentration of the solution throughout the simulation
        Returns:
            Tuple[list0, list1]: list0: values of the concentration of the solution
                                 list1: timesteps of when those values changed
        """
        return self._concentration.get_values(), self._concentration.get_timesteps()

    def get_solution(self) -> Tuple[list, list]:
        """Looks into the types of solutions throughout the simulation
        Returns:
            Tuple[list0, list1]: list0: solutions selected 
                                 list1: timestamps when the solutions changed
        """
        return self._solution.get_values(), self._solution.get_timesteps()
    
    def get_measure_recorded(self) -> Tuple[list, list]:
        """Looks into the measure recorded on the probe (absorbance or transmittance)

        Returns:
            Tuple[list0, list1]: list0: measures recorded on the probe
                                 list1: timesteps when those measures changed (which is whenever any of the other relevant components change too) 
        """
        return self._measure.get_values(), self._measure.get_timesteps()
    
    def get_measure_display(self) -> Tuple[list, list]:
        """Looks into the measure displayed on the probe (with the unit) (absorbance or transmittance)
        Returns:
            Tuple[list0, list1]: list0: measures displayed on the probe
                                 list1: timesteps when those measures changed (which is whenever any of the other relevant components change too)
        """
        return self._measure_display.get_values(), self._measure_display.get_timesteps()
    
    def get_metric(self) -> Tuple[list, list]:
        """Looks into when the measure is transmittance or absorbance
        Returns:
            Tuple[list0, list1]: list0: lists of 'transmittance' and 'absorbance'
                                 list1: timesteps of when the metric changes from transmittance and absorbance and vice versa
        """
        return self._metric.get_values(), self._metric.get_timesteps()
    
    def get_magnifier_position(self) -> dict:
        """Tracks the position of the probe catching the laser which measures the transmittance/absorbances
        Returns:
            dict: dragging: start timestamp - values - end timestamps if when this object was dragged
        """
        return self._magnifier.get_dragging()
    
    def get_laser(self) -> dict:
        """Tracks when the laser is activated on or off (action)
        Returns:
            dict: on{begin, end}:  beginning and end timestamps of the period when the laser is on or off
        """
        return self._laser.get_switch_on(), self._laser.get_switch_off()
    
    def get_light(self) -> dict:
        """Tracks when the laser is activated on or off (even when it resets)
        Returns:
            dict: on{begin, end}:  beginning and end timestamps of the period when the light is on or off
        """
        return self._light.get_switch_on()
    
    def get_wl_preset(self) -> dict:
        """Tracks when the preset radio button is activated
        Returns:
            dict: on{begin, end}:  beginning and end timestamps of the period when the button is on and off
        """
        return self._wl_preset.get_switch_on()
    
    def get_wl_variable(self) -> dict:
        """Tracks when the variable radio button is activated
        Returns:
            dict: on{begin, end}:  beginning and end timestamps of the period when the variable is on and off
        """    
        return self._wl_variable.get_switch_on()
    
    def get_wl_slider_minus(self) -> dict:
        """Tracks the times the minus button from the wavelength slider has been fired
        Returns:
            dict{timestamps, values}: values of when that button was fired, and the resulting wavelength value
        """
        return self._wl_slider_minus.get_firing()
    
    def get_wl_slider_plus(self) -> dict:
        """Tracks the times the minus button from the wavelength

        Returns:
            dict{timestamps, values}: values of when that button was fired, and the resulting wavelength value
        """
        return self._wl_slider_plus.get_firing()
    
    def get_wl_slider(self) -> Tuple[dict, dict]:
        """Tracks when the slider is used (dragged) or when the value bar is touched (fired events)
        Returns:
            Tuple[dict, dict]: {begin, end, values} begin and end dragging timestamps + values
                               {values, timestamp} values and timestamps of when the bar was touched
        """
        return self._wl_slider.get_dragging(), self._wl_slider.get_firing()
    
    def get_solution_menu(self) -> Tuple[dict, dict]:
        """Tracks when the menu was visible, and the solution that was selected each time by the beginning and end of the drag. In the firing list, the reset values are also recorded

        Returns:
            Tuple[dict, dict]: {begin, end, values} begin and end dragging timestamps + values
                               {values, timestamp} values and timestamps of when the solution was reset
        """
        return self._solution_menu.get_dragging(), self._solution_menu.get_firing()
    
    def get_concentration_slider_minus(self) -> dict:
        """Tracks the times the minus button from the concentration slider has been fired
        Returns:
            dict{timestamps, values}: values of when that button was fired, and the resulting concentration value
        """
        return self._concentration_slider_minus.get_firing()
    
    def get_concentration_slider_plus(self) -> dict:
        """Tracks the times the plus button from the concentration slider has been fire
        Returns:
            dict{timestamps, values}: values of when that button was fired, and the resulting concentration value
        """
        return self._concentration_slider_plus.get_firing()
    
    def get_concentration_slider(self) -> Tuple[dict, dict]:
        """Tracks when the concentration value was moved, including when the slider was moved, and when the minus and plus buttons were activated
        Returns:
            Tuple[dict, dict]: {begin, end, values} begin and end dragging timestamps + values
                               {values, timestamp} values and timestamps of when the bar was touched 
        """
        return self._concentration_slider.get_dragging(), self._concentration_slider.get_firing()
    
    def get_flask(self) -> dict:
        """Tracks when the width was changed while pulling the flask slider
        Returns:
            dict: {begin, end, values} begin and end dragging timestamps + values of the flask
        """
        return self._flask.get_dragging()
    
    def get_ruler(self) -> dict:
        """Tracks when the ruler was moved, as well as its position

        Returns:
            dict: {begin, end, values} begin and end dragging timestamps + values of the position of the ruler
        """
        return self._ruler.get_dragging()
    
    def get_ruler_position(self) -> Tuple[list, list]:
        """Returns the coordinate of the ruler as it is dragged

        Returns:
            Tuple[list, list]: 
                - values of the ruler position
                - timesteps of when the ruler position was changed
        """
        return self._ruler_position.get_values(), self._ruler_position.get_timesteps()
    
    def get_pdf(self) -> dict:
        """Tracks when the pdf was visible and not visible
        Returns:
            dict: {begin, end} begin and end dragging timestamps of whether the values 
        """
        return self._pdf.get_switch_on(), self._pdf.get_switch_off()
    
    def get_restarts(self) -> list:
        """Returns the list of when the simulation was resetted

        Returns:
            list: [description]
        """
        return self._timestamps_restarts[1:]
    
    def get_concentrationlab_state(self) -> dict:
        """Returns when the concentration lab is in use or not
        Returns:
            dict: {begin, end} begin and end dragging timestamps of whether the concentration lab is in use or not
        """
        return self._concentration_lab_state.get_switch_on()
    
    def get_concentrationlab_actions(self) -> Tuple[dict, dict]:
        """Looks into any actions performed on the concentration lab

        Returns:
            Tuple[dict, dict]:  {begin, end, values} begin and end dragging timestamps + values
                                {values, timestamp} values and timestamps of when any interaction was conducted on the concentration lab
        """
        return self._concentration_actions.get_dragging(), self._concentration_actions.get_firing()
        
    def get_timeline(self) -> Tuple[list, list]:
        """Looks into anything that was recorded into the timeline list. _timeline records any parsed event registered during the interaction with simulation
        Returns:
            Tuple[list, list]: returns the timeline
                               returns the timestamps
        """
        return self._timeline, self._timestamps
    
    def get_active_timeline(self) -> Tuple[list, list]:
        """Returns the list of the active actions conducted on the platform
        Returns:
            Tuple[list, list]: wanted _timeline objects
                               associated timestamps
        """
        active_states = [
            'startdrag_magnifier', 'stopdrag_magnifier', 'startdrag_flask', 'stopdrag_flask',
            'toggle_laser', 'show_pdf', 'hide_pdf', 'toggle_wlvariable', 'toggle_wlpreset', 'startdrag_wl', 'stopdrag_wl', 'touch_wl', 'untouch_wl', 'press_wlminus', 'press_wlplus', 
            'startdrag_concentration', 'stopdrag_concentration', 'touch_concentration', 'untouch_concentration', 'press_concentrationminus', 'press_concentrationplus', 'show_solution', 'hide_Solution', 'select_solution', 'startdrag_ruler', 'enddrag_ruler',
            'toggle_absorbance', 'toggle_transmittance', 'start_concentrationlab', 'startdrag_concentrationlab', 'stopdrag_concentrationlab', 'fire_concentration', 
            'visit_menu'
        ]
        indices = [i for i in list(range(len(self._timeline))) if self._timeline[i] in active_states]
        active_line = [self._timeline[i] for i in indices]
        active_timestamps = [self._timestamps[i] for i in indices]
        return active_line, active_timestamps
    
    def set_permutation(self, permutation: str):
        """Sets the initial permutation (the answer from the ranking task)
        Args:
            permutation (str): permutation in the format 'wxyz', where w, x, y and z are in {0, 1, 2, 3}
        """
        self._permutation = permutation
        
    def get_permutation(self) -> str:
        return self._permutation
    
    def get_last_timestamp(self) -> float:
        return self._last_timestamp
    
    ##############################
    # Parsing the simulation    
    #
    def parse_simulation(self):
        """This function parses the log files that were loaded at initiation, and creates the simulation object.
        """
        e = Event(self._event_df.iloc[0])
        self._start_simulation(e, e.get_timestamp())
        for i in range(len(self._event_df)):
            event = Event(self._event_df.iloc[i])
            # print(event.get_name(), event.get_phetio_id())
            self._filter_event(event)
        self._close_simulation()
        
    def _get_timestamp(self, timestamp:datetime.datetime) -> float:
        """Returns the timestamp as the amount of seconds since the simulation started
        Args:
            timestamp (datetime): timestamp as taken from the logs
        Returns:
            [float]: amount of seconds since the beginning of the simulation
        """
        if self._start_timestamp == -1:
            print('Time stamp not initialised')
        return ((timestamp - self._start_timestamp).total_seconds())
        
    def _debug_event(self, event: Event):
        print(event.get_params())
        print(event.get_children())
            
    def _filter_event(self, event: Event):
        """Goes through all events to process this simulation
        Args:
            event (Event): event object
        """
        event_name = event.get_name()
        pid = event.get_phetio_id()
        
        if 'concentrationScreen' in pid and 'concentrationScreenButton' not in pid:
            pid = 'concentration_lab'
        
        timestamp = self._get_timestamp(event.get_timestamp()   )
        
        try:
            self._event_map[pid][event_name](event, timestamp)
        except KeyError:
            print(pid, event_name)
            exit(1)
        except AssertionError:
            self.debug.append('assertion error: {} {}'.format(pid, event_name))
        
        self._last_timestamp = timestamp
        self._last_event = event_name
        
    def _filter_children(self, event:Event, timestamp: float):
        children = self._process_children(event, [])
        children = pd.DataFrame(children).transpose()
        # if len(children.columns) == 1:
        children = children.transpose() 
        children.columns = [
            'message_index', 'timestamp', 'year', 'month', 'day', 'hour', 'minute', 'second',
            'phetio_id', 'event_name', 'event_type', 'event_component', 'event_params', 'event_children'
        ]
        children = children.sort_values(['year', 'month', 'day', 'hour', 'minute', 'second', 'message_index'])    
        children = children[children['message_index'] != event.get_message_index()]
        
        for i, child in children.iterrows():
            child_event = Event(child)
            event_name = child_event.get_name()
            pid = child_event.get_phetio_id()
            
            if event_name not in self._variablesmap[pid] or pid not in self._variablesmap:
                print(event_name, pid)
            self._variablesmap[pid][event_name](child_event, timestamp)
            
    ##############################
    # Updating the variables
    #
    def _no_update(self, event: Event, timestamp: float):
        'helloworld'
    
    def _update_magnifier_position(self, event: Event, timestamp: float):
        self._magnifier_position.set_state((event.get_params()['newValue']['x'], event.get_params()['newValue']['y']), timestamp)
               
    def _update_flask_width(self, event: Event, timestamp: float):
        self._width.set_state(event.get_params()['newValue'], timestamp)
        
    def _update_measure(self, event: Event, timestamp: float):
        self._measure.set_state(event.get_params()['newValue'], timestamp)
        
    def _update_measure_display(self, event: Event, timestamp:datetime):
        self._measure_display.set_state(event.get_params()['newText'], timestamp)
        
    def _update_laser_wavelength(self, event: Event, timestamp: float):
        self._wavelength.set_state(event.get_params()['newValue'], timestamp)
        
    def _update_laser_wavelength_display(self, event: Event, timestamp: float):
        self._wavelength_display.set_state(event.get_params()['newText'], timestamp)
        
    def _update_concentration(self, event: Event, timestamp: float):
        self._concentration.set_state(event.get_params()['newValue'], timestamp)
        
    def _update_solution(self, event: Event, timestamp: float):
        self._solution.set_state(event.get_params()['newValue'], timestamp)
        
    def _update_ruler(self, event: Event, timestamp: float):
        new_state = event.get_params()['newValue']
        new_state = (new_state['x'], new_state['y']) 
        self._ruler_position.set_state(new_state, timestamp)
        
    def _update_metric(self, event: Event, timestamp: float):
        self._metric.set_state(event.get_params()['newValue'], timestamp)
        
    def _update_light(self, event: Event, timestamp: float):
        self._light.switch(event.get_params()['newValue'], timestamp)
        
    def _update_wavelength_variable(self, event: Event, timestamp: float):
        self._wavelength_variable.switch(event.get_params()['newValue'], timestamp)
        
    ##############################
    # Processing the events
    #
    def _debug(self, event: Event, timestamp: float):
        print(event.get_phetio_id(), event.get_name())
        raise NotImplementedError
    
    def _start_simulation(self, event: Event, timestamp: float):
        if self._start_timestamp == -1:
            self._timeline.append('start_simulation')
            self._timestamps.append(0)
            self._started = 1
            self._start_timestamp = timestamp
            self._timestamps_restarts.append(0)
        else:
            print('simulation already started')
            
    def _start_chemlab(self, event: Event, timestamp: float):
        self._menu_state.switch(0, timestamp)
        self._chemlab_state.switch(1, timestamp)
        self._concentration_lab_state.switch(0, timestamp)
        self._timeline.append('hover_chemlab')
        self._timestamps.append(timestamp)
            
    # magnifier probe catching the laser
    def _magnifier_dragstart(self, event:Event, timestamp:datetime.datetime):
        self._measure.set_state(self._measure.get_state(), timestamp)
        self._magnifier.start_dragging(self._measure.get_state, timestamp)
        self._timeline.append('startdrag_magnifier')
        self._timestamps.append(timestamp)
        
    def _magnifier_drag(self, event:Event, timestamp:datetime.datetime):
        self._filter_children(event, timestamp)
        self._magnifier.is_dragging(self._magnifier_position.get_state(), timestamp)
        self._timeline.append('drag_magnifier')
        self._timestamps.append(timestamp)
        
    def _magnifier_enddrag(self, event:Event, timestamp: float):
        self._magnifier.stop_dragging(self._magnifier_position.get_state(), timestamp)
        self._timeline.append('stopdrag_magnifier')
        self._timestamps.append(timestamp)
        
    # flask
    def _flask_dragstart(self, event:Event, timestamp: float):
        self._flask.start_dragging(self._width.get_state(), timestamp)
        self._timeline.append('startdrag_flask')
        self._timestamps.append(timestamp)
        
    def _flask_drag(self, event: Event, timestamp: float):
        self._filter_children(event, timestamp)
        self._flask.is_dragging(self._width.get_state(), timestamp)
        self._timeline.append('drag_flask')
        self._timestamps.append(timestamp)
        
    def _flask_enddrag(self, event: Event, timestamp: float):
        self._flask.stop_dragging(self._width.get_state(), timestamp)
        self._timeline.append('stopdrag_flask')
        self._timestamps.append(timestamp)
        
    # laser
    def _laser_toggle(self, event: Event, timestamp: float):
        self._filter_children(event, timestamp)
        self._laser.switch(event.get_params()['newValue'], timestamp)
        self._timeline.append('toggle_laser')
        self._timestamps.append(timestamp)
        
    def _pdf_show(self, event: Event, timestamp: float):
        self._pdf.switch(1, timestamp)
        self._timeline.append('show_pdf')
        self._timestamps.append(timestamp)
        
    def _pdf_hide(self, event:Event, timestamp: float):
        self._pdf.switch(0, timestamp)
        self._timeline.append('hide_pdf')
        self._timestamps.append(timestamp)
       
    # wavelength radio buttons
    def _wlvariable_toggle(self, event: Event, timestamp: float):
        self._wl_variable.switch(True, timestamp)
        self._wl_preset.switch(False, timestamp)
        self._timeline.append('toggle_wlvariable')
        self._timestamps.append(timestamp)
        
    def _wlpreset_toggle(self, event: Event, timestamp: float):
        self._filter_children(event, timestamp)
        self._wl_preset.switch(True, timestamp)
        self._wl_variable.switch(False, timestamp)
        self._timeline.append('toggle_wlpreset')
        self._timestamps.append(timestamp)
        
    # wavelength slider
    def _wlslider_startdrag(self, event: Event, timestamp: float):
        self._wl_slider.start_dragging(self._wavelength.get_state(), timestamp)
        self._timeline.append('startdrag_wl')
        self._timestamps.append(timestamp)
        
    def _wlslider_drag(self, event: Event, timestamp: float):
        self._filter_children(event, timestamp)
        self._wl_slider.is_dragging(self._wavelength.get_state(), timestamp)
        self._timeline.append('drag_wl')
        self._timestamps.append(timestamp)
        
    def _wlslider_enddrag(self, event: Event, timestamp: float):
        self._wl_slider.stop_dragging(self._wavelength.get_state(), timestamp)
        self._timeline.append('stopdrag_wl')
        self._timestamps.append(timestamp)
        
    def _wlslider_touch(self, event: Event, timestamp: float):
        self._filter_children(event, timestamp)
        self._wl_slider.start_dragging(self._wavelength.get_state(), timestamp)
        self._timeline.append('touch_wl')
        self._timestamps.append(timestamp)
        
    def _wlslider_untouch(self, event: Event, timestamp: float):
        self._wl_slider.stop_dragging(self._wavelength.get_state(), timestamp)
        self._timeline.append('untouch_wl')
        self._timestamps.append(timestamp)
        
    def _wlslider_minus(self, event: Event, timestamp: float):
        self._filter_children(event, timestamp)
        self._wl_slider_minus.fire(self._wavelength.get_state(), timestamp)
        self._wl_slider.fire(self._wavelength.get_state(), timestamp)
        self._timeline.append('press_wlminus')
        self._timestamps.append(timestamp)
        
    def _wlslider_plus(self, event: Event, timestamp: float):
        self._filter_children(event, timestamp)
        self._wl_slider_plus.fire(self._wavelength.get_state(), timestamp)
        self._wl_slider.fire(self._wavelength.get_state(), timestamp)
        self._timeline.append('press_wlplus')
        self._timestamps.append(timestamp)
        
    def _concentration_startdrag(self, event: Event, timestamp: float):
        self._concentration_slider.start_dragging(self._concentration.get_state(), timestamp)
        self._timeline.append('startdrag_concentration')
        self._timestamps.append(timestamp)
        
    def _concentration_drag(self, event: Event, timestamp: float):
        self._filter_children(event, timestamp)
        self._concentration_slider.is_dragging(self._concentration.get_state(), timestamp)
        self._timeline.append('drag_concentration')
        self._timestamps.append(timestamp)
        
    def _concentration_enddrag(self, event: Event, timestamp: float):
        self._concentration_slider.stop_dragging(self._concentration.get_state(), timestamp)
        self._timeline.append('stopdrag_concentration')
        self._timestamps.append(timestamp)
        
    def _concentration_touch(self, event: Event, timestamp: float):
        self._filter_children(event, timestamp)
        self._concentration_slider.start_dragging(self._concentration.get_state(), timestamp)
        self._timeline.append('touch_concentration')
        self._timestamps.append(timestamp)
        
    def _concentration_untouch(self, event: Event, timestamp: float):
        self._concentration_slider.stop_dragging(self._concentration.get_state(), timestamp)
        self._timeline.append('untouch_concentration')
        self._timestamps.append(timestamp)
        
    def _concentration_minus(self, event: Event, timestamp: float):
        self._filter_children(event, timestamp)
        self._concentration_slider_minus.fire(self._concentration.get_state(), timestamp)
        self._concentration_slider.fire(self._concentration.get_state(), timestamp)
        self._timeline.append('press_concentrationminus')
        self._timestamps.append(timestamp)
        
    def _concentration_plus(self, event: Event, timestamp: float):
        self._filter_children(event, timestamp)
        self._concentration_slider_plus.fire(self._concentration.get_state(), timestamp)
        self._concentration_slider.fire(self._concentration.get_state(), timestamp)
        self._timeline.append('press_concentrationplus')
        self._timestamps.append(timestamp)
        
    def _solution_shown(self, event: Event, timestamp: float):
        self._solution_menu.start_dragging(self._solution.get_state(), timestamp)
        self._timeline.append('show_solution')
        self._timestamps.append(timestamp)
        
    def _solution_hidden(self, event: Event, timestamp: float):
        self._solution_menu.stop_dragging(self._solution.get_state(), timestamp)
        self._timeline.append('hide_solution')
        self._timestamps.append(timestamp)
        
    def _solution_select(self, event: Event, timestamp: float):
        self._filter_children(event, timestamp)
        self._solution_menu.stop_dragging(self._solution.get_state(), timestamp)
        self._timeline.append('select_solution')
        self._timestamps.append(timestamp)
        
    def _ruler_startdrag(self, event: Event, timestamp:datetime):
        self._ruler.start_dragging(self._ruler_position.get_state(), timestamp)
        self._timeline.append('startdrag_ruler')
        self._timestamps.append(timestamp)
        
    def _ruler_drag(self, event: Event, timestamp: float):
        self._filter_children(event, timestamp)
        self._ruler.is_dragging(self._ruler_position.get_state(), timestamp)
        self._timeline.append('drag_ruler')
        self._timestamps.append(timestamp)
        
    def _ruler_stopdrag(self, event: Event, timestamp:datetime):
        self._ruler.stop_dragging(self._ruler_position.get_state(), timestamp)
        self._timeline.append('enddrag_ruler')
        self._timestamps.append(timestamp)
        
    def _absorbance_click(self, event: Event, timestamp: float):
        self._checkbox_absorbance.switch(1, timestamp)
        self._checkbox_transmittance.switch(0, timestamp)
        self._timeline.append('toggle_absorbance')
        self._timestamps.append(timestamp)
        
    def _transmittance_click(self, event: Event, timestamp: float):
        self._checkbox_transmittance.switch(1, timestamp)
        self._checkbox_absorbance.switch(0, timestamp)
        self._timeline.append('toggle_transmittance')
        self._timestamps.append(timestamp)
        
    def _concentrationlab_activate(self, event: Event, timestamp:float):
        self._menu_state.switch(0, timestamp)
        self._concentration_lab_state.switch(1, timestamp)
        self._chemlab_state.switch(0, timestamp)
        self._concentration_actions.fire(0, timestamp)
        self._timeline.append('start_concentrationlab')
        self._timestamps.append(timestamp)
        
    def _concentrationlab_startdrag(self, event: Event, timestamp:float):
        self._concentration_actions.fire(1, timestamp)
        self._timeline.append('startdrag_concentrationlab')
        self._timestamps.append(timestamp)
        
    def _concentrationlab_drag(self, event: Event, timestamp: float):
        self._concentration_actions.fire(0, timestamp)
        self._timeline.append('drag_concentrationlab')
        self._timestamps.append(timestamp)
        
    def _concentrationlab_stopdrag(self, event: Event, timestamp: float):
        self._concentration_actions.fire(-1, timestamp)
        self._timeline.append('stopdrag_concentrationlab')
        self._timestamps.append(timestamp)
        
    def _concentrationlab_fire(self, event: Event, timestamp: float):
        self._concentration_actions.fire(0, timestamp)
        self._timeline.append('fire_concentration')
        self._timestamps.append(timestamp)
        
    def _visit_menu(self, event: Event, timestamp: float):
        self._menu_state.switch(1, timestamp)
        self._timeline.append('visit_menu')
        self._timestamps.append(timestamp)
        
    def _look_menu(self, event: Event, timestamp: float):
        self._menu_state.switch(1, timestamp)
        self._timeline.append('look_menu')
        self._timestamps.append(timestamp)
        
    def _unlook_menu(self, event: Event, timestamp: float):
        self._menu_state.switch(0, timestamp)
        self._timeline.append('unlook_menu')
        self._timestamps.append(timestamp)
        
    def _reset(self, event: Event, timestamp: float):
        self._filter_children(event, timestamp)
        self._laser.switch(False, timestamp)
        self._solution_menu.fire(self._solution.get_state(), timestamp)
        self._checkbox_absorbance.switch(0, timestamp)
        self._checkbox_transmittance.switch(1, timestamp)
        self._timeline.append('reset')
        self._timestamps.append(timestamp)
        self._timestamps_restarts.append(timestamp)
        
    ########## CLose Simulation
    def _close_simulation(self):
        self._wavelength.close(self._last_timestamp)
        self._wavelength_variable.close(self._last_timestamp)
        self._wavelength_display.close(self._last_timestamp)
        self._width.close(self._last_timestamp)
        self._concentration.close(self._last_timestamp)
        self._solution.close(self._last_timestamp)
        self._ruler_position.close(self._last_timestamp)
        
        self._measure.close(self._last_timestamp)
        self._measure_display.close(self._last_timestamp)
        self._metric.close(self._last_timestamp)
        self._magnifier_position.close(self._last_timestamp)
        
        self._checkbox_transmittance.close(self._last_timestamp)
        self._checkbox_absorbance.close(self._last_timestamp)
        self._magnifier.close(self._last_timestamp)
        
        self._laser.close(self._last_timestamp)
        self._light.close(self._last_timestamp)
        self._wl_preset.close(self._last_timestamp)
        self._wl_variable.close(self._last_timestamp)
        self._wl_slider_minus.close(self._last_timestamp)
        self._wl_slider_plus.close(self._last_timestamp)
        self._wl_slider.close(self._last_timestamp)
        
        self._solution_menu.close(self._last_timestamp)
        self._concentration_slider_minus.close(self._last_timestamp)
        self._concentration_slider_plus.close(self._last_timestamp)
        self._concentration_slider.close(self._last_timestamp)
        
        self._flask.close(self._last_timestamp)
        self._ruler.close(self._last_timestamp)
        self._pdf.close(self._last_timestamp)
        
        self._concentration_lab_state.close(self._last_timestamp)
        self._concentration_actions.close(self._last_timestamp)
        self._chemlab_state.close(self._last_timestamp)
        self._menu_state.close(self._last_timestamp)
        
    def close(self):
        self._close_simulation()
        
    def save(self, version='') -> str:
        
        path = '../data/parsed simulations/perm_lid' + str(self._learner_id) + '_t' + self._task + 'v' + version + '_simulation.pkl'
        # path = '//ic1files.epfl.ch/D-VET/Projects/ChemLab/04_Processing/Jade Parsing/' + perm' + self._permutation + '_lid' + str(self._learner_id) + '_t' + self._task + 'v' + version + '_simulation.pkl'
        with open(path, 'wb') as fp:
            dill.dump(self, fp)

        return path
        
    
        
                
                
        
                
                
        
        
        
        
        
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
          