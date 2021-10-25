import yaml
import json
import pickle

import bokeh
from extractors.parser.simulation_parser import Simulation
from extractors.sequencer.plotter_sequencer import PlotterSequencing

class Timeline:
    """
    This class is created to plot the parsed files as timelines, to represent student interactions in a human-readable way.
    Particularly, this timeline is aimed at reading parsed files from the beer's law lab [https://phet.colorado.edu/sims/html/beers-law-lab/latest/beers-law-lab_en.html]
    """
    def __init__(self, settings: dict, sequencer: PlotterSequencing):
        self._name = 'timeline'
        self._notation = 'tmln'
        self._settings = settings

    def _extract_xs(self, component):
        """
        From a component of the simulation, extract the timecoordinates for 
        the rectangle in bokeh.
        
        Args:
            component: list (of timestamps) or dict (on and off)
        """
        raise NotImplementedError

    def _create_timeline(self):
        """
        Creates the timeline
        """
        raise NotImplementedError
        

    