import yaml
import json
import pickle
from typing import Tuple
import numpy as np

from extractors.parser.simulation_parser import Simulation
from extractors.sequencer.flat.plotter_sequencer import PlotterSequencing

import bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models import ColumnDataSource, Whisker
from bokeh.sampledata.autompg import autompg as df

class Timeline:
    """
    This class is created to plot the parsed files as timelines, to represent student interactions in a human-readable way.
    Particularly, this timeline is aimed at reading parsed files from the beer's law lab [https://phet.colorado.edu/sims/html/beers-law-lab/latest/beers-law-lab_en.html]
    """
    def __init__(self, settings: dict, sequencer: PlotterSequencing):
        self._name = 'timeline'
        self._notation = 'tmln'
        self._settings = settings

    def _extract_values_xs(self, timestamps: list) -> Tuple[list, list]:
        """
        Returns 
            - the middle coordinates for the rectangle function in bokeh
            - the width of each of the rectangles to be drawn
        """
        ts0 = [x for x in timestamps[:-1]]
        ts1 = [x for x in timestamps[1:]]

        rect_width = np.array(ts1) - np.array(ts0)
        middle_coord = (rect_width / 2) + np.array(ts0)
        return rect_width, middle_coord

    def _retrieve_beginends(self, label:str, begins:list, ends:list, labels:list) -> Tuple[list, list]:
        """Gives the beginning and end timestamps of a specific label
        Args:
            label (str): label of interest
            begins (list): beginning timestamps of all labels
            ends (list): end timestamps of all labels
            labels (list): list of labels

        Returns:
            Tuple[list, list]: 
                bs: beginning timestamps of all the action of label
                es: end timestamps of all the action of label
        """
        indices = [i for i in range(len(labels)) if label == labels[i]]
        bs = [begins[idx] for idx in indices]
        es = [ends[idx] for idx in indices]
        return bs, es

    def _extract_beginsends_xs(self, begins:dict, ends:dict) -> Tuple[list, list]:
        """Returns the middle points of the bars, as well as their width (time)

        Args:
            begins (dict): begin timestamps
            ends (dict): end timestamps

        Returns:
            Tuple[list, list]: 
                widths:
        """
        widths = np.array(ends) - np.array(begins)
        middle_coords = (np.array(widths) / 2) + np.array(begins)
        return widths, middle_coords

    def _create_timeline(self):
        """
        Creates the timeline
        """
        raise NotImplementedError

    def _init_figure(self, title):
        p = figure(
            title=title,
            sizing_mode=self._settings['plot']['sizing_mode'],
            x_range=self._settings['plot']['xrange']
        )
        p.title.text_font_size = '25pt'
        p.xaxis.axis_label_text_font_size  = '15pt'
        p.yaxis.axis_label_text_font_size  = '15pt'
        
        p.xaxis.axis_label = self._settings['plot']['xlabel']
        p.yaxis.axis_label = self._settings['plot']['ylabel']
        return p

        

    