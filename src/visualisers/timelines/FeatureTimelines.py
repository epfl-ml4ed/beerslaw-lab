from xml.sax.handler import feature_string_interning
from bokeh.models.tools import Toolbar
from sklearn.pipeline import Pipeline
import yaml
import json
import pickle
from typing import Tuple

import numpy as np

import bokeh
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, HoverTool, glyph
from bokeh.plotting import figure, output_file, show, save
from bokeh.io import export_svg

from extractors.pipeline_maker import PipelineMaker
from extractors.parser.simulation_parser import Simulation
from extractors.sequencer.flat.plotter_sequencer import PlotterSequencing

from visualisers.timelines.Timeline import Timeline

class FeatureTimeline(Timeline):
    """
    This class is created to plot the parsed files as feature timelines, to represent students interactions across times to see the evolution of features
    across times.
    Particularly, this timeline is aimed at reading parsed files from the beer's law lab [https://phet.colorado.edu/sims/html/beers-law-lab/latest/beers-law-lab_en.html]

    """
    def __init__(self, settings: dict):
        self._name = 'featuretimeline'
        self._notation = 'fttmln'
        self._settings = dict(settings)
        
        self._load_palette()
        self._load_plotter()
        

    def _load_palette(self):
        with open('./visualisers/maps/colourtimeline_cm.yaml', 'rb') as fp:
            self._palette = yaml.load(fp, Loader=yaml.FullLoader)

    def _load_features(self):
        self._settings['paths'] = {}
        self._settings['experiment'] = {
            'root_name': 'feature_plotting',
            'name': '',
            'class_name': 'vector_labels'
        }
        self._settings['data'] = {
            'min_length': 10,
            'pipeline': {
                'sequencer': 'simplestate_secondslstm',
                'concatenator': {
                    'type': 'chemconcat',
                    'tasks': ['2']
                },
                'demographic_filter': 'chemlab',
                'event_filter': 'nofilt',
                'break_filter': -1,
                'break_threshold': 0.6,
                'adjuster': 'tscrp',
                'encoder': -1,
                'aggregator': -1,
            },
            'adjuster': {
                'limit': -1
            },
            'filters': {
                'interactionlimit': 10
            }
        }
        self._settings['ML'] = {'pipeline':{'scorer':{}}}

        if self._settings['features'] == 'simplestate_secondslstm':
            self._settings['sequencer'] = 'simplestate_secondslstm'
            self._settings['data']['break_filter'] = 'cumulseconds'
            self._settings['data']['aggregator'] = 'minmax'
            self._settings['data']['encoder'] = 'raw'
        if self._settings['features'] == 'simplestate_actioncount':
            self._settings['sequencer'] = 'simplestate_secondsflat'
            self._settings['data']['break_filter'] = 'nobrfilt'
            self._settings['data']['aggregator'] = 'aveagg'
            self._settings['data']['encoder'] = '1hot'
        if self._settings['features'] == 'simplestate_actionspan':
            self._settings['sequencer'] = 'simplestate_secondsflat'
            self._settings['data']['break_filter'] = 'nobrfilt'
            self._settings['data']['aggregator'] = 'normagg'
            self._settings['data']['encoder'] = 'actionspan'
        
    def collect_sequences(self):
        config = dict(self._settings)
        features = {}
        for i in range(200):
            config['data']['adjuster']['limit'] = i
            pipeline = Pipeline(config)
            sequences, labels, indices, id_dictionary = pipeline.build_data()
            features[i] = {
                'sequences': sequences,
                'indices': indices,
                'labels': labels
            }
        return features

    def dataframe_per_user(self, features):
        

    def create_timelines(self):
        all_features = self._collect_sequences()

        

        





        glyphs = {}
        title = 'Timeline for student ' + sim.get_learner_id() + ' with permutation ' + sim.get_permutation() + ' for task ' + str(sim.get_task())
        plot = self._init_figure(title)

        glyphs, plot = self._plot_wavelength(sim, glyphs, plot)
        glyphs, plot = self._plot_solutioncolour(sim, glyphs, plot)
        glyphs, plot = self._plot_actions(sim, glyphs, plot, self._plotter)
        glyphs, plot = self._plot_ruler(sim, glyphs, plot, self._plotter)
        glyphs, plot = self._plot_measuring(sim, glyphs, plot, self._plotter)
        glyphs, plot = self._frame_timeline(glyphs, plot, sim)

        plot.legend.click_policy="hide"

        if self._settings['saveimg']:
            plot.output_backend = 'svg'
            path = '../reports/' + self._settings['image']['report_folder']
            path += '/colour timelines/p' + sim.get_permutation() 
            path += '_t' + str(sim.get_task())
            path += '_l' + sim.get_learner_id() + '.svg'
            export_svg(plot, filename=path)

        if self._settings['save']:
            path = '../reports/' + self._settings['image']['report_folder']
            path += '/colour timelines/p' + sim.get_permutation() 
            path += '_t' + str(sim.get_task())
            path += '_l' + sim.get_learner_id() + '.html'
            save(plot, filename=path)

        if self._settings['show']:
            show(plot)







        
        











