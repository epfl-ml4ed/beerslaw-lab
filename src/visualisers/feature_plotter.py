import os
import yaml
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Tuple

from ml.gridsearches.gridsearch import GridSearch

import bokeh
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models import ColumnDataSource, Whisker
from bokeh.sampledata.autompg import autompg as df
from bokeh.layouts import gridplot

from sklearn.metrics import confusion_matrix
from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns

from extractors.pipeline_maker import PipelineMaker

class FeaturePlotter:
    def __init__(self, settings:dict):
        """Plotting the distribution of features per label

        Args:
            settings (dict): [description]
        """
        self._name = 'feature plotter'
        self._notation = 'ftrpltr'
        self._settings = settings
        
        self._load_palettes()
        
    def _load_palettes(self):
        with open('./visualisers/maps/concat_labels.yaml', 'rb') as fp:
            self._palette = yaml.load(fp, Loader=yaml.FullLoader)[self._settings['experiment']['class_name']]
        with open('./visualisers/maps/stratifier.yaml', 'rb') as fp:
            self._stratifier_palette = yaml.load(fp, Loader=yaml.FullLoader)
        self._palette.update(self._stratifier_palette)
        
    
    def _get_features(self):
        pipeline = PipelineMaker(self._settings)
        self._sequences, self._labels, self._indices, self._id_dictionary = pipeline.build_data()
        self._index_state = pipeline.get_state_index()
        self._label_map = pipeline.get_label_map()
        indices = []
        states = []
        for i in self._index_state:
            indices.append(self._index_state[i])
            states.append(i)
        self._states = states
        self._indices = indices
        if self._settings['plotter']['ordered_states']:
            self._states.sort()
            
        self._group_states = []
        for group in self._settings['plotter']['groups']:
            self._group_states = self._group_states + [x for x in self._states if group in x]
        self._group_states = self._group_states + [x for x in self._states if x not in self._group_states]
            
            
        
    def _get_feature_name(self, encoder:str):
        if encoder == 'sgenc':
            return 'skipgram'
        if encoder == '1hot':
            return 'action count'
        if encoder == 'actionspan':
            return 'action span'
        
    def _get_ylabel(self, encoder:str):
        label_map = {
            'sgenc': 'embedding',
            '1hot': 'count',
            'actionspan': 'seconds'
        }
        
    def plot_distribution_label(self):
        """Plot distribution per labels
        """
        self._get_features()
        labels = [l for l in self._labels]
        sequences = [s for s in self._sequences]
        for label in np.unique(labels):
            plt.figure(figsize=(15, 10))
            label_name = self._label_map['index_target'][label]
            indices = [i for i in range(len(labels)) if labels[i] == label]
            n_labels = [labels[i] for i in indices]
            seqs = [sequences[i] for i in indices]
            for seq in seqs:
                ordered_seq = [seq[self._index_state[s]] for s in self._states]
                plt.bar(x=self._states, height=ordered_seq, alpha=0.2, color=self._palette[label_name])
            
            feature = self._get_feature_name(self._settings['data']['pipeline']['encoder'])
            
            plt.title('Distribution of ' + feature + ' features for label ' + label_name)
            plt.ylabel(self._get_ylabel(self._settings['data']['pipeline']['encoder']))
            plt.ylim([0, 1])
            plt.xlabel(feature)
            plt.xticks(self._indices, self._states, rotation=90)
            plt.tight_layout()
            if self._settings['save']:
                path = '../reports/' + self._settings['plotter']['report_folder'] + '/figures/distribution_feature_f' + feature
                path += '_c' + self._settings['experiment']['class_name']
                path += '_s' + self._settings['data']['pipeline']['sequencer']
                path += '_tasks' + ''.join(self._settings['data']['pipeline']['concatenator']['tasks'])
                path += '_lab' + label_name
                path +='.svg'
                plt.savefig(path, format='svg')
            if self._settings['show']:
                plt.show()
            else:
                plt.close()
                
            plt.figure(figsize=(15, 10))
            for seq in seqs:
                ordered_seq = [seq[self._index_state[s]] for s in self._group_states]
                plt.bar(x=self._group_states, height=ordered_seq, alpha=0.2, color=self._palette[label_name])
            
            feature = self._get_feature_name(self._settings['data']['pipeline']['encoder'])
            
            plt.title('Grouped distribution of ' + feature + ' features for label ' + label_name)
            plt.ylabel(self._get_ylabel(self._settings['data']['pipeline']['encoder']))
            plt.ylim([0, 1])
            plt.xlabel(feature)
            plt.xticks(self._indices, self._group_states, rotation=90)
            plt.tight_layout()
            if self._settings['save']:
                path = '../reports/' + self._settings['plotter']['report_folder'] + '/figures/grouped_distribution_feature_f' + feature
                path += '_c' + self._settings['experiment']['class_name']
                path += '_s' + self._settings['data']['pipeline']['sequencer']
                path += '_tasks' + ''.join(self._settings['data']['pipeline']['concatenator']['tasks'])
                path += '_lab' + label_name
                path +='.svg'
                plt.savefig(path, format='svg')
            if self._settings['show']:
                plt.show()
            else:
                plt.close()
            
                
            
            
            
            
            
        
          
    
