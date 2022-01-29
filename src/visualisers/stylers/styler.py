import os
import yaml
import numpy as np

import bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models import ColumnDataSource, Whisker
from bokeh.sampledata.autompg import autompg as df

class Styler:
    """Processes the plot config files to style the figures
    """
    
    def __init__(self, settings:dict):
        self._settings = settings
        
    def _get_maps(self):
        path = './visualisers/maps/' + self._styler_settings['colour_map'] + '.yaml'
        with open(path, 'r') as f:
            self._cm = yaml.load(f, Loader=yaml.FullLoader)
            
        path = './visualisers/maps/' + self._styler_settings['label_map'] + '.yaml'
        with open(path, 'r') as f:
            self._lm = yaml.load(f, Loader=yaml.FullLoader)
            
    def get_cm(self):
        return dict(self._cm)
    
    def get_lm(self):
        return dict(self._lm)
        
        
    def init_figure(self, xaxis:dict):
        p = figure(
            title=self._styler_settings['title'],
            sizing_mode=self._styler_settings['sizing_mode'],
            y_range=self._styler_settings['ystyle']['range']
        )
        p.title.text_font_size = '25pt'
        p.xaxis.axis_label_text_font_size  = '20pt'
        p.yaxis.axis_label_text_font_size  = '20pt'
        
        
        xticks_labels = dict(zip(xaxis['ticks'], xaxis['labels']))
        xticks_labels = {float(xx)+0.000001:label for xx, label in xticks_labels.items()}
        p.xaxis.ticker = list(xticks_labels.keys())
        p.xaxis.major_label_overrides = xticks_labels
        p.xaxis.axis_label = self._styler_settings['xstyle']['label']
        p.yaxis.axis_label = self._styler_settings['ystyle']['label']
        p.xaxis.major_label_text_font_size = '20px'
        return p
    
    def _get_algo(self, path:str) -> str:
        if '1nn' in path:
            algo = '1nn'
        elif 'rf' in path:
            algo = 'rf'
        elif 'sknn' in path:
            algo = 'sknn'
        elif 'svc' in path:
            algo = 'svc'
        elif 'tsne' in path:
            algo = 'tsne'
        elif 'sgd' in path:
            algo = 'sgd'
        elif 'knn' in path:
            algo = 'knn'
        elif 'adaboost' in path:
            algo = 'adaboost'
        elif 'BiLSTM' in path:
            algo = 'BiLSTM'
        elif 'lstm' in path:
            algo = 'LSTM'
        return algo
    
    def _get_feature(self, path:str) -> str:
        if '1hotactionspan' in path:
            feature = 'vas'
        elif 'raw_aveagg' in path:
            feature= 'vac'
        elif '1hot' in path:
            feature = 'ac'
        elif 'actionspan' in path:
            feature = 'as'
        elif 'colourbreak' in path:
            feature = 'col'
        elif 'simplestate' in path:
            feature = 'simple'
        elif 'sgenc_e50' in path or 'sgenc_e100' in path:
            feature = 'longpw'
        elif 'sgenc_e10' in path or 'sgenc_e15' in path:
            feature = 'shortpw'
        elif 'sgenc' in path:
            feature = 'pw'
        elif 'raw_full' in path and 'stateaction_secondslstm':
            feature = 'seconds_pad'
        elif 'raw_scrop' in path and 'stateaction_secondslstm':
            feature = 'seconds_crop'
        elif 'raw_full' in path and 'stateaction_encodedlstm':
            feature = 'encoded_pad'
        elif 'raw_scrop' in path and 'stateaction_secondslstm':
            feature = 'encoded_crop'
        return feature
    
    def _get_alpha(self, path:str) -> str:
        if 'reproduction' in path:
            return 0.3
        else:
            return 0.9
        
    def _get_label_key(self, path:str) -> str:
        print(path)
        if 'sgenc_e50' in path or 'sgenc_e100' in path:
            feature = 'sgenc_long'
        elif 'sgenc_e10' in path or 'sgenc_e15' in path:
            feature = 'sgenc_short'
        elif 'sgenc_aveagg' in path:
            feature = 'sgenc_aveagg'
        elif 'sgenc_cumulaveagg' in path:
            feature = 'sgenc_cumulaveagg'
        elif 'sgenc_flatagg' in path:
            feature = 'sgenc_flatagg'
        elif 'sgenc' in path:
            feature = 'sgenc_cumulaveagg'
        elif 'colourbreak' in path:
            feature = 'colourbreak'
        elif 'simplestate' in path:
            feature = 'simplestate'
        elif '1hotactionspan' in path:
            feature = 'vas'
        elif 'raw_aveagg' in path:
            feature= 'vac'
        elif '1hot_aveagg' in path:
            feature = '1hot_aveagg'
        elif '1hot' in path:
            feature = '1hot_aveagg'
        elif '1hot_cumulaveagg' in path:
            feature = '1hot_cumulaveagg'
        elif 'actionspan_noagg' in path:
            feature = 'actionspan_noagg'
        elif 'actionspan' in path:
            feature = 'actionspan_normagg'
        elif 'actionspan_normagg' in path:
            feature = 'actionspan_normagg'
        elif 'raw_full' in path and 'stateaction_secondslstm':
            feature = 'seconds_pad'
        elif 'raw_scrop' in path and 'stateaction_secondslstm':
            feature = 'seconds_crop'
        elif 'raw_full' in path and 'stateaction_encodedlstm':
            feature = 'encoded_pad'
        elif 'raw_scrop' in path and 'stateaction_secondslstm':
            feature = 'encoded_crop'
        return feature
        
    def _algofeatures_plot_styling(self, paths:list) -> dict:
        # TODO: use colour map
        alphas = []
        colours = []
        labels = []
        linedashes = []
        
        labels_colours_alpha = {}
        for path in paths:
            alpha = self._get_alpha(path)
            alphas.append(alpha)
            algo = self._get_algo(path)
            feature = self._get_feature(path)
            colour = self._cm[algo][feature]
            colours.append(colour)
            label_key = self._get_label_key(path)
            label = algo + ' with ' + self._lm['abbreviations'][label_key]
            if 'reproduction' in path:
                label += ' (reproduction)'
                linedash = 'dotted'
            else:
                linedash = 'solid'
            linedashes.append(linedash)
            labels.append(label)
            labels_colours_alpha[label] = {
                'colour': colour,
                'alpha': alpha
            }
            
        plot_styling = {
            'alphas': alphas,
            'colours' : colours,
            'labels': labels,
            'labels_colours_alpha' : labels_colours_alpha,
            'linedashes' : linedashes
        }
        return plot_styling
    
    def get_plot_styling(self, paths:list) -> dict:
        if self._styler_settings['style'] == 'algo_features':
            return self._algofeatures_plot_styling(paths)
        