import os
import re
import pickle
import logging
from bokeh.models.glyphs import Patch
import numpy as np
import pandas as pd
from typing import Tuple

from wcwidth import wcswidth

from ml.gridsearches.gridsearch import GridSearch

import bokeh
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models import ColumnDataSource, Whisker
from bokeh.sampledata.autompg import autompg as df

from visualisers.stylers.early_pred_styler import EarlyPredStyler

class EarlyPredPlotter:
    """This class plots nested crossvalidation results
    """
    
    def __init__(self, settings:dict):
        self._settings = settings
        self._styler = EarlyPredStyler(settings)
        
    def _process_path(self, path:str) -> Tuple[str, int]:
        # Process length
        regex = re.compile('l([0-9]+)')
        length = regex.findall(path)
        p = path.replace('_l' + str(length[0]), '')

        # process _date
        date_re = re.compile('.*(202[0-9]_[0-9]+_[0-9]+_[0-9]+/)')
        date = date_re.findall(path)
        p = p.replace(date[0], '')

        return p, int(length[0]), date[0]
        
    def _crawl(self):
        # crawl paths
        xval_path = []
        experiment_path = '../experiments/' + self._settings['experiment']['name'] + '/'
        for (dirpath, dirnames, filenames) in os.walk(experiment_path):
            files = [os.path.join(dirpath, file) for file in filenames]
            xval_path.extend(files)
        kw = self._settings['experiment']['keyword']
        xval_path = [xval for xval in xval_path if kw in xval]
        print('* paths *')
        for path in xval_path:
            print(path)
        print()
        
        nf = []
        for path in xval_path:
            for g in self._settings['plot_style']['xstyle']['groups']:
                if g in path:
                    nf.append(path)
                    continue
        xval_path = nf
        
        # Load xvals
        xvs = {}
        for xv in xval_path:
            path, l, date = self._process_path(xv)
            regex = re.compile('([0-9]+classes)')
            with open(xv, 'rb') as fp:
                if path not in xvs:
                    xvs[path] = {}
                if date not in xvs[path]:
                    xvs[path][date] = {}
                xvs[path][date][l] = pickle.load(fp)
        return xvs
    
    def _crawl_reproduction(self):# crawl paths
        with open('../data/reproduction/early_pred_reproduction.pkl', 'rb') as fp:
            data = pickle.load(fp)
            
        dfs = {}
        for classcase in data:
            for model in data[classcase]:
                for feature in data[classcase][model]:
                    if classcase in self._settings['plot_style']['xstyle']['groups']:
                        path = 'reproduction_' + classcase + '_' + model + '_' + feature
                        dfs[path] = data[classcase][model][feature]
                    
        return dfs
                    
       
    def _create_lineframes(self, xvals: dict):
        xs = []
        means = {}
        stds = {}
        for date in xvals:
            for length in xvals[date]:
                folds = []
                for fold in xvals[date][length]:
                    if fold != 'x' and fold != 'y' and fold != 'optim_scoring' and fold != 'id_indices' and fold != 'limit':
                        if self._settings['plot_style']['carry_on']:
                            folds.append(xvals[date][length][fold]['carry_on_scores'][self._settings['plot_style']['measure']])
                        else:
                            folds.append(xvals[date][length][fold][self._settings['plot_style']['measure']])
                    xs.append(length)
                if length not in means:
                    means[length] = []
                    stds[length] = []
                
                means[length].append(np.mean(folds))
                stds[length].append(np.std(folds))
            
        xs = list(set(xs))
        xs.sort()
        plot_means = [np.mean(means[length]) for length in xs]
        plot_stds = [np.std(stds[length]) for length in xs]
            
        data = pd.DataFrame()
        data['x'] = xs
        data['mean'] = plot_means
        data['std'] = plot_stds
        data['upper'] = data['mean'] + data['std']
        data['lower'] = data['mean'] - data['std']
        data = data.sort_values('x')
        return data
         
    def _create_reproduction_dataframes(self, df):
        lens = [10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 400]
        data = pd.DataFrame()
        data['x'] = lens[0:len(df)]
        data['mean'] = df
        return data
    
    def _save(self, p):
        if self._settings['save']:
            path = '../experiments/' + self._settings['experiment']['name'] 
            path += '/early_prediction_' + '_'.join(self._settings['plot_style']['xstyle']['groups']) + self._settings['plot_style']['style'] + '_' + self._settings['plot_style']['type'] 
            if self._settings['reproduction']:
                path += '_reproduction'
            
            path += '.html'    
            output_file(path, mode='inline')
            save(p)
            
    def _show(self, p):
        if self._settings['show']:
            show(p)
            
    def _multiple_plots(self, data:list, plot_styling:dict):
        glyphs = {
            'line': {}
        }
        
        p = self._styler.init_figure(data[0]['x'])
        print('data', data)
        
        for i in range(len(data)):
            colour = plot_styling['colours'][i]
            label = plot_styling['labels'][i]
            alpha = plot_styling['alphas'][i]
            linedash = plot_styling['linedashes'][i]
            styler = {'colour': colour, 'label': label, 'alpha': alpha, 'linedash': linedash}
            # print(len(data))
            # print(glyphs)
            # print(i)
            # print(styler)
            # print(p)
            # print()
            glyphs, p = self._styler.get_individual_plot(data[i], glyphs, i, styler, p)
        self._styler.add_legend(plot_styling, p)
        self._save(p)
        self._show(p)
        
    def plot_experiment(self):
        dots, parameters, boxplots = [], [], []
        xvs = self._crawl()
        plot_styling = self._styler.get_plot_styling(list(xvs.keys()))
        
        datas = []
        for path in xvs:
            print('path', path)
            data = self._create_lineframes(xvs[path])
            data['name'] = path
            datas.append(data)
        self._multiple_plots(datas, plot_styling)
        
    def plot_reproduction(self):
        dots, parameters, boxplots = [], [], []
        xvs = self._crawl()
        dfs = self._crawl_reproduction()
        
        plot_styling = self._styler.get_plot_styling(list(xvs.keys()) + list(dfs.keys()))
        
        datas = []
        for path in xvs:
            data = self._create_lineframes(xvs[path])
            data['name'] = path
            datas.append(data)
            print(path)
        for path in dfs:
            data = self._create_reproduction_dataframes(dfs[path])
            data['name'] = path
            datas.append(data)
            print(path)
            
        self._multiple_plots(datas, plot_styling)
        
        
            
        
        
    
                    
                
            
                