import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Tuple

from ml.gridsearches.gridsearch import GridSearch

import bokeh
from bokeh.io import export_svg, export_png
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models import ColumnDataSource, Whisker
from bokeh.sampledata.autompg import autompg as df
from bokeh.layouts import gridplot

from visualisers.stylers.full_sequences_styler import FullStyler
from matplotlib import pyplot as plt

class NestedXValPlotter:
    """This class plots nested crossvalidation results
    """
    
    def __init__(self, settings:dict):
        self._settings = settings
        self._styler = FullStyler(settings)
        
    def _crawl(self):
        # crawl paths
        xval_path = []
        experiment_path = '../experiments/' + self._settings['experiment']['name'] + '/'
        for (dirpath, dirnames, filenames) in os.walk(experiment_path):
            files = [os.path.join(dirpath, file) for file in filenames]
            xval_path.extend(files)
        kw = self._settings['experiment']['keyword']
        xval_path = [xval for xval in xval_path if kw in xval]
        xval_path = [xval for xval in xval_path if 'exclude' not in xval]
        xval_path = [xval for xval in xval_path if '.pkl' in xval]
        
        # Load xvals
        xvs = {}
        for xv in xval_path:
            with open(xv, 'rb') as fp:
                print(xv)
                xvs[xv] = {
                    'data': pickle.load(fp)
                }
        return xvs
    
    def _crawl_reproduction(self):# crawl paths
        xval_path = []
        experiment_path = self._settings['experiment']['reproduction_path']
        for (dirpath, dirnames, filenames) in os.walk(experiment_path):
            files = [os.path.join(dirpath, file) for file in filenames]
            xval_path.extend(files)
        kw = 'reproduction.csv'
        df_path = [xval for xval in xval_path if kw in xval]
        
        
        # Load df
        dfs = {}
        for df in df_path:
            dfs[df] = {
                'data': pd.read_csv(df, sep=';', index_col=0)
            }
        return dfs
        
    def _create_dataframes(self, gs: GridSearch):
        """Generates the dataframes used to plot the nested xval from a gridsearch object

        Args:
            gs ([type]): [description]

        Returns:
            [type]: [description]
        """
        # dots dataframe
        dots = {}
        params = []
        for fold in gs:
            if fold != 'x' and fold != 'y' and fold != 'optim_scoring' and fold != 'indices':
                dots[fold] = {}
                dots[fold]['data'] = gs[fold][self._settings['plot_style']['measure']]
                for parameter in gs[fold]['best_params']:
                    param = parameter.replace('_', ' ')
                    if 'score' not in param and 'fold' not in param and 'index' not in param:
                        dots[fold][param] = str(gs[fold]['best_params'][parameter])
                        params.append(param.replace('_', ' '))
                dots[fold]['fold'] = fold
        dots_df = pd.DataFrame(dots).transpose()
        
        
        # statistics
        q1 = float(dots_df['data'].quantile(q=0.25))
        q2 = float(dots_df['data'].quantile(q=0.5))
        q3 = float(dots_df['data'].quantile(q=0.75))
        mean = float(dots_df['data'].mean())
        std = float(dots_df['data'].std())
        iqr = q3 - q1
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr
        
        # boxplot dataframe
        boxplot = pd.DataFrame()
        boxplot['q1'] = [q1]
        boxplot['lower_error'] = [mean - std]
        boxplot['median'] = [q2]
        boxplot['mean'] = [mean]
        boxplot['std'] = std
        boxplot['upper_error'] = [mean + std]
        boxplot['q3'] = [q3]
        boxplot['upper'] = [upper]
        boxplot['lower'] = [lower]
        
        print(dots_df)

        return dots_df, set(list(params)), boxplot
    
    def _create_reproduction_dataframes(self, df):
        dots = {}
        params = []
        for fold in range(10):
            dots[fold] = {}
            dots[fold]['data'] = df.iloc[fold]['auc']
            
            dots[fold]['algorithm'] = df.iloc[fold]['algorithm']
            params.append('algorithm')
            
            dots[fold]['reproduction'] = 'EDM2021'
            params.append('reproduction')
            
            dots[fold]['fold'] = fold
            
        dots_df = pd.DataFrame(dots).transpose()
        
        # statistics
        q1 = float(dots_df['data'].quantile(q=0.25))
        q2 = float(dots_df['data'].quantile(q=0.5))
        q3 = float(dots_df['data'].quantile(q=0.75))
        mean = float(dots_df['data'].mean())
        std = float(dots_df['data'].std())
        iqr = q3 - q1
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr
        
        # boxplot dataframe
        boxplot = pd.DataFrame()
        boxplot['q1'] = [q1]
        boxplot['lower_error'] = [mean - std]
        boxplot['median'] = [q2]
        boxplot['mean'] = [mean]
        boxplot['std'] = std
        boxplot['upper_error'] = [mean + std]
        boxplot['q3'] = [q3]
        boxplot['upper'] = [upper]
        boxplot['lower'] = [lower]
        
        return dots_df, set(list(params)), boxplot
        
    def _save(self, p, extension_path=''):

        if self._settings['saveimg'] and extension_path == '':
            path = '../experiments/' + self._settings['experiment']['name'] 
            path += '/full_sequence_' + '_'.join(self._settings['plot_style']['xstyle']['groups']) + self._settings['plot_style']['style'] + '_' + self._settings['plot_style']['type'] 
            if self._settings['reproduction']:
                path += '_reproduction'
            
            path += '.svg'
            p.output_backend = 'svg'
            export_svg(p, filename=path)    
            save(p)
            
        elif self._settings['saveimg'] and extension_path !='':
            path = '../experiments/' + self._settings['experiment']['name'] 
            path += '/full_sequence_' + '_'.join(self._settings['plot_style']['xstyle']['groups']) + extension_path
            path += '.svg'    

            p.output_backend = 'svg'
            export_svg(p, filename=path)   
            save(p)
            

        if self._settings['savepng'] and extension_path == '':
            path = '../experiments/' + self._settings['experiment']['name'] 
            path += '/full_sequence_' + '_'.join(self._settings['plot_style']['xstyle']['groups']) + self._settings['plot_style']['style'] + '_' + self._settings['plot_style']['type'] 
            if self._settings['reproduction']:
                path += '_reproduction'
            
            path += '.png'
            export_png(p, filename=path)    
            save(p)
            
        elif self._settings['savepng'] and extension_path !='':
            path = '../experiments/' + self._settings['experiment']['name'] 
            path += '/full_sequence_' + '_'.join(self._settings['plot_style']['xstyle']['groups']) + extension_path
            path += '.png'    

            export_png(p, filename=path)   
            save(p)
            

        if self._settings['save'] and extension_path == '':
            path = '../experiments/' + self._settings['experiment']['name'] 
            path += '/full_sequence_' + '_'.join(self._settings['plot_style']['xstyle']['groups']) + self._settings['plot_style']['style'] + '_' + self._settings['plot_style']['type'] 
            if self._settings['reproduction']:
                path += '_reproduction'
            
            path += '.html'    
            output_file(path, mode='inline')
            save(p)
            
        elif self._settings['save'] and extension_path !='':
            path = '../experiments/' + self._settings['experiment']['name'] 
            path += '/full_sequence_' + '_'.join(self._settings['plot_style']['xstyle']['groups']) + extension_path
            
            path += '.html'    
            output_file(path, mode='inline')
            save(p)
            
    def _show(self, p):
        if self._settings['show']:
            show(p)
            
    def _multiple_plots(self, dots:list, param:list, boxplot:list, xaxis:dict, plot_styling:dict):
        glyphs = {
            'datapoints': {},
            'upper_moustache': {},
            'lower_moustache': {},
            'upper_rect': {},
            'lower_rect': {}
        }
        
        p = self._styler.init_figure(xaxis)
        for i in range(len(dots)):
            x = xaxis['position'][i]
            colour = plot_styling['colours'][i]
            label = plot_styling['labels'][i]
            alpha = plot_styling['alphas'][i]
            styler = {'colour': colour, 'label': label, 'alpha': alpha}
            glyphs, p = self._styler.get_individual_plot(dots[i], param[i], boxplot[i], glyphs, x, styler, p)
        self._styler.add_legend(plot_styling, p)
        self._save(p)
        self._show(p)
        
    def _plot_parameter_distribution(self, dots, parameters, extension_path='parameters'):
        params = {}
        for i, dot in enumerate(dots):
            print('dot', dot)
            param = parameters[i]
            for p in param:
                if p not in params:
                    params[p] = []
                params[p] = list(dot[p])
                    
        plots = []
        for p in params:
            plots.append(self._styler._individual_parameter_countplot(params[p], p, self._styler.get_random_colour()))
        
        modulo = len(plots) % self._settings['plot_style']['ncols']
        if modulo != 0:
            plots = plots + [None for n in range(self._settings['plot_style']['ncols'] - modulo)]
                
        grids = []
        temp = []
        while len(plots) != 0:
            temp.append(plots.pop())
            if len(plots) % self._settings['plot_style']['ncols'] == 0:
                grids.append(temp)
                temp = []
        if len(temp) != 0:
            grids.append(temp)
            
        grid = gridplot(grids)
        self._save(grid, extension_path=extension_path)
        self._show(grid)
        
    def plot_parameters(self):
        dots, parameters, boxplots = [], [], []
        xvs = self._crawl()
        xvs, x_axis = self._styler.get_x_styling(xvs)
        plot_styling = self._styler.get_plot_styling(x_axis['paths'])
        for path in x_axis['paths']:
            xv = xvs[path]['data']
            d, p, b = self._create_dataframes(xv)
            dots.append(d)
            parameters.append(p)
            boxplots.append(b)
        self._plot_parameter_distribution(dots, parameters)
        
    def plot_separate_parameters(self):
        dots, parameters, boxplots = [], [], []
        xvs = self._crawl()
        print(xvs)
        xvs, x_axis = self._styler.get_x_styling(xvs)
        plot_styling = self._styler.get_plot_styling(x_axis['paths'])
        for i, path in enumerate(x_axis['paths']):
            xv = xvs[path]['data']
            d, p, b = self._create_dataframes(xv)
            dots = [d]
            parameters = [p]
            boxplots = [b]
            feat = self._styler._get_feature(path)
            algo = self._styler._get_algo(path)
            ext_path = 'parameters_' + feat + '_' + algo
            self._plot_parameter_distribution(dots, parameters, extension_path=ext_path)
        
        
    def plot_experiment(self):
        means = []
        dots, parameters, boxplots = [], [], []
        xvs = self._crawl()
        xvs, x_axis = self._styler.get_x_styling(xvs)
        print(x_axis)
        plot_styling = self._styler.get_plot_styling(x_axis['paths'])
        for path in x_axis['paths']:
            xv = xvs[path]['data']
            d, p, b = self._create_dataframes(xv)
            means.append(b.iloc[0]['mean'])
            print('****')
            print(path)
            print(b)
            print()
            dots.append(d)
            parameters.append(p)
            boxplots.append(b)
            
        print(np.mean(means))
        
        self._multiple_plots(dots, parameters, boxplots, x_axis, plot_styling)

    def plot_seed_experiments(self):
        means = []
        stds = []
        seeds = []
        dots, parameters, boxplots = [], [], []
        xvs = self._crawl()
        xvs, x_axis = self._styler.get_x_styling(xvs)
        print(x_axis)
        plot_styling = self._styler.get_plot_styling(x_axis['paths'])
        for path in x_axis['paths']:
            print(path)
            xv = xvs[path]['data']
            d, p, b = self._create_dataframes(xv)
            means.append(b.iloc[0]['mean'])
            stds.append(b.iloc[0]['std'])
            seeds.append(d.iloc[0]['seed'])
            
        sort_indices = np.argsort(means)
        ordered_means = [means[idx] for idx in sort_indices]
        ordered_stds = [stds[idx] for idx in sort_indices]

        ordered_mins = np.array(ordered_means) - np.array(ordered_stds)
        ordered_maxs = np.array(ordered_means) + np.array(ordered_stds)

        # x = list(range(len(ordered_means)))
        # plt.figure(figsize=(16, 4))
        # plt.plot(x, ordered_means, color='yellowgreen')
        # plt.fill_between(x, ordered_mins, ordered_maxs, color='yellowgreen', alpha=0.3)
        # plt.xticks(x, seeds)
        # plt.ylim([0, 1])
        # if self._settings['show']:
        #     plt.show()


        x = list(range(len(ordered_means)))
        large_markers = [10 for _ in ordered_mins]
        small_markers = [5 for _ in ordered_mins]
        plt.figure(figsize=(16, 4))
        plt.scatter(x, ordered_means, s=large_markers, color='yellowgreen')
        plt.scatter(x, ordered_mins, s=small_markers, color='yellowgreen', alpha=0.3)
        plt.scatter(x, ordered_maxs, s=small_markers, color='yellowgreen', alpha=0.3)
        plt.xticks(x, seeds)
        plt.ylim([0, 1])
        if self._settings['show']:
            plt.show()


        print(np.mean(means))
        
    def plot_reproduction(self):
        dots, parameters, boxplots = [], [], []
        
        dfs = self._crawl_reproduction()
        xvs = self._crawl()
        xvs.update(dfs)
        df, x_axis = self._styler.get_x_styling(xvs)
        plot_styling = self._styler.get_plot_styling(x_axis['paths'])
        
        for path in x_axis['paths']:
            xv = xvs[path]['data']
            if 'reproduction' not in path:
                d, p, b = self._create_dataframes(xv)
            else:
                d, p, b = self._create_reproduction_dataframes(xv)
            dots.append(d)
            parameters.append(p)
            boxplots.append(b)
            
        self._multiple_plots(dots, parameters, boxplots, x_axis, plot_styling)
            
        
        
    
                    
                
            
                