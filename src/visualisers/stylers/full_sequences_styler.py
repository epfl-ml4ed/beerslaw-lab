import os
import yaml
import numpy as np
import pandas as pd
from typing import Tuple

import bokeh
from collections import Counter

from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool, Whisker
from bokeh.sampledata.autompg import autompg as df
from bokeh.io import output_file, show
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap

from visualisers.stylers.styler import Styler

class FullStyler(Styler):
    """Processes the plot config files to style the boxplot figures from the full sequences
    """
    
    def __init__(self, settings:dict):
        self._settings = dict(settings)
        self._styler_settings = dict(settings['plot_style'])
        self._get_maps()
        self._load_palettes()
        
    def _load_palettes(self):
        self._random_palette = [
            'maroon', 'orangered', 'saddlebrown', 'goldenrod', 'olive', 'yellowgreen', 'forestgreen', 'teal', 'lightseagreen', 'paleturquoise', 'darkturquoise', 'cadetblue', 'deepskyblue', 'lightskyblue', 'steelblue', 'dodgerblue', 'cornflowerblue', 'royalblue', 'slateblue', 'mediumorchid', 'plum', 'orchid', 'deeppink'
        ]
        
    def get_random_colour(self):
        return self._random_palette[np.random.randint(len(self._random_palette))]
        
    def _individual_parameter_countplot(self, parameters:list, param_name:str, colour:str):
        p = figure(
            x_range=list(set(parameters)), 
            title=param_name,
            sizing_mode=self._styler_settings['sizing_mode']
        )
            
        p.title.text_font_size = '15pt'
        p.xaxis.axis_label_text_font_size  = '2pt'
        p.yaxis.axis_label_text_font_size  = '2pt'
        
        params = Counter(parameters)
        labels, heights = list(params.keys()), list(params.values())
        source = ColumnDataSource(data=dict(params=labels, counts=heights))
        p.vbar(x=labels, top=heights, width=0.9, color=colour)
        
        return p

    def _individual_boxplot(self, dots:pd.DataFrame, param:pd.DataFrame, boxplot:pd.DataFrame, glyphs:dict, x:float, styler:dict, p:bokeh.models.glyphs):
        dots['x'] = x
        boxplot['x'] = x
        
        # segment
        bx = pd.concat([boxplot, boxplot])
        bx['x'] = [x, x]
        bx['y0'] = [boxplot.iloc[0]['upper'], boxplot.iloc[0]['lower']]
        bx['y1'] = [boxplot.iloc[0]['q3'], boxplot.iloc[0]['q1']]
        bx['bar_length'] = self._settings['plot_style']['bar_length']
        
        p.segment('x', 'y0', 'x', 'y1', line_color="black", source=bx)
        
        # quartiles
        boxplot['bar_length'] = 0.5
        glyphs['upper_rect'][x] = p.vbar('x', 'bar_length', 'median', 'q3', fill_color=styler['colour'], alpha=styler['alpha'], line_color="black", source=boxplot)
        glyphs['lower_rect'][x] = p.vbar('x', 'bar_length', 'q1', 'median', fill_color=styler['colour'], alpha=styler['alpha'], line_color="black", source=boxplot)
        # moustaches
        glyphs['lower_moustache'][x] = p.rect('x', 'lower', 0.2, 0.0001, fill_color="black", alpha=styler['alpha'], line_color="black", source=boxplot)
        glyphs['upper_moustache'][x] = p.rect('x', 'upper', 0.2, 0.0001, fill_color="black", alpha=styler['alpha'], line_color="black", source=boxplot)
        
        p.add_tools(HoverTool(renderers=[
            glyphs['upper_moustache'][x],
            glyphs['lower_moustache'][x],
            glyphs['lower_rect'][x],
            glyphs['upper_rect'][x]
        ], tooltips=[
            ('lower whisker',"@lower"),
            ('1st quartile', "@q1"),
            ('median', "@median"),
            ('3rd quartile', "@q3"),
            ('upper whisker', "@upper")
        ], mode='mouse'))
        
        # folds
        glyphs['datapoints'][x] = p.circle(x='x', y='data', radius=0.007, source=dots, alpha=styler['alpha'], color=styler['colour'])
        tooltips = [
            ('FOLD', "@fold"),
            ('SCORE', "@data"),
        ]
        
        for parameter in param:
            tooltips.append(('   ' + parameter, "@{" + parameter + "}"))
        p.add_tools(HoverTool(renderers=[glyphs['datapoints'][x]], tooltips=tooltips, mode='mouse'))

        return glyphs, p
    
    def _individual_errorplot(self, dots:pd.DataFrame, param:pd.DataFrame, boxplot:pd.DataFrame, glyphs:dict, x:float, styler:dict, p: bokeh.models.glyphs):
        dots['x'] = x
        boxplot['x'] = x
        
        # segment
        bx = pd.concat([boxplot, boxplot])
        bx['x'] = [x, x]
        bx['y0'] = [boxplot.iloc[0]['lower_error'], boxplot.iloc[0]['mean']]
        bx['y1'] = [boxplot.iloc[0]['mean'], boxplot.iloc[0]['upper_error']]
        bx['bar_length'] = self._styler_settings['bar_length']
        p.segment('x', 'y0', 'x', 'y1', line_color="black", source=bx)
        
        # moustaches
        glyphs['lower_moustache'][x] = p.rect('x', 'lower_error', self._styler_settings['bar_length'], 0.0001, fill_color="black", alpha=0.3, line_color="black", source=boxplot)
        glyphs['upper_rect'][x] = p.rect('x', 'mean', self._styler_settings['bar_length']*0.66, 0.0001, fill_color="black", alpha=styler['alpha'], line_color="black", source=boxplot)
        glyphs['upper_moustache'][x] = p.rect('x', 'upper_error', self._styler_settings['bar_length'], 0.0001, fill_color="black", alpha=styler['alpha'], line_color="black", source=boxplot)
        
        p.add_tools(HoverTool(renderers=[
            glyphs['upper_moustache'][x],
            glyphs['lower_moustache'][x],
            glyphs['upper_rect'][x]
        ], tooltips=[
            ('upper std',"@upper_error"),
            ('mean', "@mean"),
            ('lower std', "@lower_error"),
        ], mode='mouse'))
        
        # folds
        glyphs['datapoints'][x] = p.circle(x='x', y='data', radius=self._styler_settings['radius'], source=dots, alpha=styler['alpha'], color=styler['colour'])
        tooltips = [
            ('FOLD', "@fold"),
            ('SCORE', "@data"),
        ]
        
        for parameter in param:
            tooltips.append(('   ' + parameter, "@{" + parameter + "}"))
        p.add_tools(HoverTool(renderers=[glyphs['datapoints'][x]], tooltips=tooltips, mode='mouse'))

        return glyphs, p
    
    def get_individual_plot(self, dots:pd.DataFrame, param:pd.DataFrame, boxplot:pd.DataFrame, glyphs:dict, x:float, styler:dict, p: bokeh.models.glyphs):
        if self._styler_settings['type'] == 'boxplot':
            return self._individual_boxplot(dots, param, boxplot, glyphs, x, styler, p)
        elif self._styler_settings['type'] == 'errorplot':
            return self._individual_errorplot(dots, param, boxplot, glyphs, x, styler, p)
    
    def _x_groupings(self, xvals:dict) -> Tuple[dict, dict]:
        groups = []
        paths = list(xvals.keys())
        
        # group formation
        for group in self._styler_settings['xstyle']['groups']:
            g = [gg for gg in paths if group in gg]
            g_label = [self._get_algo(gg) + '_' + self._get_feature(gg) for gg in g]
            g_indices = np.argsort(g_label)
            g = [g[i] for i in g_indices]
            groups.append(g)
            
            
        # get x's
        x = 0
        xpos = []
        xticks = []
        xlabels = []
        xpaths = []
        
        for i, group in enumerate(self._styler_settings['xstyle']['groups']):
            xs = [x]
            for path in groups[i]:
                x += self._styler_settings['boxplot_spacing']
                xvals[path]['x'] = x
                xpos.append(x)
                xpaths.append(path)
            x += self._styler_settings['boxplot_spacing']
            xs.append(x)
            xticks.append(np.mean(xs))
            xlabels.append(group)
            
        x_axis = {
            'position': xpos,
            'ticks' : xticks,
            'labels': xlabels,
            'paths': xpaths
        }
        return xvals, x_axis
    
    def get_x_styling(self, xvals:dict) -> Tuple[dict, dict]:
        if self._styler_settings['xstyle']['type'] == 'groups':
            return self._x_groupings(xvals)
            
    def add_legend(self, plot_styling:dict, p):
        for label in plot_styling['labels_colours_alpha']:
            p.circle(x=0, y=self._styler_settings['ystyle']['range'][0] - 10, radius=0.5, color=plot_styling['labels_colours_alpha'][label]['colour'], alpha=plot_styling['labels_colours_alpha'][label]['alpha'], legend_label=label)
            
                
            
            
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    