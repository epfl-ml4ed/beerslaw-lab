import os
import re
import yaml
import numpy as np
import pandas as pd
from typing import Tuple

import bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models import ColumnDataSource, Whisker
from bokeh.sampledata.autompg import autompg as df

from visualisers.stylers.styler import Styler

class EarlyPredStyler(Styler):
    """Processes the plot config files to style the boxplot figures from the full sequences
    """
    
    def __init__(self, settings:dict):
        self._settings = dict(settings)
        self._styler_settings = dict(settings['plot_style'])
        self._get_maps()
        
    def init_figure(self, xaxis:dict):
        p = figure(
            title=self._styler_settings['title'],
            sizing_mode=self._styler_settings['sizing_mode'],
            y_range=self._styler_settings['ystyle']['range']
        )
        p.title.text_font_size = '25pt'
        p.xaxis.axis_label_text_font_size  = '15pt'
        p.yaxis.axis_label_text_font_size  = '15pt'
        
        p.xaxis.ticker = xaxis
        p.xaxis.major_label_overrides = dict(zip(xaxis, [str(x) for x in xaxis]))
        p.xaxis.axis_label = self._styler_settings['xstyle']['label']
        p.yaxis.axis_label = self._styler_settings['ystyle']['label']
        return p
        
    def _individual_lineplot(self, data:pd.DataFrame, glyphs:dict, x:float, styler:dict, p:bokeh.models.glyphs):
        
        glyphs['line'][x] = p.line(x='x', y='mean', source=data, alpha=styler['alpha'], color=styler['colour'], line_width=2, line_dash=styler['linedash'])
        
        p.add_tools(HoverTool(renderers=[
            glyphs['line'][x],
        ], tooltips=[
            ('mean score', "@mean"),
            ('std', '@std')
        ], mode='mouse'))
        
        return glyphs, p
    
    def get_individual_plot(self, data:pd.DataFrame, glyphs:dict, x:float, styler:dict, p: bokeh.models.glyphs):
        if self._styler_settings['type'] == 'lineplot':
            return self._individual_lineplot(data, glyphs, x, styler, p)
    
    def add_legend(self, plot_styling:dict, p):
        for label in plot_styling['labels_colours_alpha']:
            p.circle(x=0, y=self._styler_settings['ystyle']['range'][0] - 10, radius=0.5, color=plot_styling['labels_colours_alpha'][label]['colour'], alpha=plot_styling['labels_colours_alpha'][label]['alpha'], legend_label=label)
            
                
            
            
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    