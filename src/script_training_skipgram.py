import os
import re
import pickle
import yaml
import pickle
import argparse
import logging 

import numpy as np
import pandas as pd

from ml.xval_maker import XValMaker
from extractors.pipeline_maker import PipelineMaker

from ml.models.modellers.pairwise_skipgram import PWSkipgram
from utils.config_handler import ConfigHandler
from sklearn.metrics import balanced_accuracy_score

from bokeh.plotting import figure, output_file, show, save
from bokeh.models import ColumnDataSource, HoverTool, Whisker
from bokeh.sampledata.autompg import autompg as df
from bokeh.io import output_file, show
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap
from bokeh.layouts import gridplot

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

def offline_skipgram_bestparameters(settings):
    log_path = '../experiments/' + settings['experiment']['root_name'] + '/' + settings['experiment']['name'] + '/skipgram_training_logs.txt'
    logging.basicConfig(
        filename=log_path,
        level=logging.DEBUG, 
        format='', 
        datefmt=''
    )
    logging.debug('**' * 50)
    logging.debug('- -' * 50)
    logging.debug('**' * 50)

    
    experiment_rootname = settings['experiment']['root_name']
    param_grid = settings['ML']['xvalidators']['unsup_nested_xval']['param_grid']
    for e in param_grid['embeddings']:
        for w in param_grid['window_size']:
            for ep in param_grid['epochs']:
                settings['ML']['models']['modellers']['skipgram']['window_size'] = w
                settings['ML']['models']['modellers']['skipgram']['embeddings'] = e
                settings['ML']['models']['modellers']['skipgram']['epochs'] = ep
                exp_name = experiment_rootname + '/e' + str(e) + '_w' + str(w) + '_ep' + str(ep)
                settings['experiment']['root_name'] = exp_name
                handler = ConfigHandler(settings)
                settings = handler.handle_experiment_name()
    
                data_pipeline = PipelineMaker(settings)
                x, y, indices, id_dictionary = data_pipeline.build_data()
                xval = XValMaker(settings)
                xval.train(x[0], y[0], indices)
    
def train_skipgram(settings):
    data_pipeline = PipelineMaker(settings)
    x, y, indices, id_dictionary = data_pipeline.build_data()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.30, random_state=42)
    sg = PWSkipgram(settings)
    sg.fit(x_train, x_val)
    sg.save()
    
def train_offline_embeddings(settings):
    experiment_rootname = settings['experiment']['root_name']
    param_grid = settings['ML']['xvalidators']['unsup_nested_xval']['param_grid']
    
    
    if settings['sequencer'] != '':
        settings['experiment']['root_name'] += 'pw training ' + settings['sequencer']
        settings['data']['pipeline']['sequencer'] = settings['sequencer']
        if settings['sequencer'] == 'minimise':
            settings['experiment']['n_classes'] = 52
            settings['ML']['models']['modellers']['skipgram']['n_states'] = 52
        if settings['sequencer'] == 'extended':
            settings['experiment']['n_classes'] = 100
            settings['ML']['models']['modellers']['skipgram']['n_states'] = 100
    
    if settings['embeddings'] != '':
        embeddings = settings['embeddings'].split('.')
        param_grid['embeddings'] = [int(e) for e in embeddings]
    
    if settings['windows'] != '':
        windows = settings['windows'].split('.')
        param_grid['window_size'] = [int(w) for w in windows]
        
    for e in param_grid['embeddings']:
        for w in param_grid['window_size']:
            for ep in param_grid['epochs']:
                settings['ML']['models']['modellers']['skipgram']['window_size'] = w
                settings['ML']['models']['modellers']['skipgram']['embeddings'] = e
                settings['ML']['models']['modellers']['skipgram']['epochs'] = ep
                exp_name = experiment_rootname + '/e' + str(e) + '_w' + str(w) + '_ep' + str(ep)
                settings['experiment']['root_name'] = exp_name
                handler = ConfigHandler(settings)
                settings = handler.handle_experiment_name()
                data_pipeline = PipelineMaker(settings)
                x, y, indices, id_dictionary = data_pipeline.build_data()
                # x = x[0:9]
                # y = y[0:9]
                x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.30, random_state=42)
                sg = PWSkipgram(settings)
                sg.fit(x_train, x_val, y_val=y_val)
                # print(balanced_accuracy_score(y_val, sg.predict(x_val)))
                sg.save()
                
def get_embeddings(path: str) -> str:
    em = re.compile('e([0-9]+)')
    return em.findall(path)[0]

def get_windows(path: str) -> str:
    win = re.compile('w([0-9]+)')
    return win.findall(path)[0]

def get_epochs(path: str) -> str:
    epochs = re.compile('ep([0-9]+)')
    return epochs.findall(path)[0]   

def get_colour() -> str:
    random_palette = [
            'maroon', 'orangered', 'saddlebrown', 'goldenrod', 'olive', 'yellowgreen', 'forestgreen', 'teal', 'lightseagreen', 'paleturquoise', 'darkturquoise', 'cadetblue', 'deepskyblue', 'lightskyblue', 'steelblue', 'dodgerblue', 'cornflowerblue', 'royalblue', 'slateblue', 'mediumorchid', 'plum', 'orchid', 'deeppink'
    ]
    return random_palette[np.random.randint(len(random_palette))]

def plot_line_results(settings):
    path = '../experiments/' + settings['experiment']['root_name'] + '/'
    result_files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith(".csv"):
                result_files.append(os.path.join(r, file))
                
    results = {}
    for i, file in enumerate(result_files):
        f = pd.read_csv(file, sep=';')
        results[i] = {
            'embeddings': get_embeddings(file),
            'windows': get_windows(file),
            'epochs': get_epochs(file),
            'accuracy': f['accuracy'],
            'loss': f['loss'],
            'val_accuracy': f['val_accuracy'],
            'val_loss': f['val_loss'],
        }
        
    for i in results:
        result = results[i]
        label = 'emb:' + str(result['embeddings']) + ' w:' + str(result['windows']) + ' ep:' + str(result['epochs'])
        plt.plot(result['val_loss'], label=label, color=get_colour())
        
    plt.show()

def plot_results(settings):
    path = '../experiments/' + settings['experiment']['root_name'] + '/'
    result_files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith(".csv"):
                result_files.append(os.path.join(r, file))
                
    results = {}
    for i, file in enumerate(result_files):
        f = pd.read_csv(file, sep=';')
        results[i] = {
            'embeddings': get_embeddings(file),
            'windows': get_windows(file),
            'epochs': get_epochs(file),
            'accuracy': f.iloc[-1]['accuracy'],
            'loss': f.iloc[-1]['loss'],
            'val_accuracy': f.iloc[-1]['val_accuracy'],
            'val_loss': f.iloc[-1]['val_loss'],
        }
    results_df = pd.DataFrame(results).transpose().reset_index()
    print(results_df)
    
    
    loss = figure(
            title='Parameter Comparison',
            sizing_mode='stretch_both'
    )
    loss.title.text_font_size = '15pt'
    loss.xaxis.axis_label_text_font_size  = '2pt'
    loss.yaxis.axis_label_text_font_size  = '2pt'

    loss.vbar(x='index', top='loss',source=results_df, width=0.9, color=get_colour())
    loss.add_tools(HoverTool(tooltips=[
        ('embeddings', '@embeddings'), 
        ('windows', '@windows'),
        ('epochs', '@epochs')
    ]))
    
    valloss = figure(
            title='Parameter Comparison',
            sizing_mode='stretch_both'
    )
    valloss.title.text_font_size = '15pt'
    valloss.xaxis.axis_label_text_font_size  = '2pt'
    valloss.yaxis.axis_label_text_font_size  = '2pt'

    valloss.vbar(x='index', top='val_loss',source=results_df, width=0.9, color=get_colour())
    valloss.add_tools(HoverTool(tooltips=[
        ('embeddings', '@embeddings'), 
        ('windows', '@windows'),
        ('epochs', '@epochs')
    ]))
    
    
    path = '../experiments/' + settings['experiment']['root_name'] 
    path += '/parameter exploration - loss.html'
    output_file(path, mode='inline')
    save(loss)
    path = '../experiments/' + settings['experiment']['root_name'] 
    path += '/parameter exploration - val loss.html'
    output_file(path, mode='inline')
    save(valloss)
    
    show(gridplot([[loss], [valloss]]))
                
    
    
def main(settings):
    if settings['parameterssearch']:
        offline_skipgram_bestparameters(settings)
    if settings['train']:
        train_skipgram(settings)
    if settings['offline']:
        train_offline_embeddings(settings)
    if settings['plot']:
        plot_line_results(settings)

if __name__ == '__main__':
    with open('./configs/sg_config.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser(description='Test Model Training')
    parser.add_argument('--parameterssearch', dest='parameterssearch', default=False, action='store_true')
    parser.add_argument('--train', dest='train', default=False, action='store_true')
    parser.add_argument('--offline', dest='offline', default=False, action='store_true')
    parser.add_argument('--plot', dest='plot', default=False, action='store_true')
    
    parser.add_argument('--sequencer', dest='sequencer', default='', help='sequencer to use', action='store')
    parser.add_argument('--embeddings', dest='embeddings', default='', help='embeddings: 10, 15, 20, 25', action='store')
    parser.add_argument('--windows', dest='windows', default='', help='windows: 4, 6, 8', action='store')
    
    settings.update(vars(parser.parse_args())) 
    
    cfg_handler = ConfigHandler(settings)
    settings = cfg_handler.handle_settings()
        
    main(settings)