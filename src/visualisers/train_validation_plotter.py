import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

class TrainValidationPlotter:
    """Plots the training and validation means and stds of different performances
    across folds for all epochs
    """
    def __init__(self, settings:dict):
        self._name = 'train validation plotter'
        self._notation = 'tvpltr'

        self._settings = settings

    def _crawl(self):
        paths= []
        experiment_path = '../experiments/' + self._settings['experiment']['name'] + '/'
        for (dirpath, dirnames, filenames) in os.walk(experiment_path):
            files = [os.path.join(dirpath, file) for file in filenames]
            paths.extend(files)
        kw = 'model_training.csv'
        paths = [path for path in paths if kw in path]
        paths = [xval for xval in paths if 'exclude' not in xval]

        loggers_paths = {}
        for path in paths:
            splitted = path.split('/')
            key = splitted[-2]
            if key not in loggers_paths:
                loggers_paths[key] = []

            loggers_paths[key].append(path)

        return loggers_paths

    def _plot_shaded_folds(self, pathname, files, metric):
        files.sort()
        
        metrics = []
        val_metrics = []
        epochs = []
        
        plt.figure(figsize=(12, 8))
        for file in files:
            model = pd.read_csv(file, sep=';')
            metrics.append(list(model[metric]))
            val_metrics.append(list(model['val_' + metric]))
            epochs.append(model['epoch'])
            
        if self._settings['partial']:
            maximum = np.max([len(metr) for metr in metrics])
            metrics = [metri for metri in metrics if len(metri) == maximum]
            epochs = [epoc for epoc in epochs if len(epoc) == maximum]
            means = np.mean(metrics, axis=0)
            stds = np.std(metrics, axis= 0)
        else:
            minimums = np.min([(len(metr)) for metr in metrics])
            metrics = [metri[:minimums] for metri in metrics]
            means = np.mean(metrics, axis=0)
            stds = np.std(metrics, axis=0)
        
        min_plot = min(means-stds)
        max_plot = max(means+stds)
        plt.plot(epochs[0], means, color='#abc4ff')
        plt.fill_between(epochs[0], means - stds, means + stds, alpha=0.3, color='#abc4ff', label='train')
        if self._settings['partial']:
            maximum = np.max([len(metr) for metr in val_metrics])
            val_metrics = [metri for metri in val_metrics if len(metri) == maximum]
            epochs = [epoc for epoc in epochs if len(epoc) == maximum]
            means = np.mean(val_metrics, axis=0)
            stds = np.std(val_metrics, axis= 0)
        else:
            minimums = np.min([(len(metr)) for metr in val_metrics])
            val_metrics = [metri[:minimums] for metri in val_metrics]
            means = np.mean(val_metrics, axis=0)
            stds = np.std(val_metrics, axis=0)
        
        min_plot = min(min_plot, min(means-stds))
        max_plot = max(max_plot, max(means+stds))
        plt.plot(epochs[0], means, color='#ff5c8a')
        plt.fill_between(epochs[0], means - stds, means + stds, alpha=0.3, color='#ff5c8a', label='validation')

        plt.ylim([min_plot, max_plot])
        plt.legend()
        plt.title(pathname)
        
        if self._settings['save']:
            path = files[0].split('/')[:-1]
            path = '/'.join(path)
            path += '/train_validation_' + metric + 'epochsplot.svg'
            plt.savefig(path, format='svg')
        if self._settings['show']:
            plt.show()

    def plot(self, metric):
        paths = self._crawl()
        for experiment in paths:
            self._plot_shaded_folds(experiment, paths[experiment], metric)
            




    
        

    

    