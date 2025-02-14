import os
import re
import pickle
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
            date_re = re.compile('(.*202[0-9]_[0-9]+_[0-9]+_[0-9]+/)')
            experiment = date_re.findall(path)[0]
            if experiment not in loggers_paths:
                loggers_paths[experiment] = []

            loggers_paths[experiment].append(path)

        return loggers_paths

    def _plot_shaded_folds(self, pathname, files, metric):
        files.sort()
        
        metrics = []
        val_metrics = []
        epochs = []
        
        plt.figure(figsize=(12, 8))
        for i, file in enumerate(files):
            print(file)
            model = pd.read_csv(file, sep=';')
            metrics.append(list(model[metric]))
            val_metrics.append(list(model['val_' + metric]))
            epochs.append(model['epoch'])
            # if i == 9:
            #     break
            
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
        plt.plot(epochs[0], means, color='#004648')
        plt.fill_between(epochs[0], means - stds, means + stds, alpha=0.3, color='#004648', label='train')
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
        plt.plot(epochs[0], means, color='#D1AC00')
        plt.fill_between(epochs[0], means - stds, means + stds, alpha=0.3, color='#D1AC00', label='validation')

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
        else:
            plt.close()

    def plot(self, metric):
        paths = self._crawl()
        for experiment in paths:
            print(experiment)
            self._plot_shaded_folds(experiment, paths[experiment], metric)

############

    def _get_results(self):
        paths= []
        experiment_path = '../experiments/' + self._settings['experiment']['name'] + '/'
        for (dirpath, dirnames, filenames) in os.walk(experiment_path):
            files = [os.path.join(dirpath, file) for file in filenames]
            paths.extend(files)
        paths = [path for path in paths if 'supgs' in path]

        results_paths = {}
        l_re = re.compile('_l([0-9]+)_')
        f_re = re.compile('_f([0-9]+)\.pkl')
        date_re = re.compile('(.*202[0-9]_[0-9]+_[0-9]+_[0-9]+/)')
        for path in paths:
            l = l_re.findall(path)[0]
            f = f_re.findall(path)[0]
            date = date_re.findall(path)[0]
            if date not in results_paths:
                results_paths[date] = {} 
            if l not in results_paths[date]:
                results_paths[date][l] = {}

            results_paths[date][l][f] = path
            print(results_paths)
        return results_paths

    def _get_trainsummary(self, results_path:str):
        with open(results_path, 'rb') as fp:
            results = pickle.load(fp)

        parameters = results._parameters

        for res_key in results._results:
            print('- '* 30)
            res = results._results[res_key]
            param_str = ''
            for param in parameters:
                param_str += '{}: {} * '.format(param, res[param])
            
            print('    {}'.format(param_str))
            print('    mean: {}'.format(res['mean_score']))
            print('    str: {}'.format(res['std_score']))

    def print_validation_scores(self):
        results_paths = self._get_results()
        for date in results_paths:
            print('*' * 100)
            print('*' * 100)
            print('{}'.format(date))
            for l in results_paths[date]:
                print('*' * 100)
                print('Validation results for timelines with length {}'.format(l))
                for f in results_paths[date][l]:
                    print('-'*50)
                    print(' outer fold {}'.format(f))
                    self._get_trainsummary(results_paths[date][l][f])

            print()
            print()
            




    
        

    

    