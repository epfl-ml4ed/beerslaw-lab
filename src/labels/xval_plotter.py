import os
import re
from tensorboard import summary
import yaml
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Tuple

from ml.gridsearches.gridsearch import GridSearch
from ml.scorers.binaryclassification_scorer import BinaryClfScorer
from ml.scorers.multiclassification_scorer import MultiClfScorer

import bokeh

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns


from labels.concatenators.concatenator import Concatenator
from visualisers.label_plotter import LabelPlotter

class XvalLabelPlotter(LabelPlotter):
    
    def __init__(self, settings):
        """Plotting the results of the concatenated labels

        Args:
            settings ([type]): [description]
        """
        self._name = 'xvallabel_plotter'
        self._notation = 'xvllblpltr'
        self._settings = settings
        
        self._load_palettes()

    def _crawl_paths(self):
        directory = '../experiments/' + self._settings['experiment']['name'] + '/'
        paths = []
        for (dirpath, dirnames, filenames) in os.walk(directory):
            files = [os.path.join(dirpath, file) for file in filenames]
            paths.extend(files)

        configs = [path for path in paths if 'config.yaml' in path]
        xvals = [path for path in paths if 'xval' in path]


        full_paths = {}        
        date_re = re.compile('(.*202[0-9]_[0-9]+_[0-9]+_[0-9]+/)')
        model_re = re.compile('.*logger/(.*)/')
        fold_re = re.compile('.*f([\-0-9]+)')
        for config_path in configs:
            try:
                experiment_date_path = date_re.findall(config_path)[0]
                exp_xvals = [xval for xval in xvals if experiment_date_path in xval]
                full_paths[experiment_date_path] = {
                    'config': config_path,
                    'xval': exp_xvals[0],
                }
            except IndexError:
                continue

        return full_paths
        
    def _load_palettes(self):
        with open('./visualisers/maps/concat_labels.yaml', 'rb') as fp:
            self._palette = yaml.load(fp, Loader=yaml.FullLoader)[self._settings['concatenation']['concatenator']]
        with open('./visualisers/maps/stratifier.yaml', 'rb') as fp:
            self._stratifier_palette = yaml.load(fp, Loader=yaml.FullLoader)
        self._palette.update(self._stratifier_palette)
        
    def _load_id_dictionary(self, config_path):
        """Loads the summaries of sequences and information about the students

        => Implement it in the settings
        """
        with open(config_path, 'rb') as fp:
            config = pickle.load(fp)
        return config['id_dictionary']

    def _load_label_map(self, class_name:str):
        if class_name == 'colbin':
            with open('../data/experiment_keys/permutation_maps/colour_binary.yaml', 'rb') as fp:
                label_map = yaml.load(fp, Loader=yaml.FullLoader)
        if class_name == 'binconcepts':
            with open('../data/experiment_keys/permutation_maps/nconcepts_binary.yaml', 'rb') as fp:
                label_map = yaml.load(fp, Loader=yaml.FullLoader)
        if class_name == 'vector_labels':
            with open('../data/experiment_keys/permutation_maps/vector_binary.yaml', 'rb') as fp:
                label_map = yaml.load(fp, Loader=yaml.FullLoader)

        return label_map
        
    def _load_demographics(self, id_dictionary):
        """Returns the demographics 
        """
        index = {v:k for (k, v) in id_dictionary['index'].items()}
        with open('../data/post_test/rankings.pkl', 'rb') as fp:
            post_test = pickle.load(fp)
            post_test = post_test.set_index('username')
            
        demographics_map = {}
        for iid in id_dictionary['sequences']:
            lid = index[iid]
            demographics_map[lid] = {
                'gender': post_test.loc[lid]['gender'],
                'year': post_test.loc[lid]['year'],
                'field': post_test.loc[lid]['field'],
                'permutation': post_test.loc[lid]['ranking'],
                'language': post_test.loc[lid]['language']
            }
        return demographics_map

    def _load_experiment_files(self, paths:dict):
        """Returns the id dictionary, the xval file

        Args:
            paths (dict): paths to read it from 
        """
        print(paths['config'])
        with open(paths['config'], 'rb') as fp:
            try:
                config = pickle.load(fp)
            except:
                print('USING TEMPORARY FIX')
                # Issue is that some of the config yamls are not being read (copy error?)
                path = '../experiments/baselines/algorithms/GRU/colourbreak_secondslstm/binconcepts/lstm/raw_full/2022_01_01_0/config.yaml'
                with open(path, 'rb') as fp:
                    config = {'experiment':{}}
                    if 'binconcepts' in paths['config']:
                        config['experiment']['class_name'] = 'binconcepts'
                    elif 'colbin' in paths['config']:
                        config['experiment']['class_name'] = 'colbin'
                    config['id_dictionary'] = pickle.load(fp)['id_dictionary']
            if 'id_dictionary' not in config:
                print('USING TEMPORARY FIX')
                # Issue is that for static features, the id_dictionary was not in the config.
                path = '../experiments/baselines/algorithms/GRU/colourbreak_secondslstm/binconcepts/lstm/raw_full/2022_01_01_0/config.yaml'
                with open(path, 'rb') as fp:
                    config['id_dictionary'] = pickle.load(fp)['id_dictionary']

            id_dictionary = config['id_dictionary']

        with open(paths['xval'], 'rb') as fp:
            nested = pickle.load(fp)

        return config, id_dictionary, nested

    def _load_y_true(self, config:dict, indices:list, id_dictionary:dict, demographics_map: dict):
        """Load y true from the x_indices

        Args:
            xval ([type]): [description]
            config ([type]): [description]
        """
        class_name = config['experiment']['class_name']
        label_map = self._load_label_map(class_name)
        rankings = [id_dictionary['sequences'][ind]['learner_id'] for ind in indices]
        rankings = [demographics_map[ind]['permutation'] for ind in rankings]
        rankings = [label_map['map'][r] for r in rankings]
        rankings = [label_map['target_index'][r] for r in rankings]
        return rankings
        
    def _summary_df(self, config, id_dictionary:dict, nested:dict, demographics_map:dict):
        """Plots the confusion matrix for the predictions

        Args:
            experiment_info (dict): dictionary with:
                config: path to the yaml
                xval: path to the xval pkl file
        Plot:
            confusion
        """

        # Get predictions
        ys_test = []
        ys_pred = []
        ys_proba = []
        ys_indices = []
        folds = []
        for fold in nested:
            if fold != 'x' and fold != 'y' and fold != 'indices' and fold != 'optim_scoring':
                test_indices = nested[fold]['test_indices']
                y_test = self._load_y_true(config, test_indices, id_dictionary, demographics_map)
                y_pred = nested[fold]['y_pred']
                y_proba = nested[fold]['y_proba']

                ys_indices = ys_indices + list(test_indices)
                ys_test = ys_test + list(y_test)
                ys_pred = ys_pred + list(y_pred)
                ys_proba = ys_proba + list(y_proba)
                folds = folds + [fold for _ in y_proba]

        # Get demographics + Get dataframes
        cm = {}
        index = {v:k for (k, v) in id_dictionary['index'].items()}
        for i, ind in enumerate(ys_indices):
            iid = index[ind]
            gender = demographics_map[iid]['gender']
            permutation = demographics_map[iid]['permutation']
            year = demographics_map[iid]['year']
            field = demographics_map[iid]['field']
            language = demographics_map[iid]['language']
            cm[iid] = {
                'preds': ys_pred[i],
                'truth': ys_test[i],
                'probas': ys_proba[i],
                'gender': gender,
                'year': year,
                'field': field,
                'language': language,
                'no_strat': 'nothing',
                'permutation': permutation,
                'fold': folds[i]
            }
        cm_df = pd.DataFrame(cm).transpose()
        return cm_df

###### Confusion Matrix
    def _confusion_matrix(self, summary_df:pd.DataFrame, stratifier:str, config:dict, experiment:str):
        class_name = config['experiment']['class_name']
        label_map = self._load_label_map(class_name)
        labels_idx = (label_map['index_target'].keys())
        labels_idx = [int(l) for l in labels_idx]
        labels = label_map['labels']
        for strat in summary_df[stratifier].unique():
            sum_df = summary_df[summary_df[stratifier] == strat] 
            if len(sum_df) == 0:
                continue
            df = confusion_matrix(list(sum_df['truth']), list(sum_df['preds']), labels=labels_idx)
            df = [[norm/np.sum(row)  if np.sum(row) >0 else 0 for norm in row] for row in df]
            df = pd.DataFrame(df, columns=labels, index=labels)
            df.set_index(labels)
            sns.heatmap(df, cmap="YlGnBu", vmin=0, vmax=1, annot=True)
            plt.title('Confusion Matrix stratified with {} = {} (n={})'.format(stratifier, str(strat), len(sum_df)))
            plt.ylabel('truth')
            plt.xlabel('prediction')

            if self._settings['save']:
                path = experiment + '/confusion_matrices/'
                os.makedirs(path, exist_ok=True)
                path += 'strat{}_{}.svg'.format(stratifier, strat)
                plt.savefig(path, format='svg')
            if self._settings['show']:
                plt.show()
            else:
                plt.close()

        return df

    def _plot_confusion_matrix(self, config:dict, summary_df:dict, experiment:str):
        for strat in self._settings['scorer']['stratifiers']:
            self._confusion_matrix(summary_df, strat, config, experiment)

###### Prediction Probabilities
    def _prediction_vectorlabels_vs_binconcepts(self, summary_df:pd.DataFrame, stratifier:str, config:dict, experiment:str):
        class_name = self._settings['experiment']['classname']
        label_map = self._load_label_map(class_name)
        labels_idx = (label_map['index_target'].keys())
        labels_idx = [int(l) for l in labels_idx]
        labels = label_map['target_index']
        summary_df['vector_binary'] = summary_df['permutation'].apply(lambda x: label_map['map'][x])

        for label in summary_df['vector_binary'].unique():
            title = 'test probability density for label {}'.format(label)
            sum_df = summary_df[summary_df['vector_binary'] == label]
            for strat in summary_df[stratifier].unique():
                strat_df = sum_df[sum_df[stratifier] == strat]
                probabilities = strat_df['probas']
                probabilities = [proba[1] for proba in probabilities]

                plt.figure(figsize=(12, 4))
                plt.hist(probabilities, color='#cbf3f0', alpha=1, density=True, label='test')
                plt.xlim([0, 1])
                plt.legend()
                plt.title(title)

                if self._settings['save'] or self._settings['saveimg']:
                    print('saving experiment!')
                    path = experiment + '/test_predictionprobabilities_densities/{}/'.format(strat)
                    os.makedirs(path, exist_ok=True)
                    path += 'label{}.svg'.format(label)
                    plt.savefig(path, format='svg')
                if self._settings['savepng']:
                    print('saving experiment!')
                    path = experiment + '/test_predictionprobabilities_densities/{}/'.format(strat)
                    os.makedirs(path, exist_ok=True)
                    path += 'label{}.png'.format(label)
                    plt.savefig(path, format='png')

                if self._settings['show']:
                    plt.show()
                else:
                    plt.close()

    def plot_vectorlabels_vs_binconcepts(self, summary_df:pd.DataFrame, config:dict, experiment:str):
        for strat in self._settings['scorer']['stratifiers']:
            self._prediction_vectorlabels_vs_binconcepts(summary_df, strat, config, experiment)

    def _load_scorer(self, summary_df:pd.DataFrame, measure:str):
        n_classes = len(summary_df['preds'].unique())

        parameters = {}
        parameters['experiment'] = {'n_classes': n_classes}
        parameters['ML'] = {
            'scorers': {
                'scoring_metrics': ['roc', 'balanced_accuracy']
            }
        }
        
        if n_classes == 2:
            self._scorer = BinaryClfScorer(parameters)
        else:
            self._scorer = MultiClfScorer(parameters)

        if measure == 'roc':
            self._get_predictions = self._scorer._score_dictionary['roc']
        if measure == 'balanced_accuracy':
            self._get_predictions = self._scorer._score_dictionary['balanced_accuracy']

####### Slicing Analysis
    def _plot_bubbleplot(self, summary_df:pd.DataFrame, x_coordinate:float, stratifier:str):
        fold_performance = []
        for fold in summary_df['fold'].unique():
            fold_df = summary_df[summary_df['fold'] == fold]
            score = self._get_predictions(list(fold_df['truth']), list(fold_df['preds']), list(fold_df['probas']))
            fold_performance.append(score)

        score_mean = np.mean(fold_performance)
        score_std = np.std(fold_performance)
        mean_width = self._settings['barplot_fairness']['x_spacing'] / 3
        mean_quartiles = self._settings['barplot_fairness']['x_spacing'] / 2

        plt.scatter([x_coordinate for _ in fold_performance], fold_performance, color=self._palette[stratifier], alpha=0.4)
        plt.hlines(
            [score_mean + score_std, score_mean, score_mean - score_std],
            [x_coordinate - (mean_quartiles / 2), x_coordinate - (mean_width / 2), x_coordinate - (mean_quartiles / 2)],
            [x_coordinate + (mean_quartiles / 2), x_coordinate + (mean_width / 2), x_coordinate + (mean_quartiles / 2)],
            color=self._palette[stratifier]
        )
        plt.vlines(
            [x_coordinate],
            [score_mean - score_std],
            [score_mean + score_std],
            color=self._palette[stratifier]
        )

    def _prediction_slicing_bubbleplots(self, summary_df:pd.DataFrame, stratifier:str, measure:str, experiment:str):
        self._load_scorer(summary_df, measure)
        plt.figure(figsize=(12, 4))
        xs = [0]
        xlabels = ['all']
        self._plot_bubbleplot(summary_df, xs[-1], stratifier)
        stratifiers = summary_df[stratifier].unique()
        stratifiers.sort()
        for strat in stratifiers:
            xs.append(xs[-1] + self._settings['barplot_fairness']['x_spacing'])
            strat_df = summary_df[summary_df[stratifier] == strat]
            self._plot_bubbleplot(strat_df, xs[-1], stratifier)
            lab = '{} [{}]'.format(strat, len(strat_df))
            xlabels.append(lab)
        plt.xticks(xs, xlabels)
        plt.title('{} for the {} demographic attribute'.format(measure, stratifier))

        if self._settings['save'] or self._settings['saveimg']:
            path = experiment + '/slicing_plots/'
            os.makedirs(path, exist_ok=True)
            path += 'stratifier_{}.svg'.format(stratifier)
            plt.savefig(path, format='svg')
        if self._settings['savepng']:
            print('saving experiment!')
            path = experiment + '/slicing_plots/'
            os.makedirs(path, exist_ok=True)
            path += 'stratifier_{}.png'.format(stratifier)
            plt.savefig(path, format='png')

        if self._settings['show']:
            plt.show()
        else:
            plt.close()

    def _plot_slicing_bubbleplots(self, summary_df:pd.DataFrame, experiment:str):
        for measure in self._settings['barplot_fairness']['measures']:
            for stratifier in self._settings['scorer']['stratifiers']:
                if stratifier != 'no_strat':
                    self._prediction_slicing_bubbleplots(summary_df, stratifier, measure, experiment)

    def get_summary_df(self, experiment):
        paths = self._crawl_paths()
        config, id_dictionary, nested = self._load_experiment_files(paths[experiment])
        demographics = self._load_demographics(id_dictionary)
        summary_df = self._summary_df(config, id_dictionary, nested, demographics)
        return summary_df

####### Colours

    def plot(self):
        paths = self._crawl_paths()

        for experiment in paths:
            print(experiment)
            config, id_dictionary, nested = self._load_experiment_files(paths[experiment])
            demographics = self._load_demographics(id_dictionary)
            summary_df = self._summary_df(config, id_dictionary, nested, demographics)
            if self._settings['confusion_matrix']:
                self._plot_confusion_matrix(config, summary_df, experiment)

            if self._settings['prob_vecbin']:
                self.plot_vectorlabels_vs_binconcepts(summary_df, config, experiment)

            if self._settings['slicingplot']:
                self._plot_slicing_bubbleplots(summary_df, experiment)
        

