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

from labels.concatenators.concatenator import Concatenator

class LabelPlotter:
    
    def __init__(self, settings, concatenator: Concatenator):
        """Plotting the results of the concatenated labels

        Args:
            settings ([type]): [description]
        """
        self._name = 'label_plotter'
        self._notation = 'lblpltr'
        self._settings = settings
        self._legend_addition = concatenator.get_legend_addition()
        self._order = concatenator.get_order()
        self._combinations = concatenator.get_combinations()
        
        self._load_palettes()
        self._load_id_dictionary()
        self._load_demographics()
        
        
    def _load_palettes(self):
        with open('./visualisers/maps/concat_labels.yaml', 'rb') as fp:
            self._palette = yaml.load(fp, Loader=yaml.FullLoader)[self._settings['concatenation']['concatenator']]
        with open('./visualisers/maps/stratifier.yaml', 'rb') as fp:
            self._stratifier_palette = yaml.load(fp, Loader=yaml.FullLoader)
        self._palette.update(self._stratifier_palette)
        
    def _load_id_dictionary(self):
        """Loads the summaries of sequences and information about the students

        => Implement it in the settings
        """
        raise NotImplementedError
        
    def _load_demographics(self):
        """Returns the demographics 
        """
        index = {v:k for (k, v) in self._id_dictionary['index'].items()}
        with open('../data/beerslaw/post_test/rankings.pkl', 'rb') as fp:
            post_test = pickle.load(fp)
            post_test = post_test.set_index('username')
            
        self._demographics_map = {}
        for iid in self._id_dictionary['sequences']:
            lid = index[iid]
            self._demographics_map[iid] = {
                'gender': post_test.loc[lid]['gender'],
                'year': post_test.loc[lid]['year'],
                'field': post_test.loc[lid]['field'],
                'permutation': post_test.loc[lid]['ranking'],
                'language': post_test.loc[lid]['language']
            }
        
        
    def _get_percentage(self, df, grouping:list, stratification: str):
        """Used to retrieve the percentage of students in each categories, potentially stratified by "stratification"

        Args:
            df ([type]): dataframe with the information
            grouping (list): what to sort the students in
            stratification (str): how to stratify the students

        Returns:
            [type]: dataframe with the percentage of each of the students in each of the groups in grouping, potentially stratified by stratification
        """
        d = df[['year', 'lid', 'gender', 'field', 'no_strat', 'predicted_label', 'ground_truth', 'language']].groupby(grouping).nunique()['lid'].reset_index()
        
        strat = df[[stratification, 'lid']]
        strat = strat.groupby(stratification).nunique()['lid'].reset_index()
        strat.columns = [stratification, 'total_counts']
        
        new = d.merge(strat, how='inner', on=stratification)
        new['height'] = new['lid'] / new['total_counts']
        return new
        
    def _plot_label_distribution(self, new_predictions: dict):
        """Plot the label distribution in absolute numbers, without stratification
        Args:
            new_predictions (dict): 
                key [index from the id_dictionary]: {
                    class 1 [the first class to concatenate with the others]:
                        raw proba: the vector of probability
                        pred: the actual prediction (integer of the label)
                        proba: the probability of being pred
                    ...
                    class n [the nth class to concatenate with the others]:
                        raw proba: the vector of probability
                        pred: the actual prediction (integer of the label)
                        proba: the probability of being pred
                }
        """
        # labels = [new_predictions[iid]['predicted_label'] for iid in new_predictions]
        # sns.countplot(x=labels, order=self._order, palette=self._palette)
        # plt.title('Distribution of student per label' + self._legend_addition)
        # plt.ylabel('#student')
        # plt.xlabel('vector label' + self._legend_addition)
        
        # if self._settings['save']:
        #     path = '../reports/' + self._settings['concatenation']['report_folder'] + '/figures/label_distribution_'
        #     path += 'exp' + self._settings['experiment']['name'] + '_algorithm' + self._settings['experiment']['algorithm']
        #     path += '_feat' + self._settings['experiment']['features'] + '_seq' + self._settings['experiment']['sequencer']
        #     path += '_class' + self._settings['experiment']['classname']
        #     path +='.svg'
        #     plt.savefig(path, format='svg')
        # if self._settings['show']:
        #     plt.show()
        # else:
        #     plt.close()
        raise NotImplementedError
            
    def _plot_label_distribution_stratified(self, new_predictions: dict, stratifier: str):
        """Plot the percentage of students in each of the labels, stratified by the stratified

        Args:
            new_predictions (dict):
                key [index from the id_dictionary]: {
                    class 1 [the first class to concatenate with the others]:
                        raw proba: the vector of probability
                        pred: the actual prediction (integer of the label)
                        proba: the probability of being pred
                    ...
                    class n [the nth class to concatenate with the others]:
                        raw proba: the vector of probability
                        pred: the actual prediction (integer of the label)
                        proba: the probability of being pred
                }
            stratifier (str): what to stratify the students with
        """
        # preds = {}
        # for iid in new_predictions:
        #     preds[iid] = {
        #         'predicted_label': new_predictions[iid]['predicted_label'],
        #         'ground_truth': new_predictions[iid]['ground_truth'],
        #         'gender': self._demographics_map[iid]['gender'],
        #         'year': self._demographics_map[iid]['year'],
        #         'field': self._demographics_map[iid]['field'],
        #     }
            
        # preds_df = pd.DataFrame(preds).transpose().reset_index()
        # preds_df = preds_df.rename(columns={'index': 'lid'})
        # preds_df['no_strat'] = 0
        
        # data = self._get_percentage(preds_df, ['predicted_label', stratifier], stratifier)
        # s = sns.catplot(x='predicted_label', y='height', hue=stratifier, kind='bar', data=data, palette=self._palette, legend=False)
        # s.fig.set_size_inches(15, 7)
        # plt.title('Distribution of student per label' + self._legend_addition + ' stratified with: ' + stratifier)
        # plt.ylabel('%students')
        # plt.legend(loc='upper center')
        # plt.tight_layout() 
        
        # if self._settings['save']:
        #     path = '../reports/' + self._settings['concatenation']['report_folder'] + '/figures/label_distribution_strat' + stratifier
        #     path += 'exp' + self._settings['experiment']['name'] + '_algorithm' + self._settings['experiment']['algorithm']
        #     path += '_feat' + self._settings['experiment']['features'] + '_seq' + self._settings['experiment']['sequencer']
        #     path += '_class' + self._settings['experiment']['classname']
        #     path +='.svg'
        #     plt.savefig(path, format='svg')
        # if self._settings['show']:
        #     plt.show()
        # else:
        #     plt.close()
        raise NotImplementedError
  
  
    def _plot_confusion_matrix(self, new_predictions:dict, stratifier='no_strat'):
        """Plot the confusion matrix for the new labels, possibly stratified by a factor

        Args:
            new_predictions (dict): 
            key [index from the id_dictionary]: {
                    class 1 [the first class to concatenate with the others]:
                        raw proba: the vector of probability
                        pred: the actual prediction (integer of the label)
                        proba: the probability of being pred
                    ...
                    class n [the nth class to concatenate with the others]:
                        raw proba: the vector of probability
                        pred: the actual prediction (integer of the label)
                        proba: the probability of being pred
                }
        """
        cm = {}
        for iid in new_predictions:
            cm[iid] = {
                'preds': new_predictions[iid]['predicted_label'],
                'truth': new_predictions[iid]['ground_truth'],
                'gender': self._demographics_map[iid]['gender'],
                'year': self._demographics_map[iid]['year'],
                'field': self._demographics_map[iid]['field'],
                'no_strat': 'nothing'
        }
            
        cm_df = pd.DataFrame(cm).transpose()
        
        for strat in cm_df[stratifier].unique():
            df = cm_df[cm_df[stratifier] == strat] 
            df = confusion_matrix(df['truth'], df['preds'], labels=self._combinations['labels'])
            df = pd.DataFrame(df, columns=self._combinations['labels'], index=self._combinations['labels'])
            df.set_index(self._combinations['labels'])
            sns.heatmap(df, cmap="YlGnBu")
            plt.title('Confusion Matrix stratified with ' + stratifier + '=' + str(strat))
            plt.ylabel('truth')
            plt.xlabel('prediction')
        
            if self._settings['save']:
                path = '../reports/' + self._settings['concatenation']['report_folder'] + '/figures/confusion_matrix_n' + self._notation 
                path += '_strat' + strat
                path += 'exp' + self._settings['experiment']['name'] + '_algorithm' + self._settings['experiment']['algorithm']
                path += '_feat' + self._settings['experiment']['features'] + '_seq' + self._settings['experiment']['sequencer']
                path += '_class' + self._settings['experiment']['classname']
                path +='.svg'
                plt.savefig(path, format='svg')
            if self._settings['show']:
                plt.show()
            else:
                plt.close()
        raise NotImplementedError
            
    def _measure_comparison(self, results: pd.DataFrame):
        # preds = results.reset_index()
        # for measure in results.columns:
        #     s = sns.catplot(x='index', y=measure, kind='bar', data=preds, palette=self._palette, legend=False)
        #     s.fig.set_size_inches(15, 7)
        #     plt.title('Comparison of performance [' + measure + ']')
        #     plt.ylabel(measure)
        #     plt.xlabel('category')
        #     plt.ylim([0, 1])
        #     # plt.legend(loc='upper center')
        #     plt.tight_layout() 
            
        #     if self._settings['save']:
        #         path = '../reports/' + self._settings['concatenation']['report_folder'] + '/figures/performance_comparison_'
        #         path += 'exp' + self._settings['experiment']['name'] + '_algorithm' + self._settings['experiment']['algorithm']
        #         path += '_feat' + self._settings['experiment']['features'] + '_seq' + self._settings['experiment']['sequencer']
        #         path += '_class' + self._settings['experiment']['classname']
        #         path +='.svg'
        #         plt.savefig(path, format='svg')
        #     if self._settings['show']:
        #         plt.show()
        #     else:
        #         plt.close()
        raise NotImplementedError
        
    def plot(self, new_predictions={}, results=pd.DataFrame()):
        # if self._settings['label_distribution']:
        #     for strat in self._settings['plotter']['stratifiers']:
        #         if strat == 'no_strat':
        #             self._plot_label_distribution(new_predictions)
        #         else:
        #             self._plot_label_distribution_stratified(new_predictions, strat)
        # if self._settings['confusion_matrix']:
        #     for strat in self._settings['plotter']['stratifiers']:
        #         self._plot_confusion_matrix(new_predictions, strat)
        # if self._settings['score']:
        #     self._measure_comparison(results)
        raise NotImplementedError
                
            
        
    