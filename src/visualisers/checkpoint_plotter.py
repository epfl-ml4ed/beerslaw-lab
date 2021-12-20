import os
import pickle
import yaml

import numpy as np
import pandas as pd

import re
from shutil import copyfile, copytree

from extractors.pipeline_maker import PipelineMaker
from bokeh.plotting import figure, output_file, show, save
from ml.xval_maker import XValMaker
from ml.models.classifiers.lstm import LSTMModel
from ml.samplers.sampler import Sampler

from visualisers.stylers.full_sequences_styler import FullStyler
import tensorflow as tf



class CheckpointPlotter:
    """This class recreates the test performances on the best checkpoint
    """

    def __init__(self, settings):
        self._name = 'checkpoint plotter'
        self._notation = 'ckptpltr'

        self._settings = settings
        self._load_palette()
        self._styler = FullStyler(settings)

    def _load_palette(self):
        self._palette = [
            '#b7094c', '#a01a58', '#892b64', '#723c70', '#5c4d7d', '#455e89', '#2e6f95', '#1780a1', '#0091ad',
        ]
    
    def _get_colour(self):
        return np.random.choice(self._palette)

    def _crawl_modelcheckpoints(self):
        """Checks the paths where a model checkpoint is found

        Returns:
            [dict]: For each separate experiment folder:
                        for each model:
                            fold:paths with the checkpoints in the experiment
        """

        # get all the paths
        model_paths = []
        experiment_path = '../experiments/' + self._settings['experiment']['name']
        for (dirpath, dirnames, filenames) in os.walk(experiment_path):
            files = [os.path.join(dirpath, file) for file in filenames]
            model_paths.extend(files)
        kw = 'model_checkpoint'
        model_paths = [path for path in model_paths if 'exclude' not in path]
        model_paths = ['/'.join(path.split('/')[:-1]) for path in model_paths]
        model_paths = [path for path in model_paths if path.endswith(kw)]
        model_paths = list(set(model_paths))

        # Sort per experiment
        date_re = re.compile('(.*202[0-9]_[0-9]+_[0-9]+_[0-9]+/)')
        model_re = re.compile('.*logger/(.*)/')
        fold_re = re.compile('.*f([\-0-9]+)')
        paths = {}
        for path in model_paths:
            experiment = date_re.findall(path)[0]
            model = model_re.findall(path)[0]
            fold = int(fold_re.findall(path)[0]) + 1
            if experiment not in paths:
                paths[experiment] = {}

            if model not in paths[experiment]:
                paths[experiment][model] = {}

            paths[experiment][model][fold] = path

        return paths

    def _get_trainset(self, sequences:list, labels:list, indices:list, id_dictionary:dict, sampler:Sampler):
        """Returns the train and test ids, making the assumption that we are using the flatstrat strategy for the outer fold

        Args:
            sequences (list): sequences of data as returned by the pipeline maker
            labels (list): labels of data as returned by the pipeline maker
            indices (list): list of indices from the id_dictoinary corresponding to the sequences and labels
            id_dictionary (dict): id dictionary as described in pipeline maker
            sampler (Sampler): there to over/under/not sample the data

        Returns:
            [list]: x_train
            [list]: y_train
        """
        with open(self._settings['model_checkpoint']['train_ids'], 'rb') as fp:
            train_ids = pickle.load(fp)

        train_ids = [idx for idx in train_ids if idx in id_dictionary['index']]
        train_ids = [id_dictionary['index'][username] for username in train_ids]
        train_ids = [int(indices.index(idx)) for idx in train_ids]

        x_train = [sequences[tid] for tid in train_ids]
        y_train = [labels[tid] for tid in train_ids]

        x_train, y_train = sampler.sample(x_train, y_train)

        return x_train, y_train

    def _retrieve_model(self, fold_path:str, sequences:list, config:dict, xval: XValMaker):
        """Loads the weights achieving the best validation scores into a model object (designed from our pipeline).

        Args:
            experiment_path (str): path to the experiment (up to the date tag)
            model_name (str): name of the model (where for each fold, the train-validation and checkpoints are saved)
            fold_path (str): name of the particular checkpoint
            config (dict): [description]
            xval (XValMaker): xval object

        Returns:
            [type]: model object
        """
        path = '/'.join(fold_path.split('/')[:-1])
        architecture_path = path + '/architecture.pkl'
        with open(architecture_path, 'rb') as fp:
            architecture = pickle.load(fp)
         
        checkpoint_path = '../experiments/temp_checkpoints/plotter/'
        copytree(fold_path, checkpoint_path, dirs_exist_ok=True)
        settings = dict(config)
        settings['ML']['models']['classifiers'][settings['ML']['pipeline']['model']] = architecture
        model = xval._model(settings)
        model.set_outer_fold(0)
        model.load_checkpoints(checkpoint_path, sequences)
        return model

    def _get_architecture(self, architecture_path:str) -> dict:
        """Retrieves the architecture of a model in a dictionary form based on the paths

        Args:
            experiment_path ([type]): path of the experiment (up to the date tag)
            model_name ([type]): name of the model folder where the train-validation logs are saved

        Returns:
            dict: architecture of the model
        """
        path = '/'.join(architecture_path.split('/')[:-1])
        path = path + '/architecture.pkl'
        with open(path, 'rb') as fp:
            architecture = pickle.load(fp)
        return architecture

    def _compute_test_scores(self, sequences:list, labels:list, indices:list, scoring_croissant:bool, id_dictionary:dict, best_models:dict):
        with open(self._settings['model_checkpoint']['test_ids'], 'rb') as fp:
            test_ids = pickle.load(fp)

        test_ids = [idx for idx in test_ids if idx in id_dictionary['index']]
        test_ids = [id_dictionary['index'][username] for username in test_ids]
        test_ids = [int(indices.index(idx)) for idx in test_ids]

        x_test = [sequences[tid] for tid in test_ids]
        y_test = [labels[tid] for tid in test_ids]

        scores_df = pd.DataFrame()
        scores_df['scores'] = [best_models['folds'][f]['scores'] for f in best_models['folds']]
        scores_df['model'] = [best_models['folds'][f]['model'] for f in best_models['folds']]
        scores_df['architecture'] = [best_models['folds'][f]['architecture'] for f in best_models['folds']]
        scores_df['folds'] = [f for f in best_models['folds']]
        scores_df = scores_df.sort_values(['scores'], ascending=not scoring_croissant)

        model_name = scores_df.iloc[0]['model']
        scores = scores_df.iloc[0]['scores']
        architecture = scores_df.iloc[0]['architecture']
        fold = scores_df.iloc[0]['folds']

        best_model = {
            'model_name': model_name,
            'validation_scores': scores,
            'architecture': architecture,
            'fold': fold
        }
        return x_test, y_test, best_model

    def _recreate_folds(self, experiment_path:str, experiment_info:dict) -> dict:
        """Goes through all the folds of an experiment (date tag), loads the weights and makes the predictions on the best
        checkpoint

        Args:
            experiment_path (str): path of the experiment (up untill the date tag)
            experiment_info (dict): dictonary of the experiment containing the model names, and then the paths to each of the
            checkpoints per fold

        Returns:
            dict: dictoinary with the scores and best parameters for each fold on the validation set
        """
        with open(experiment_path + 'config.yaml', 'rb') as fp:
            config = pickle.load(fp)
        #     top_name = config['experiment']['root_name'].split('/')[0] #used when experiments folder are shuffled around
        #     settings_name = self._settings['experiment']['name'].split('/')
        #     config['experiment']['root_name'] = config['experiment']['root_name'].replace(top_name, self._settings['experiment']['name'])
        #     print(experiment_path)
        #     config['experiment']['name'] = experiment_path.split('/')[-2]
        #     print('HELLO', config['experiment']['name'])
        # print('EXPERIMENT', config['experiment'])
        pipeline = PipelineMaker(config)
        sequences, labels, indices, id_dictionary = pipeline.build_data()

        xval = XValMaker(config)
        config['ML']['splitters']['n_folds'] = config['ML']['xvalidators']['nested_xval']['inner_n_folds']
        splitter = xval.get_gridsearch_splitter()(config)
        sampler = xval.get_sampler()()
        scorer = xval.get_scorer()(config)
        scorer.set_optimiser_function(config['ML']['xvalidators']['nested_xval']['optim_scoring'])
        scoring_function = scorer.get_optim_function()
        scoring_croissant = scorer.get_optim_croissant()
        x_train, y_train = self._get_trainset(sequences, labels, indices, id_dictionary, sampler)

        best_models = {
            'experiment_path': experiment_path,
            'experiment_info': experiment_info,
            'folds': {}
        }
        for f, (train_index, validation_index) in enumerate(splitter.split(x_train, y_train)):
            x_val = [x_train[xx] for xx in validation_index]
            y_val = [y_train[yy] for yy in validation_index]

            scores = []
            names = []
            for model_name in experiment_info:
                fold_path = experiment_info[model_name][f]
                model = self._retrieve_model(fold_path, sequences, config, xval)

                y_predict = model.predict(x_val)
                y_proba = model.predict_proba(x_val)
                score = scoring_function(y_val, y_predict, y_proba)
                names.append(model_name)
                scores.append(score)

            score_df = pd.DataFrame()
            score_df['model_name'] = names
            score_df['scores'] = scores
            score_df = score_df.sort_values(['scores'], ascending= not scoring_croissant)
            best_models['folds'][f] = {
                'scores': score_df.iloc[0]['scores'],
                'model': score_df.iloc[0]['model_name'],
                'architecture': self._get_architecture(experiment_info[score_df.iloc[0]['model_name']][f])
            }

        x_test, y_test, best_validation_model = self._compute_test_scores(sequences, labels, indices, scoring_croissant, id_dictionary, best_models)
        validation_model_name = best_validation_model['model_name']
        validation_fold = best_validation_model['fold']
        fold_path = experiment_info[validation_model_name][validation_fold]
        model = self._retrieve_model(fold_path, sequences, config, xval)
        test_predict = model.predict(x_test)
        test_proba = model.predict_proba(x_test)
        score = scoring_function(y_test, test_predict, test_proba)
        
        best_models['best_validation_model'] = best_validation_model
        best_models['test_predict'] = test_predict
        best_models['test_proba'] = test_proba
        best_models['test_score'] = score        

        return best_models

    def _individual_boxplot_df(self, best_models):
        dots = {}
        params = []
        for fold in best_models['folds']:
            dots[fold] = {}
            dots[fold]['data'] = best_models['folds'][fold]['scores']
            for parameter in best_models['folds'][fold]['architecture']:
                dots[fold][parameter] = str(best_models['folds'][fold]['architecture'][parameter])
                params.append(parameter.replace('_', ' '))
            dots[fold]['fold'] = fold
            
        dots_df = pd.DataFrame(dots).transpose()
            
        scores_df = pd.DataFrame()
        scores_df['scores'] = [best_models['folds'][f]['scores'] for f in best_models['folds']]
        scores_df['model'] = [best_models['folds'][f]['model'] for f in best_models['folds']]
        scores_df['architecture'] = [best_models['folds'][f]['architecture'] for f in best_models['folds']]
        scores_df['folds'] = [f for f in best_models['folds']]

        q1 = float(scores_df['scores'].quantile(q=0.25))
        q2 = float(scores_df['scores'].quantile(q=0.5))
        q3 = float(scores_df['scores'].quantile(q=0.75))
        mean = float(scores_df['scores'].mean())
        std = float(scores_df['scores'].std())

        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr
        
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
        return dots_df, boxplot, list(set(params))

    def _plot_individual_errorplot(self, best_models, name, x, glyphs, plot_styling, p):
        styler = {
            'colour': self._get_colour(),
            'label': name,
            'alpha': 0.9
        }
        plot_styling['colour'].append(styler['colour'])
        plot_styling['label'].append(styler['label'])
        plot_styling['alpha'].append(styler['alpha'])
        plot_styling['labels_colours_alpha'].append({'colour': styler['colour'], 'alpha':styler['alpha']})
        plot_styling['linedashes'].append('dotted')
        dots_df, boxplot_df, params = self._individual_boxplot_df(best_models)
        glyphs, p = self._styler.get_individual_plot(dots_df, params, boxplot_df, glyphs, x, styler, p)
        return plot_styling, glyphs, p

    def _plot_multiple_boxplots(self):
        glyphs = {
            'datapoints': {},
            'upper_moustache': {},
            'lower_moustache': {},
            'upper_rect': {},
            'lower_rect': {}
        }
        x_axis = {
            'position': [],
            'ticks' : [],
            'labels': [],
            'paths': []
        }
        xs = []
        models = []
        paths = self._crawl_modelcheckpoints()
        for i, experiment in enumerate(paths):
            best_models = self._recreate_folds(experiment, paths[experiment])
            x_axis['position'].append(i*2)
            x_axis['ticks'].append(i*2)
            model_name = list(paths[experiment].keys())[0] # Imply one model per cross validation
            x_axis['labels'].append(model_name),
            models.append(best_models)
            # print(paths[experiment])
            break
        
        plot_styling = {
            'colour':[],
            'label': [],
            'alpha': [],
            'labels_colours_alpha': [],
            'linedashes': []
        }
        p = self._styler.init_figure(x_axis)
        for i, model in enumerate(models):
            plot_styling, glyphs, p = self._plot_individual_errorplot(model, 'tbf', i*2, glyphs, plot_styling, p)

        plot_styling['labels_colours_alpha'] = {
            'incremental': {
                'colour': 'red',
                'alpha': 0.9
            }
        }
        self._styler.add_legend(plot_styling, p)
        if self._settings['show']:
            show(p)


    def plot(self):
        tf.get_logger().setLevel('ERROR')
        # print('Testing the functions')
        # paths = self._crawl_modelcheckpoints()
        # first_key = list(paths.keys())[0]
        # best_models = self._recreate_folds(first_key, paths[first_key])
        # print('*'*50)
        # print('BEST MODELS')
        # print(best_models)

        self._plot_multiple_boxplots()
    
    def test(self):
        tf.get_logger().setLevel('ERROR')
        print('Testing the functions')
        paths = self._crawl_modelcheckpoints()
        for key in paths:
            best_models = self._recreate_folds(key, paths[key])
            print(best_models['best_validation_model'])
            print(best_models['test_proba'])

            print('*'*50)
            print()

        # self._plot_multiple_boxplots()


        






















































            

            

            

            



    