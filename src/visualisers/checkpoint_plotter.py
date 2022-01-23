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
from bokeh.io import export_svg, export_png
from matplotlib import pyplot as plt


from visualisers.stylers.full_sequences_styler import FullStyler
import tensorflow as tf

class CheckpointPlotter:
    """This class recreates the test performances on the best checkpoint for xval without gridsearch
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

        # print(paths.keys())
        # exit(1)

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
            fold_path (str): name of the particular checkpoint
            config (dict): [description]
            xval (XValMaker): xval object

        Returns:
            [type]: model object
        """
        path = '/'.join(fold_path.split('/')[:-1])
        architecture_path = path + '/architecture.pkl'
        # print('    model')
        architecture = self._get_architecture(architecture_path)
         
        checkpoint_path = '../experiments/temp_checkpoints/plotter/'
        copytree(fold_path, checkpoint_path, dirs_exist_ok=True)
        settings = dict(config)
        settings['ML']['models']['classifiers'][settings['ML']['pipeline']['model']] = architecture
        model = xval._model(settings)
        model.set_outer_fold(0)
        model.load_checkpoints(checkpoint_path, sequences)
        return model

    def _extract_features(self, model_name: str):
        """
        Retrieves the architecture details from the name path.
        
        Args
            model_name: path of the folder 
            
        Returns
            dictionary with the parameters
        """
        # cell types
        print('*'*100)
        print(model_name)
        re_ct = re.compile('/ct([A-z]*)_')
        ct = re_ct.findall(model_name)[0]

        # nlayers
        re_nlayers = re.compile('[A-z]_nlayers([0-9]+)_')
        nlayers = re_nlayers.findall(model_name)[0]

        # ncells
        re_ncells = re.compile('.*ncells\[([0-9,\s]+)\]')
        ncells = re_ncells.findall(model_name)[0]
        ncells = ncells.split(', ')
        ncells = [int(cell) for cell in ncells]

        # dropout
        re_dropout = re.compile('.*drop([0-9\.]+)')
        dropout = re_dropout.findall(model_name)[0]
        dropout = dropout[0] + '.' + dropout[1:]

        # optimiser
        re_optimi = re.compile('.*optim([A-z]+)_loss')
        optimi = re_optimi.findall(model_name)[0]

        # batch size
        re_bs = re.compile('.*bs([0-9]+)_')
        bs = re_bs.findall(model_name)[0]

        # epochs
        re_epochs = re.compile('.*ep([0-9]+)lstm')
        epochs = re_epochs.findall(model_name)[0]

        architecture = {
            'cell_type': ct,
            'n_layers': int(nlayers),
            'n_cells': ncells,
            'dropout': float(dropout),
            'optimiser': optimi,
            'batch_size': int(bs),
            'epochs': int(epochs),
            'padding_value': -1,
            'loss': 'auc',
            'shuffle':True,
            'verbose': 1,
            'early_stopping': False
        }
        return architecture

    def _extract_modellabel(self, model_name: str):
        """
        Retrieves the architecture details from the name path.
        
        Args
            model_name: path of the folder 
            
        Returns
            dictionary with the parameters
        """
        # cell types
        print('*'*100)
        print(model_name)
        re_ct = re.compile('/ct([A-z]*)_')
        ct = re_ct.findall(model_name)[0]

        # nlayers
        re_nlayers = re.compile('[A-z]_nlayers([0-9]+)_')
        nlayers = re_nlayers.findall(model_name)[0]

        # ncells
        re_ncells = re.compile('.*ncells\[([0-9,\s]+)\]')
        ncells = re_ncells.findall(model_name)[0]
        ncells = ncells.split(', ')
        ncells = [int(cell) for cell in ncells]

        # dropout
        re_dropout = re.compile('.*drop([0-9\.]+)')
        dropout = re_dropout.findall(model_name)[0]
        dropout = dropout[0] + '.' + dropout[1:]

        # optimiser
        re_optimi = re.compile('.*optim([A-z]+)_loss')
        optimi = re_optimi.findall(model_name)[0]

        # batch size
        re_bs = re.compile('.*bs([0-9]+)_')
        bs = re_bs.findall(model_name)[0]

        # epochs
        re_epochs = re.compile('.*ep([0-9]+)lstm')
        epochs = re_epochs.findall(model_name)[0]

        if 'secondslstm' in model_name:
            re_sequencer = re.compile('/([A-z]*)_secondslstm')
            sequencer = re_sequencer.findall(model_name)[0]
        elif 'flat' in model_name:
            re_sequencer = re.compile('/([A-z]*)_secondslstm')
            sequencer = re_sequencer.findall(model_name)[0]
        return sequencer

    def _dump_architecture(self, model_path:str):
        """
        Reads the path, retrieves the architecture, and dumps the file there
        """
        
        model_name = model_path.split('/')[-1]
        architectures = self._extract_features(model_name)
        with open(model_path + '/architecture.pkl', 'wb') as fp:
            pickle.dump(architectures, fp)
    
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
        try:
            with open(path, 'rb') as fp:
                architecture = pickle.load(fp)
        except FileNotFoundError:
            architecture = self._extract_features(path)
            with open(path, 'wb') as fp:
                pickle.dump(architecture, fp)
            return architecture
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

        Assumption:
            Only one outer fold
            No resampling

        Args:
            experiment_path (str): path of the experiment (up untill the date tag)
            experiment_info (dict): dictonary of the experiment containing the model names, and then the paths to each of the
            checkpoints per fold

        Returns:
            dict: dictoinary with the scores and best parameters for each fold on the validation set
        """

        if not self._settings['nocache'] and 'best_models.pkl' in os.listdir(experiment_path):
            with open(experiment_path + '/best_models.pkl', 'rb') as fp:
                best_models = pickle.load(fp)
        else:
            with open(experiment_path + 'config.yaml', 'rb') as fp:
                config = pickle.load(fp)

            # gridsearch object for folds
            gs_file = os.listdir(experiment_path + '/gridsearch results/')[0]
            with open(experiment_path + '/gridsearch results/' + gs_file, 'rb') as fp:
                gs = pickle.load(fp)
            gs_results = gs.get_results()

            # xval object for trainset
            xval_path = os.listdir(experiment_path + '/results/')[0]
            with open(experiment_path + '/results/' + xval_path, 'rb') as fp:
                xval_object = pickle.load(fp)
            oversampled_indices = xval_object[0]['oversample_indices']
            test_indices = xval_object[0]['test_indices']

            # data
            pipeline = PipelineMaker(config)
            sequences, labels, indices, id_dictionary = pipeline.build_data()

            # scoring function
            xval = XValMaker(config)
            sampler = xval.get_sampler()()
            scorer = xval.get_scorer()(config)
            scorer.set_optimiser_function(config['ML']['xvalidators']['nested_xval']['optim_scoring'])
            scoring_function = scorer.get_optim_function()
            scoring_croissant = scorer.get_optim_croissant()

            # data
            train_idx = [xval_object['indices'].index(idx) for idx in oversampled_indices]
            x_train = [sequences[idx] for idx in train_idx]
            y_train = [labels[idx] for idx in train_idx]


            best_models = {
                'experiment_path': experiment_path,
                'experiment_info': experiment_info,
                'id_dictionary': config['id_dictionary'],
                'label_map': pipeline.get_label_map(),
                'test_indices': test_indices,
                'folds': {}
            }
            for fold in gs_results[0]['fold_index']:
                train_index = gs_results[0]['fold_index'][fold]['train']
                validation_index = gs_results[0]['fold_index'][fold]['validation']

                xx_train = [x_train[idx] for idx in train_index]
                yy_train = [y_train[idx] for idx in train_index]
                x_val = [x_train[idx] for idx in validation_index]
                y_val = [y_train[idx] for idx in validation_index]
                val_indices = [oversampled_indices[idx] for idx in validation_index]

                scores = []
                names = []
                probas = {}
                for model_name in experiment_info:
                    fold_path = experiment_info[model_name][fold]
                    model = self._retrieve_model(fold_path, sequences, config, xval)

                    y_predict = model.predict(x_val)
                    y_proba = model.predict_proba(x_val)
                    score = scoring_function(y_val, y_predict, y_proba)
                    names.append(model_name)
                    scores.append(score)
                    probas[model_name] = y_proba

                score_df = pd.DataFrame()
                score_df['model_name'] = names
                score_df['scores'] = scores
                score_df = score_df.sort_values(['scores'], ascending= not scoring_croissant)
                best_models['folds'][fold] = {
                    'scores': score_df.iloc[0]['scores'],
                    'model': score_df.iloc[0]['model_name'],
                    'architecture': self._get_architecture(experiment_info[score_df.iloc[0]['model_name']][fold]),
                    'probabilities': probas,
                    'val_indices': val_indices,
                    'y_val': y_val,
                    'y_predict': y_predict,
                    'y_proba': y_proba 
                }

            best_models['validation_summaries'] = {
                'mean': np.mean([best_models['folds'][fold]['scores'] for fold in best_models['folds']]),
                'std': np.std([best_models['folds'][fold]['scores'] for fold in best_models['folds']]),
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
            with open(experiment_path + '/best_models.pkl', 'wb') as fp:
                pickle.dump(best_models, fp) 

        return best_models


    def _recreate_foldstest(self, experiment_path:str, experiment_info:dict) -> dict:
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

################################
## Create boxplots for validation scores
##

    def _individual_boxplot_df(self, best_models):
        dots = {}
        params = []
        for fold in best_models['folds']:
            dots[fold] = {}
            dots[fold]['data'] = best_models['folds'][fold]['scores']
            for parameter in best_models['folds'][fold]['architecture']:
                dots[fold][parameter.replace('_', ' ')] = str(best_models['folds'][fold]['architecture'][parameter])
                params.append(parameter.replace('_', ' ')) 
            dots[fold]['fold'] = fold
            
        dots_df = pd.DataFrame(dots).transpose()
        params = list(set(params))
            
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
        return dots_df, boxplot, dots_df[params]

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
        # params = params.iloc[0]
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

        experiment_best_models = {}
        model_names = []
        simple_mns = []
        for i, experiment in enumerate(paths):
            best_models = self._recreate_folds(experiment, paths[experiment])
            # x_axis['position'].append(i*2)
            # x_axis['ticks'].append(i*2)

            # model_name = 'model ' + str(i) # Imply one model per cross validation
            # x_axis['labels'].append(model_name),
            experiment_best_models[experiment] = best_models
            models.append(best_models)
            # print(paths[experiment])
            model_names.append(experiment + best_models['folds'][0]['model'])
            simple_mns.append(best_models['folds'][0]['model'])

        x_paths = [mn for mn in model_names]

        x_order = []
        if len(self._settings['plot_style']['xstyle']['groups']) > 0:
            indices = []
            for group in self._settings['plot_style']['xstyle']['groups']:
                indices = indices + [idx for idx in range(len(x_paths)) if group in x_paths[idx]]
                x_order = x_order + indices

            x_axis['position'] = [i*2 for i in range(len(x_order))]
            x_axis['ticks'] = [i*2 for i in range(len(x_order))]
            x_axis['labels'] = [model_names[idx] for idx in indices]

        x_axis['labels'] = [self._extract_modellabel(l) for l in x_axis['labels']]


        plot_styling = {
            'colour':[],
            'label': [],
            'alpha': [],
            'labels_colours_alpha': [],
            'linedashes': []
        }
        p = self._styler.init_figure(x_axis)
        for i, model in enumerate([models[idx] for idx in indices]):
            plot_styling, glyphs, p = self._plot_individual_errorplot(model, 'tbf', i*2, glyphs, plot_styling, p)

        plot_styling['labels_colours_alpha'] = {
            'incremental': {
                'colour': 'red',
                'alpha': 0.9
            }
        }
        self._styler.add_legend(plot_styling, p)

        if self._settings['saveimg']:
            p.output_backend = 'svg'
            path = '../experiments/' + self._settings['experiment']['name'] + '/checkpoint_validation_folds.svg'
            export_svg(p, filename=path)
        if self._settings['save']:
            path = '../experiments/' + self._settings['experiment']['name'] + '/checkpoint_validation_folds.html'
            save(p, filename=path)
        if self._settings['show']:
            show(p)

################################
## Create prediction distribution on validation sets.
##
    def _get_label_map(self):
        label_map = self._settings['model_checkpoint']['label_map']
        if self._settings['model_checkpoint']['label_map'] == 'vector_labels':
            class_map = '../data/experiment_keys/permutation_maps/vector_binary.yaml'
            n_classes = 8
                        
        with open(class_map) as fp:
            label_map = yaml.load(fp, Loader=yaml.FullLoader)
        return label_map

    def _load_indices_label(self, id_dictionary:dict):
        indices_labels = {}
        for idx in id_dictionary['sequences']:
            with open(id_dictionary['sequences'][idx]['path'], 'rb') as fp:
                sim = pickle.load(fp)
                indices_labels[idx] = sim['permutation']
        return indices_labels

    def _predictionprobabilities_per_class(self, best_models: dict):
        label_map = self._get_label_map()
        labels_probs = {}
        
        id_dictionary = best_models['id_dictionary']
        indices_labels = self._load_indices_label(id_dictionary)

        for fold in best_models['folds']:
            for model in best_models['folds'][fold]['probabilities']:
                if model not in labels_probs:
                    labels_probs[model] = {label:[] for label in label_map['labels']}
                for i, val_index in enumerate(best_models['folds'][fold]['val_indices']):
                    plot_label = indices_labels[val_index]
                    plot_label = label_map['map'][plot_label]
                    labels_probs[model][plot_label].append(best_models['folds'][fold]['probabilities'][model][i][1])

        test_indices = best_models['test_indices']
        labels_probs['test'] = {label:[] for label in label_map['labels']}
        for i, idx in enumerate(test_indices):
            plot_label = indices_labels[idx]
            plot_label = label_map['map'][plot_label]
            labels_probs['test'][plot_label].append(best_models['test_proba'][i][1])

        return labels_probs
        
    def _predictionprobabilities_plot(self, labels_probs:dict, model:str, label:str, experiment:str):
        """Returns the plot with the probability density for  a specific model, and a specific label

        Args:
            labels_probs (dict): dictionary containing the probabilities per label and per model
            model (str): model name
            label (str): label name
            experiment (str): experiment name up untill the date stamp
        """
        probabilities = labels_probs[model][label]
        test_probabilities = labels_probs['test'][label]
        title = 'probability density for model {} and label {}'.format(model, label)

        plt.figure(figsize=(12, 4))
        plt.hist(probabilities, color='#cbf3f0', alpha=0.6, density=True, label='validation')
        plt.hist(test_probabilities, color='#ffbf69', alpha=0.6, density=True, label='test')
        plt.xlim([0, 1])
        plt.legend()
        plt.title(title)

        if self._settings['save'] or self._settings['saveimg']:
            print('saving experiment!')
            path = experiment + '/predictionprobabilities_densities/'
            os.makedirs(path, exist_ok=True)
            path += 'm{}_label{}.svg'.format(model, label)
            plt.savefig(path, format='svg')
        if self._settings['savepng']:
            print('saving experiment!')
            path = experiment + '/predictionprobabilities_densities/'
            os.makedirs(path, exist_ok=True)
            path += 'm{}_label{}.png'.format(model, label)
            plt.savefig(path, format='png')

        if self._settings['show']:
            plt.show()
        else:
            plt.close()

    def _plot_validation_test_predictionprobabilities(self):
        paths = self._crawl_modelcheckpoints()
        for i, experiment in enumerate(paths):
            best_models = self._recreate_folds(experiment, paths[experiment])
            labels_probs = self._predictionprobabilities_per_class(best_models)
            for model in labels_probs:
                if model != 'test':
                    for label in labels_probs[model]:
                        self._predictionprobabilities_plot(labels_probs, model, label, experiment)


################################
## public functions
##
    def boxplot_validation(self):
        tf.get_logger().setLevel('ERROR')
        self._plot_multiple_boxplots()
    
    def plot_probability_distribution(self):
        tf.get_logger().setLevel('ERROR')
        self._plot_validation_test_predictionprobabilities()

    def get_validation_summaries(self):
        tf.get_logger().setLevel('ERROR')
        paths = self._crawl_modelcheckpoints()

        bm = {}
        for i, experiment in enumerate(paths):
            best_models = self._recreate_folds(experiment, paths[experiment])
            bm[experiment] = {
                'architecture': best_models['folds'][0]['architecture'],
                'val': best_models['validation_summaries']
            }

        for experiment in bm:
            print('*' * 10)
            print(experiment)
            print(bm[experiment]['architecture'])
            print(bm[experiment]['val'])
            print()
            
    def confusion_matrix_plot(self):
        tf.get_logger().setLevel('ERROR')
        paths = self._crawl_modelcheckpoints()
        for i, experiment in enumerate(paths):
            best_models = self._recreate_folds(experiment, paths[experiment])

    def get_bestmodels(self):
        paths = self._crawl_modelcheckpoints()
        for key in paths:
            best_models = self._recreate_folds(key, paths[key])
        return best_models

    def get_predictedprobabilities_printed(self):
        """Look into the predicted probabilities for each instance
        """
        tf.get_logger().setLevel('ERROR')
        print('Testing the functions')
        paths = self._crawl_modelcheckpoints()
        for key in paths:
            best_models = self._recreate_folds(key, paths[key])
        for key in best_models:
            print(best_models['best_validation_model'])
            print(best_models['test_proba'])

            print('*'*50)
            print()


        # self._plot_multiple_boxplots()
        

        






















































            

            

            

            



    