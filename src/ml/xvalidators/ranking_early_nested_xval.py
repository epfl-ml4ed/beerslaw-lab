import os
import yaml
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Tuple

from sklearn.model_selection import StratifiedKFold

from extractors.pipeline_maker import PipelineMaker
from ml.samplers.sampler import Sampler
from ml.models.model import Model
from ml.splitters.splitter import Splitter
from ml.xvalidators.xvalidator import XValidator
from ml.scorers.scorer import Scorer
from ml.gridsearches.gridsearch import GridSearch

from utils.config_handler import ConfigHandler

class RankingEarlyNestedXVal(XValidator):
    """Implements nested cross validation: 
            For each fold, get train and test set:
                split the train set into a train and validation set
                perform gridsearch on the chosen model, and choose the best model according to the validation set
                Predict the test set on the best model according to the gridsearch
            => Outer loop computes the performances on the test set
            => Inner loop selects the best model for that fold

    Args:
        XValidator (XValidators): Inherits from the model class
    """
    
    def __init__(self, settings:dict, gridsearch:GridSearch, inner_splitter:Splitter, gridsearch_splitter:Splitter, outer_splitter:Splitter, sampler:Sampler, model:Model, scorer:Scorer):
        super().__init__(settings, inner_splitter, model, scorer)
        self._name = 'early nested cross validator'
        self._notation = 'early_nested_xval'
        
        settings['ML']['splitters']['n_folds'] = settings['ML']['xvalidators']['nested_xval']['inner_n_folds']
        self._gs_splitter = gridsearch_splitter # To create the folds within the gridsearch from the train set 
        settings['ML']['splitters']['n_folds'] = settings['ML']['xvalidators']['nested_xval']['outer_n_folds']
        self._outer_splitter = outer_splitter(settings) # to create the folds between development and test
        
        self._sampler = sampler()
        self._scorer = scorer(settings)
        self._gridsearch = gridsearch
        
        #debug
        self._model = model
        
    def _init_gs(self, fold, oversampled_indices):
        self._scorer.set_optimiser_function(self._xval_settings['nested_xval']['optim_scoring'])
        self._settings['ML']['splitters']['n_folds'] = self._settings['ML']['xvalidators']['nested_xval']['inner_n_folds']
        self._gs = self._gridsearch(
            model=self._model,
            grid=self._xval_settings['nested_xval']['param_grid'],
            scorer=self._scorer,
            splitter = self._gs_splitter,
            settings=self._settings,
            outer_fold=fold,
            oversampled_indices=oversampled_indices
        )
        
    def _write_predictions(self, test_pred: list, test_proba: list, test_y:list, test_indices: list):
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/results/'
        os.makedirs(path, exist_ok=True)
        
        if 'predictions.pkl' in os.listdir(path):
            with open(path + 'predictions.pkl', 'rb') as fp:
                predictions = pickle.load(fp)
        else:
            predictions = {}
        
        for i, index in enumerate(test_indices):
            learner_id = self._id_dictionary['sequences'][index]['learner_id']
            if learner_id not in predictions:
                predictions[learner_id] = {}
                
            predictions[learner_id][self._settings['data']['adjuster']['limit']] ={
                    'pred': test_pred[i],
                    'proba': test_proba[i],
                    'truth': test_y[i]
            }
            
        with open(path + 'predictions.pkl', 'wb') as fp:
            pickle.dump(predictions, fp)
            
    def _read_predictions(self, indices:list) -> list:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/results/'
        os.makedirs(path, exist_ok=True)
        if 'predictions.pkl' in os.listdir(path):
            with open(path + 'predictions.pkl', 'rb') as fp:
                predictions = pickle.load(fp)
            
            new_preds = []
            new_probas = []
            new_truths = []
            for iid in indices:
                learner_id = self._id_dictionary['sequences'][iid]['learner_id']
                if learner_id in predictions:
                    crop_limits = list(predictions[learner_id].keys())
                    crop_limits.sort()
                    new_preds.append(predictions[learner_id][crop_limits[-1]]['pred'])
                    new_probas.append(predictions[learner_id][crop_limits[-1]]['proba'])
                    new_truths.append(predictions[learner_id][crop_limits[-1]]['truth'])
            return new_preds, new_probas, new_truths
        else:
            logging.print('Some sequences were too short: {}'.format(indices))
            return []

    def _get_map(self) -> dict:
        label_map = self._settings['ML']['permutation']['label_map']
        if label_map == 'none':
            return lambda x: x

        if label_map == '2classes':
            map_path = '../data/experiment_keys/permutation_maps/closedopen.yaml'
            
        with open(map_path) as fp:
            map_label = yaml.load(fp, Loader=yaml.FullLoader)

        return map_label
        
    def _get_y_to_rankings(self, indices):
        rankings = pd.read_csv('../data/post_test/sim_details.tsv', index_col=0, sep='\t')
        rankings['username'] = rankings.index
        rankings['username'] = rankings['username'].apply(lambda x: str(int(str(x)[:-2])))
        id_rankings = {rankings.iloc[i]['username']: rankings.iloc[i]['permutation'] for i in range(len(rankings))}
        id_dictionary = self._settings['id_dictionary']
        vector_map = self._get_map()

        lids = [id_dictionary['sequences'][idx]['learner_id'] for idx in indices]
        rankings = [id_rankings[lid] for lid in lids]
        rankings = [vector_map['map'][str(ranking)] for ranking in rankings]
        return rankings
        
            
    def xval(self, x:list, y:list, indices:list) -> dict:
        # indices will refer to the actual indices from id _dictionary
        # index are the indices from the splits
        results = {}
        
        pipeline = PipelineMaker(self._settings)
        begins, sequences, ends, labels, indices = pipeline.load_data()
        results['x'] = sequences
        results['y'] = labels

        self._id_dictionary = pipeline.get_id_dictionary()
        self._id_indices = [x for x in indices]
        self._outer_splitter.set_indices(indices)
        results['id_indices'] = [x for x in indices]
        results['limit'] = self._settings['data']['adjuster']['limit']
        
        logging.debug('x:{}, y:{}'.format(x, y))
        results['optim_scoring'] = self._xval_settings['nested_xval']['optim_scoring']
        rankings = self._get_y_to_rankings(indices)
        for f, (train_index, test_index) in enumerate(self._outer_splitter.split(sequences, rankings)):
            logging.info('- ' * 30)
            logging.info('  Fold {}'.format(f))
            logging.debug('    train indices: {}'.format(train_index))
            logging.debug('    test indices: {}'.format(test_index))
            results[f] = {}
            results[f]['train_index'] = train_index
            results[f]['train_indices'] = [indices[idx] for idx in train_index]
            results[f]['test_index'] = test_index
            results[f]['test_indices'] = [indices[idx] for idx in test_index]

            # division train / test
            x_train = [sequences[xx] for xx in train_index]
            y_train = [labels[yy] for yy in train_index]
            train_index, x_train, y_train, short_train = self._pipeline.build_partial_sequence(begins, sequences, ends, labels, train_index, self._settings['ML']['pipeline']['train_pad'])
            results[f]['longenough_train_indices'] = [indices[idx] for idx in train_index]
            results[f]['tooshort_train'] = short_train
            results[f]['tooshort_trainindices'] = [indices[idx] for idx in short_train]

            x_test = [sequences[xx] for xx in test_index]
            y_test = [labels[yy] for yy in test_index]
            test_index, x_test, y_test, short_test = self._pipeline.build_partial_sequence(begins, sequences, ends, labels, test_index, self._settings['ML']['pipeline']['test_pad'])
            results[f]['longenough_test_indices'] = [indices[idx] for idx in test_index]
            results[f]['tooshort_test'] = short_test
            results[f]['tooshort_testindices'] = [indices[idx] for idx in short_test]
            results[f]['longenough_y_test'] = y_test
            
            x_resampled, y_resampled = self._sampler.sample(x_train, y_train)
            results[f]['x_resampled'] = x_resampled
            results[f]['y_resampled'] = y_resampled
            results[f]['oversample_indexes'] = self._sampler.get_indices()
            results[f]['oversample_indices'] = [results[f]['train_indices'][idx] for idx in results[f]['oversample_indexes']]

            # Train
            self._init_gs(f, results[f]['oversample_indices'])
            if len(x_resampled) < self._xval_settings['nested_xval']['inner_n_folds'] or len(x_test) == 0:
                continue

            self._gs.fit(x_resampled, y_resampled, f)
            if len(x_test) == 0:
                print(self._settings['data']['adjuster']['limit'])
                continue
                
            logging.debug('lens: {}, {}'.format(len(x_train), len(x_test)))
            print(self._settings['data']['adjuster']['limit'])
            print(len(train_index), len(test_index))
            print(np.array(x_train[0]).shape, np.array(x_test[0]).shape)
            print(x_train[0])
            
            y_pred = self._gs.predict(x_test)
            y_proba = self._gs.predict_proba(x_test)
            test_results = self._scorer.get_scores(y_test, y_pred, y_proba)
            self._write_predictions(y_pred, y_proba, y_test, results[f]['longenough_test_indices'] )
            logging.debug('    predictions: {}'.format(y_pred))
            logging.debug('    probability predictions: {}'.format(y_proba))
            results[f]['y_pred'] = y_pred
            results[f]['y_proba'] = y_proba
            results[f].update(test_results)
            
            # Carry on
            if len(short_test) > 0:
                pred, proba, truth = self._read_predictions(results[f]['tooshort_testindices'])
                pred = list(y_pred) + list(pred)
                proba = list(y_proba) + list(proba)
                truth = list(y_test) + list(truth)
                carry_on_results = self._scorer.get_scores(truth, pred, proba)
                results[f]['carry_on_scores'] = carry_on_results
            else:
                results[f]['carry_on_scores'] = test_results
            
            results[f]['best_params'] = self._gs.get_best_model_settings()
            best_estimator = self._gs.get_best_model()
            results[f]['best_estimator'] = best_estimator.save_fold(f)
            results[f]['gridsearch_object'] = self._gs.get_path(f)
            logging.info('    best parameters: {}'.format(results[f]['best_params']))
            logging.info('    estimator path: {}'.format(results[f]['best_estimator']))
            logging.info('    gridsearch path: {}'.format(results[f]['gridsearch_object']))
            
            self._model_notation = best_estimator.get_notation()
            self.save_results(results)
        return results

    def save_results(self, results):
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/results/' 
        os.makedirs(path, exist_ok=True)
        
        path += self._notation + '_m' + self._model_notation + '_l' + str(self._settings['data']['adjuster']['limit']) + '.pkl'
        with open(path, 'wb') as fp:
            pickle.dump(results, fp)
            
            