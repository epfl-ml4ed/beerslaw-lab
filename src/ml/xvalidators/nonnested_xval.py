import os
import yaml
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Tuple

from sklearn.model_selection import StratifiedKFold

from ml.samplers.sampler import Sampler
from ml.models.model import Model
from ml.splitters.splitter import Splitter
from ml.xvalidators.xvalidator import XValidator
from ml.scorers.scorer import Scorer
from ml.gridsearches.gridsearch import GridSearch

from sklearn.model_selection import train_test_split

from utils.config_handler import ConfigHandler

class NonNestedRankingXVal(XValidator):
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
    
    def __init__(self, settings:dict, gridsearch:GridSearch, inner_splitter:Splitter, gridsearch_splitter: Splitter, outer_splitter: Splitter, sampler:Sampler, model:Model, scorer:Scorer):
        super().__init__(settings, inner_splitter, model, scorer)
        self._name = 'nonnested cross validator'
        self._notation = 'nonnested_xval'

        
        settings['ML']['splitters']['n_folds'] = settings['ML']['xvalidators']['nested_xval']['inner_n_folds']
        self._gs_splitter = gridsearch_splitter # To create the folds within the gridsearch from the train set 
        settings['ML']['splitters']['n_folds'] = settings['ML']['xvalidators']['nested_xval']['outer_n_folds']
        self._outer_splitter = outer_splitter(settings) # to create the folds between development and test
        
        self._sampler = sampler()
        self._scorer = scorer(settings)
        self._gridsearch = gridsearch
        
        #debug
        self._model = model
        
    def _get_map(self) -> dict:
        label_map = self._settings['ML']['permutation']['label_map']
        if label_map == 'none':
            return lambda x: x

        if label_map == 'vector_labels':
            map_path = '../data/experiment_keys/permutation_maps/vector_binary.yaml'
            
        with open(map_path) as fp:
            map = yaml.load(fp, Loader=yaml.FullLoader)

        return lambda x: map['map'][x]
        
    def _get_y_to_rankings(self, indices):
        with open('../data/post_test/rankings.pkl', 'rb') as fp:
            rankings = pickle.load(fp)
            id_rankings = {rankings.iloc[i]['username']: rankings.iloc[i]['ranking'] for i in range(len(rankings))}
        id_dictionary = self._settings['id_dictionary']
        vector_map = self._get_map()

        lids = [id_dictionary['sequences'][idx]['learner_id'] for idx in indices]
        rankings = [id_rankings[lid] for lid in lids]
        rankings = [vector_map(ranking) for ranking in rankings]
        return rankings
        
    def xval(self, x:list, y:list, indices:list) -> dict:
        # indices will refer to the actual indices from id _dictionary
        # index are the indices from the splits
        results = {}
        results['x'] = x
        results['y'] = y
        results['indices'] = indices
        logging.debug('x:{}, y:{}'.format(x, y))
        results['optim_scoring'] = self._xval_settings['nested_xval']['optim_scoring'] #debug
        self._outer_splitter.set_indices(indices)
        rankings = self._get_y_to_rankings(indices)
        for f, (train_index, test_index) in enumerate(self._outer_splitter.split(x, rankings)):
            logging.debug('outer fold, length train: {}, length test: {}'.format(len(train_index), len(test_index)))
            logging.debug('outer fold: {}'.format(f))
            logging.info('- ' * 30)
            logging.info('  Fold {}'.format(f))
            logging.debug('    train indices: {}'.format(train_index))
            logging.debug('    test indices: {}'.format(test_index))
            results[f] = {}
            results[f]['train_index'] = train_index
            results[f]['train_indices'] = [indices[idx] for idx in train_index]
            results[f]['test_index'] = test_index
            results[f]['test_indices'] = [indices[idx] for idx in test_index]
            print(test_index)
            if f != self._settings['ML']['pipeline']['outerfold_index'] and self._settings['ML']['pipeline']['outerfold_index'] != -10:
                continue
            
            # division train / test
            x_train = [x[xx] for xx in train_index]
            y_train = [y[yy] for yy in train_index]
            x_test = [x[xx] for xx in test_index]
            y_test = [y[yy] for yy in test_index]
            
            # Inner loop
            x_resampled, y_resampled = self._sampler.sample(x_train, y_train)
            results[f]['oversample_indexes'] = self._sampler.get_indices()
            results[f]['oversample_indices'] = [results[f]['train_indices'][idx] for idx in results[f]['oversample_indexes']]
            
            logging.debug('  * data format: x [{}], y{}]'.format(np.array(x_resampled).shape, np.array(y_resampled).shape))
            #print(x_resampled)

            logging.debug('  * data details, mean: {};{} - std {};{}'.format(
                np.mean([np.mean(idx) for idx in x_resampled]),
                np.mean([np.mean(idx) for idx in y_resampled]),
                np.std([np.std(idx) for idx in x_resampled]),
                np.std([np.std(idx) for idx in y_resampled])
            ))
            
            # Train
            
            model = self._model(self._settings)
            if model.get_settings()['save_best_model']:
                train_x, val_x, train_y, val_y = train_test_split(x_resampled, y_resampled, test_size=0.1, random_state=129)
                results[f]['model_train_x'] = train_x
                results[f]['model_train_y'] = train_y
                results[f]['model_val_x'] = val_x
                results[f]['model_val_y'] = val_y
            else:
                train_x, train_y = x_resampled, y_resampled
                val_x, val_y = x_test, y_test

            for xxxx in train_x:
                print(len(xxxx))
            model.set_outer_fold(f)
            model.fit(train_x, train_y, x_val=val_x, y_val=val_y)
            results[f]['x_resampled'] = x_resampled
            results[f]['y_resampled'] = y_resampled
            results[f]['x_resampled_train'] = train_x
            results[f]['y_resampled_train'] = train_y
            results[f]['x_resampled_val'] = val_x
            results[f]['y_resampled_val'] = val_y
            results[f]['best_params'] = model.get_settings()
            results[f]['best_epochs'] = model.get_best_epochs()

            model.save_fold(f)

            # Predict
            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)
            test_results = self._scorer.get_scores(y_test, y_pred, y_proba)
            logging.debug('    predictions: {}'.format(y_pred))
            logging.debug('    probability predictions: {}'.format(y_proba))
            
            results[f]['y_pred'] = y_pred
            results[f]['y_proba'] = y_proba
            results[f].update(test_results)
            
            print('Best Results on outer fold: {}'.format(test_results))
            logging.info('Best Results on outer fold: {}'.format(test_results))
            self._model_notation = model.get_notation()
            self.save_results(results)
        return results
    
    def save_results(self, results):
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/results/' 
        os.makedirs(path, exist_ok=True)
        
        path += self._notation + '_m' + self._model_notation + '_l' + str(self._settings['data']['adjuster']['limit']) + '.pkl'
        with open(path, 'wb') as fp:
            pickle.dump(results, fp)
            
            