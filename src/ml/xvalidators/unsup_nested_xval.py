import os
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Tuple


from ml.samplers.sampler import Sampler
from ml.models.model import Model
from ml.splitters.splitter import Splitter
from ml.xvalidators.xvalidator import XValidator
from ml.scorers.scorer import Scorer
from ml.gridsearches.gridsearch import GridSearch


class UnsupNestedXVal(XValidator):
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
    
    def __init__(self, settings:dict, gridsearch:GridSearch, inner_splitter:Splitter, outer_splitter: Splitter, sampler:Sampler, model:Model, scorer:Scorer):
        super().__init__(settings, inner_splitter, model, scorer)
        self._name = 'unsupervised nested cross validator'
        self._notation = 'unsup_nested_xval'
        
        self._inner_folds = settings['ML']['xvalidators']['unsup_nested_xval']['inner_n_folds']
        self._inner_splitter =  inner_splitter(settings)
        self._inner_splitter.set_n_folds(settings['ML']['xvalidators']['unsup_nested_xval']['inner_n_folds'])
        self._outer_splitter = outer_splitter(settings)
        self._splitter = inner_splitter
        self._sampler = sampler()
        self._scorer = scorer(settings)
        self._gridsearch = gridsearch
        
        #debug
        self._model = model
        
    def _init_gs(self):
        self._scorer.set_optimiser_function(self._xval_settings['unsup_nested_xval']['optim_scoring'])
        self._gs = self._gridsearch(
            model=self._model,
            grid=self._xval_settings['unsup_nested_xval']['param_grid'],
            scorer=self._scorer,
            splitter = self._inner_splitter(self._settings),
            settings=self._settings
        )
        
    def _flatten(self, data:list) -> Tuple[dict, list]:
        students = {}
        sequences = []
        for i, sequence in enumerate(data):
            sequences.append([i])
            students[i] = sequence
        return students, sequences
    
    def _unflatten(self, students:dict, sequences:list) -> list:
        data = []
        for seq in sequences:
            data.append(students[seq[0]])
        return data
        
    def xval(self, x:list, y:list, indices:list) -> dict:
        results = {}
        results['x'] = x
        results['y'] = y
        logging.debug('x:{}, y:{}'.format(x, y))
        for f, (train_index, test_index) in enumerate(self._outer_splitter.split(x, y)):
            logging.info('- ' * 30)
            logging.info('  Fold {}'.format(f))
            logging.debug('    train indices: {}'.format(train_index))
            logging.debug('    test indices: {}'.format(test_index))
            results[f] = {}
            results[f]['train_index'] = train_index
            results[f]['test_index'] = test_index
            
            # division train / test
            x_train = [x[xx] for xx in train_index]
            y_train = [y[yy] for yy in train_index]
            x_test = [x[xx] for xx in test_index]
            y_test = [y[yy] for yy in test_index]
            
            # Inner loop
            ttrain_index, val_index = next(self._inner_splitter.split(x_train, y_train))
            x_val = [x_train[xx] for xx in val_index]
            x_train = [x_train[xx] for xx in ttrain_index]
            train_dict, x_train = self._flatten(x_train)
            y_train = [y_train[yy] for yy in ttrain_index]
            
            x_resampled, y_resampled = self._sampler.sample(x_train, y_train)
            x_resampled = self._unflatten(train_dict, x_resampled)
            
            results[f]['val_index'] = val_index
            results[f]['x_resampled'] = x_resampled
            results[f]['y_resampled'] = y_resampled
            
            # Train
            self._init_gs()
            self._gs.fit(x_resampled, y_resampled, x_val, f)
            
            # Predict
            y_pred, y_test = self._gs.predict(x_test)
            y_proba, y_test = self._gs.predict_proba(x_test)
            test_results = self._scorer.get_scores(y_test, y_pred, y_proba)
            logging.debug('    predictions: {}'.format(y_pred))
            logging.debug('    probability predictions: {}'.format(y_proba))
            
            results[f]['y_pred'] = y_pred
            results[f]['y_proba'] = y_proba
            results[f]['y_test'] = y_test
            results[f].update(test_results)
            
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
        
        path += self._notation + '_m' + self._model_notation + '.pkl'
        with open(path, 'wb') as fp:
            pickle.dump(results, fp)
            