import os
import pickle
import logging
import itertools

import numpy as np
import pandas as pd

from ml.models.model import Model
from ml.splitters.splitter import Splitter
from ml.scorers.scorer import Scorer
from ml.gridsearches.gridsearch import GridSearch

class SupervisedGridSearch(GridSearch):
    def __init__(self, model:Model, grid:dict, scorer:Scorer, splitter:Splitter, settings:dict, outer_fold:int):
        super().__init__(model, grid, scorer, splitter, settings, outer_fold)
        self._name = 'supervised gridsearch'
        self._notation = 'supgs'
        
    def fit(self, x_train:list, y_train:list, fold:int):
        for i, combination in enumerate(self._combinations):
            logging.info('Testing parameters: {}'.format(combination))
            folds = []
            splitter = self._splitter(self._settings)
            for f, (train_index, validation_index) in enumerate(splitter.split(x_train, y_train)):
                logging.debug('    inner fold, train length: {}, test length: {}'.format(len(train_index), len(validation_index)))
                x_val = [x_train[xx] for xx in validation_index]
                y_val = [y_train[yy] for yy in validation_index]
                xx_train = [x_train[xx] for xx in train_index]
                yy_train = [y_train[yy] for yy in train_index]
                logging.debug('  *f{} data format: x [{}], y [{}]'.format(f, np.array(x_val).shape, np.array(y_val).shape))
                logging.debug('  *f{} data format: x [{}], y [{}]'.format(f, np.array(xx_train).shape, np.array(yy_train).shape))
        
                logging.debug('  * data details, mean: {};{} - std {};{}'.format(
                    np.mean([np.mean(idx) for idx in x_val]),
                    np.mean([np.mean(idx) for idx in y_val]),
                    np.std([np.std(idx) for idx in xx_train]),
                    np.std([np.std(idx) for idx in yy_train])
                ))   
                model = self._model(self._settings)
                model.set_outer_fold(self._outer_fold)
                model.set_gridsearch_fold(f)
                model.set_gridsearch_parameters(self._parameters, combination)
                model.fit(xx_train, yy_train, x_val=x_val, y_val=y_val)
                
                y_pred = model.predict(x_val)
                y_proba = model.predict_proba(x_val)
                
                score = self._scoring_function(y_val, y_pred, y_proba)
                logging.info('    Score for fold {}: {} {}'.format(f, score, self._scoring_name))
                folds.append(score)
            self._add_score(combination, folds)
            self.save(fold)
            
        best_parameters = self.get_best_model_settings()
        combinations = []
        for param in self._parameters:
            combinations.append(best_parameters[param])
            
        config = dict(self._settings)
        model = self._model(config)
        model.set_outer_fold(self._outer_fold)
        model.set_gridsearch_parameters(self._parameters, combinations)
        model.fit(x_train, y_train, x_val, y_val)
        self._best_model = model
        
            
    def predict(self, x_test: list) -> list:
        return self._best_model.predict(x_test)
        
        
    def predict_proba(self, x_test:list) -> list:
        return self._best_model.predict_proba(x_test)
