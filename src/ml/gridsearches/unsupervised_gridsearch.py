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

class UnsupervisedGridSearch(GridSearch):
    def __init__(self, model:Model, grid:dict, scorer:Scorer, splitter:Splitter, settings:dict, outer_fold:int):
        super().__init__(model, grid, scorer, splitter, settings, outer_fold)
        self._name = 'unsupervised gridsearch'
        self._notation = 'unsupgs'
        
    def fit(self, x_train:list, y_train:list, x_test:list, fold:int):
        for i, combination in enumerate(self._combinations):
            logging.info('Testing parameters: {}'.format(combination))
            folds = []
            for f, (train_index, validation_index) in enumerate(self._splitter.split(x_train, y_train)):
                x_val = [x_train[xx] for xx in validation_index]
                xx_train = [x_train[xx] for xx in train_index]
                
                model = self._model(self._settings)
                model.set_gridsearch_parameters(self._parameters, combination)
                model.fit(xx_train, x_val)
                
                y_pred, y_test = model.predict(x_test)
                y_proba, y_test = model.predict_proba(x_test)
                assert y_test == y_test
                
                score = self._scoring_function(y_test, y_pred, y_proba)
                folds.append(score)
                logging.info('    Score for fold {}: {} {}'.format(f, score, self._scoring_name))
            self._add_score(combination, folds)
            self.save(fold)
            
        config = dict(self._settings)
        model = self._model(config)
        model.fit(x_train, x_test)
        self._best_model = model
            
    def predict(self, x_test: list) -> list:
        return self._best_model.predict(x_test)
        
        
    def predict_proba(self, x_test:list) -> list:
        return self._best_model.predict_proba(x_test)
