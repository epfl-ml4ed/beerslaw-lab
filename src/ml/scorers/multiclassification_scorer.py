import logging
import numpy as np
import pandas as pd
from typing import Tuple

from ml.scorers.scorer import Scorer
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy as CCE
from sklearn.metrics import roc_auc_score

class MultiClfScorer(Scorer):
    """This class is used to create a scorer object tailored towards multi class classification problems

    Args:
        Scorer ([type]): inherits from scorer
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'multi class classification scorer'
        self._notation = 'multiclfscorer'
        self._score_dictionary = {
            'cce' : self._get_cce,
            'accuracy': self._get_accuracy,
            'balanced_accuracy': self._get_balanced_accuracy,
            'balanced_auc': self._get_balanced_auc,
            'overall_auc': self._get_overall_auc,
            'roc': self._get_balanced_auc
        }
        self._croissant = {
            'cce': False,
            'accuracy': True,
            'balanced_accuracy': True,
            'balanced_auc': True,
            'overall_auc': True,
            'roc': True
        }
        
        self._get_score_functions(settings)
        self._cce = CCE()

    def _onehot(self, index):
        vec = list(np.zeros(self._n_classes))
        vec[index] = 1
        return vec
        
    def _get_cce(self, y_true:list, y_pred:list, y_probs:list) -> float:
        if len(y_true) == 0:
            return 0
        else:
            yt = [self._onehot(yy) for yy in y_true]
            return float(self._cce(yt, y_probs))
        
    def _get_accuracy(self, y_true:list, y_pred:list, y_probs:list) -> float:
        return np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
    
    def _get_balanced_accuracy(self, y_true:list, y_pred:list, y_probs:list) -> float:
        classes = np.unique(y_true)
        bacc = 0
        for cl in classes:
            indices = [x for x in list(range(len(y_true))) if y_true[x] == cl]
            preds = [y_pred[x] for x in indices]
            truths = [y_true[x] for x in indices]
            bacc += np.sum(np.array(preds) == np.array(truths)) / len(truths)
        bacc /= self._settings['experiment']['n_classes']
        return bacc
    
    def _get_balanced_auc(self, y_true:list, y_pred:list, y_probs:list) -> float:
        if len(np.unique(y_true)) < self._n_classes:
            return -1
        # print('ytrue', y_true)
        # print('yprobs', y_probs)
        return roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    
    def _get_overall_auc(self, y_true:list, y_pred:list, y_probs:list) -> float:
        if len(np.unique(y_true)) < self._n_classes:
            return -1
        return roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
        
    def get_scores(self, y_true: list, y_pred: list, y_probs: list) -> dict:
        # yt = [self.__onehot(xx) for xx in y_true]
        scores = {}
        for score in self._scorers:
            scores[score] = self._scorers[score](y_true, y_pred, y_probs)
            
        return scores
            
                