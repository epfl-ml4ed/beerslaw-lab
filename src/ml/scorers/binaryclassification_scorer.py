import numpy as np
import pandas as pd
import logging
from typing import Tuple

from ml.scorers.scorer import Scorer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score

class BinaryClfScorer(Scorer):
    """This class is used to create a scorer object tailored towards binary classification

    Args:
        Scorer (Scorer): Inherits from scorer
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'binary classification scorer'
        self._notation = '2clfscorer'
        self._score_dictionary = {
            'accuracy': self._get_accuracy,
            'balanced_accuracy': self._get_balanced_accuracy,
            'precision': self._get_precision,
            'recall': self._get_recall,
            'roc': self._get_roc
        }
        
        self._croissant = {
            'accuracy': True,
            'balanced_accuracy': True,
            'precision': True,
            'recall': True,
            'roc': True
        }
        
        self._get_score_functions(settings)
        
    def _get_accuracy(self, y_true: list, y_pred: list, yprobs: list) -> float:
        return accuracy_score(y_true, y_pred)
    
    def _get_balanced_accuracy(self, y_true: list, y_pred: list, yprobs: list) -> float:
        return balanced_accuracy_score(y_true, y_pred)
    
    def _get_precision(self, y_true: list, y_pred: list, yprobs: list) -> float:
        return precision_score(y_true, y_pred)
    
    def _get_recall(self, y_true: list, y_pred: list, yprobs: list) -> float:
        return recall_score(y_true, y_pred)
    
    def _get_roc(self, y_true: list, y_pred: list, y_probs: list) -> float:
        if len(np.unique(y_true)) == 1:
            return -1
        return roc_auc_score(y_true, np.array(y_probs)[:, 1])
    
    def get_scores(self, y_true: list, y_pred: list, y_probs: list) -> dict:
        scores = {}
        for score in self._scorers:
            scores[score] = self._scorers[score](y_true, y_pred, y_probs)
            
        return scores
        
        