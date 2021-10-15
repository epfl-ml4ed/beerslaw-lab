import numpy as np
import pandas as pd
import logging

from typing import Tuple
class Model:
    """This implements the superclass which will be used in the machine learning pipeline
    """
    
    def __init__(self, settings: dict):
        self._name = 'model'
        self._notation = 'm'
        self._settings = dict(settings)
        self._experiment_root = settings['experiment']['root_name']
        self._experiment_name = settings['experiment']['name']
        self._n_classes = settings['experiment']['n_classes']
        self._random_seed = settings['experiment']['random_seed']

        self._gs_fold = -1
    
    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation
    
    def set_gridsearch_parameters(self, params, combinations):
        logging.debug('Gridsearch params: {}'.format(params))
        logging.debug('Combinations: {}'.format(combinations))
        for i, param in enumerate(params):
            logging.debug('  index: {}, param: {}'.format(i, param))
            self._model_settings[param] = combinations[i]

    def set_gridsearch_fold(self, fold:int):
        self._gs_fold = fold
            
    def get_settings(self):
        return dict(self._model_settings)
        
    def _format(self, x: list, y: list) -> Tuple[list, list]:
        """formats the data into list or numpy array according to the library the model comes from

        Args:
            x (list): features
            y (list): labels

        Returns:
            x: formatted features
            y: formatted labels
        """
        raise NotImplementedError
    
    def _format_features(self, x: list) -> list:
        """formats the data into list or numpy array according to the library the model comes from

        Args:
            x (list): features

        Returns:
            x: formatted features
        """
        raise NotImplementedError
    
    def _init_model(self):
        """Initiates a model with self._model
        """
    
    def fit(self, x_train: list, y_train: list, x_val: list, y_val:list):
        """fits the model with the training data x, and labels y. 
        Warning: Init the model every time this function is called

        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            x_val (list): validation feature data
            y_val (list): validation label data
        """
        raise NotImplementedError
    
    def predict(self, x: list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """
        raise NotImplementedError
    
    def _inpute_full_prob_vector(self, y_pred:list, y_probs:list) -> list:
        """Sometimes, during nested cross validation, samples from minority classes are missing. The probability vector is thus one cell too short. However, we can recover the mapping position -> original label via the predict function

        Returns:
            list: new probability vector, where the number of cell is the 
        """
        label_map = {cl:[] for cl in range(self._n_classes)}
        prob_index = [np.argmax(y) for y in y_probs]
        prob_value = [max(y) for y in y_probs]
        
        for index in range(len(y_probs)):
            if prob_value[index] > 0.5:
                label_map[prob_index[index]].append(y_pred[index])
                
        new_map = {cl:np.unique(label_map[cl]) for cl in range(self._n_classes)}
        new_probs = np.zeros((len(y_probs), self._n_classes))
        for label in new_map:
            assert len(new_map[label]) <= 1
            
        for index, prob in enumerate(y_probs):
            for i in range(len(prob)):
                new_probs[index][new_map[i]] = prob[i]
            
        return new_probs
    
    def predict_proba(self, x:list) -> list:
        """Predict the probabilities of each label for x

        Args:
            x (list): features

        Returns:
            list: list of probabilities for each data point
        """
        raise NotImplementedError
    
    def save(self) -> str:
        """Saving the model in the following path:
        '../experiments/run_year_month_day/models/model_name_fx.pkl

        Returns:
            String: Path
        """
        raise NotImplementedError
    
    def save_fold(self, fold) -> str:
        """Saving the model for a specific fold in the following path:

        '../experiments/run_year_month_day/models/model_name_fx.pkl

        Returns:
            String: Path
        """
        raise NotImplementedError
        
