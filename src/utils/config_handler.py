import os
import yaml
import string
from os import path as pth
from datetime import datetime

class ConfigHandler:
    def __init__(self, settings:dict):
        self._settings = settings
        
    def get_settings(self):
        return dict(self._settings)
        
    def handle_gridsearch(self):
        if self._settings['ML']['pipeline']['model'] == '1nn':
            path = './configs/gridsearch/gs_1nn.yaml'
            with open(path, 'r') as fp:
                gs = yaml.load(fp, Loader=yaml.FullLoader)
                self._settings['ML']['xvalidators']['nested_xval']['param_grid'] = gs
                
        elif self._settings['ML']['pipeline']['model'] == 'rf':
            path = './configs/gridsearch/gs_rf.yaml'
            with open(path, 'r') as fp:
                gs = yaml.load(fp, Loader=yaml.FullLoader)
                self._settings['ML']['xvalidators']['nested_xval']['param_grid'] = gs
                
        elif self._settings['ML']['pipeline']['model'] == 'sknn':
            path = './configs/gridsearch/gs_sknn.yaml'
            with open(path, 'r') as fp:
                gs = yaml.load(fp, Loader=yaml.FullLoader)
                self._settings['ML']['xvalidators']['nested_xval']['param_grid'] = gs
                
        elif self._settings['ML']['pipeline']['model'] == 'sknn':
            path = './configs/gridsearch/gs_svc.yaml'
            with open(path, 'r') as fp:
                gs = yaml.load(fp, Loader=yaml.FullLoader)
                self._settings['ML']['xvalidators']['nested_xval']['param_grid'] = gs

    def get_experiment_name(self):
        """Creates the experiment name in the following path:
            '../experiments/experiment root/yyyy_mm_dd_index/'
            index being the first index in increasing order starting from 0 that does not exist yet.
            
            This function:
            - returns the experiment config name 
            - creates the folder with the right experiment name at ../experiments/experiment root/yyyy_mm_dd_index
            - dumps the config in the newly created folder

        Args:
            settings ([type]): read config

        Returns:
            [str]: Returns the name of the experiment in the format of 'yyyy_mm_dd_index'
        """
        path = '../experiments/' + self._settings['experiment']['root_name'] + '/'
        today = datetime.today().strftime('%Y-%m-%d')
        today = today.replace('-', '_')
        starting_index = 0
        
        # first index
        experiment_name = path + today + '_' + str(starting_index) + '/'
        while (pth.exists(experiment_name)):
            starting_index += 1
            experiment_name = path + today + '_' + str(int(starting_index)) + '/'
            
        self._experiment_path = experiment_name
        os.makedirs(self._experiment_path, exist_ok=True)
        
        self._settings['experiment']['name'] = today + '_' + str(int(starting_index))
        
    def handle_scorer(self) -> dict:
        experiment_path = '../experiments/' + self._settings['experiment']['root_name'] + '/'
        experiment_path += self._settings['experiment']['name'] 
        if self._settings['experiment']['n_classes'] > 2:
            self._settings['ML']['pipeline']['scorer'] = 'multiclfscorer'
        else:
            self._settings['ML']['pipeline']['scorer'] = '2clfscorer'
            
        with open(experiment_path + 'config.yaml', 'w') as fp:
            doc = yaml.dump(self._settings, fp)
        return self._settings
        
    def handle_settings(self) -> dict:
        self.handle_gridsearch()
        self.get_experiment_name()
        with open(self._experiment_path + 'config.yaml', 'w') as fp:
            doc = yaml.dump(self._settings, fp)
        return self._settings
    
    def handle_newpair(self, exp_path:str) -> dict:
        path = '../experiments/' + self._settings['experiment']['root_name'] + '/' + exp_path + '/'
        os.makedirs(path, exist_ok=True)
        today = datetime.today().strftime('%Y-%m-%d')
        today = today.replace('-', '_')
        starting_index = 0
        
        # first index
        experiment_name = path + today + '_' + str(starting_index) + '/'
        while (pth.exists(experiment_name)):
            starting_index += 1
            experiment_name = path + today + '_' + str(int(starting_index)) + '/'
            
        self._experiment_path = experiment_name
        os.makedirs(self._experiment_path, exist_ok=True)
        
        self._settings['experiment']['name'] = exp_path + today + '_' + str(int(starting_index))
        with open(experiment_name + 'config.yaml', 'w') as fp:
            doc = yaml.dump(self._settings, fp)
        return self._settings
    
    def handle_experiment_name(self):
        path = '../experiments/' + self._settings['experiment']['root_name'] + '/'
        os.makedirs(path, exist_ok=True)
        with open(path + 'config.yaml', 'w') as fp:
            yaml.dump(self._settings, fp)
        return self._settings
    
            
 