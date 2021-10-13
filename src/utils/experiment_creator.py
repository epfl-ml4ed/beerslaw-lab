import os
import yaml
from os import path as pth
from datetime import datetime


def get_experiment_name(settings) -> str:
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
    
    path = '../experiments/' + settings['experiment']['root_name'] + '/'
    today = datetime.today().strftime('%Y-%m-%d')
    today = today.replace('-', '_')
    starting_index = 0
    
    # first index
    experiment_name = path + today + '_' + str(starting_index) + '/'
    while (pth.exists(experiment_name)):
        starting_index += 1
        experiment_name = path + today + '_' + str(int(starting_index)) + '/'
        
    os.makedirs(experiment_name, exist_ok=True)
    
    # Dumping settings into the experiment name folder
    with open(experiment_name + 'config.yaml', 'w') as fp:
        doc = yaml.dump(settings, fp)
        
    self.settings['experiment']['name'] = today + '_' + str(int(starting_index))
        



