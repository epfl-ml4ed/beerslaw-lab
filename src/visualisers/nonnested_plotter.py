import os
import re
import yaml
import json
import pickle

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


class NonNestedPlotter:
    """This class plots results about the non nested group. Look into edm 2022 notebook to finalise"""

    def __init__(self, settings:dict):
        self._settings = settings
        

    def _crawl_paths(experiment_name):
        directory = '../experiments/{}/'.format(experiment_name)
        paths = []
        for (dirpath, dirnames, filenames) in os.walk(directory):
            files = [os.path.join(dirpath, file) for file in filenames]
            paths.extend(files)
        configs = [path for path in paths if 'config.yaml' in path]
        xvals = [path for path in paths if 'xval' in path and 'results' in path]

        full_paths = {}        
        date_re = re.compile('(.*202[0-9]_[0-9]+_[0-9]+_[0-9]+/)')
        model_re = re.compile('.*logger/(.*)/')
        fold_re = re.compile('.*f([\-0-9]+)')
        for config_path in configs:
            try:
                experiment_date_path = date_re.findall(config_path)[0]
                exp_xvals = [xval for xval in xvals if experiment_date_path in xval]
                full_paths[experiment_date_path] = {
                    'config': config_path,
                    'xval': exp_xvals[0],
                }
            except IndexError:
                continue

        return full_paths



