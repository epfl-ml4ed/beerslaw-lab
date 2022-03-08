import os
import re
import pickle
import numpy as np
from tabnanny import check
from shutil import copytree, rmtree
import tensorflow as tf

import numpy as np
from extractors.pipeline_maker import PipelineMaker
from ml.xval_maker import XValMaker
from ml.models.model import Model

def _get_outer_inner(path:str):
    outer_fold_re = re.compile('.*([0-9]+)/logger')
    outer_fold = outer_fold_re.findall(path)[0]

    inner_fold_re = re.compile('.*f(\-*[0-9]+)_model_checkpoint/')
    inner_fold = inner_fold_re.findall(path)[0]

    return outer_fold, inner_fold

def _get_model(path:str):
    model = path.split('/')
    model = model[-3]
    return model


def _crawl_checkpoint_paths(experiment:str):
    """Retrieve the paths from the model checkpoints

    Args:
        experiment (str): experiment_path up untill the date stamp
    """
    # Collect paths
    checkpoint_paths = []
    for (dirpath, dirnames, filenames) in os.walk(experiment):
        files = [os.path.join(dirpath, file) for file in filenames]
        checkpoint_paths.extend(files)

    paths = [p for p in checkpoint_paths if 'saved_model.pb' in p and '/models/' not in p]
    paths = [p.replace('saved_model.pb', '') for p in paths]

    model_checkpoints = {}
    for path in paths:
        outer_fold, inner_fold = _get_outer_inner(path)
        model = _get_model(path)
        if model not in model_checkpoints:
            model_checkpoints[model] = {}
        if outer_fold not in model_checkpoints[model]:
            model_checkpoints[model][outer_fold] = {}
        model_checkpoints[model][outer_fold][inner_fold] = path

    return model_checkpoints

def _crawl_test_paths(experiment:str):
    """retrieves the best models from the test folds

    Args:
        experiment (str): [description]
    """
    checkpoint_paths = []
    for (dirpath, dirnames, filenames) in os.walk(experiment):
        files = [os.path.join(dirpath, file) for file in filenames]
        checkpoint_paths.extend(files)

    paths = [p for p in checkpoint_paths if 'saved_model.pb' in p and '/models/' in p]
    paths = [p.replace('saved_model.pb', '') for p in paths]

    model_checkpoints = {}
    outer_fold_re = re.compile('_f([0-9])/')
    for path in paths:
        print(path)
        outer_fold = outer_fold_re.findall(path)[0]
        model_checkpoints[outer_fold] = path

    return model_checkpoints

def _read_config_file(experiment:str):
    configs = []
    for (dirpath, dirnames, filenames) in os.walk(experiment):
        files = [os.path.join(dirpath, file) for file in filenames]
        configs.extend(files)
    configs = [cc for cc in configs if 'config.yaml' in cc]
    conf = [len(cc) for cc in configs]
    conf = np.argmin(conf)
    path = configs[conf]
    with open(path, 'rb') as fp:
        return pickle.load(fp)

def load_all_nn(experiment:str):
    """Loads a model from a checkpoint path

    Args:
        experiment (str): experiment_path up untill the date stamp
        outer_fold: outer fold from which to retrieve the model
        inner_fold: inner fold from which to retrieve the model
    """
    settings = _read_config_file(experiment)
    paths = _crawl_checkpoint_paths(experiment)
    temporary_path = '../experiments/temp_checkpoints/plotter/'

    pipeline = PipelineMaker(settings)
    sequences, labels, indices, id_dictionary = pipeline.build_data()

    xval = XValMaker(settings)
    model = xval.get_model()

    models = {}
    for model_name in paths:
        models[model_name] = {}
        for outer_fold in paths[model_name]:
            models[model_name][outer_fold] = {}
            for inner_fold in paths[model_name][outer_fold]:
                models[model_name][outer_fold][inner_fold] = model(settings)
                models[model_name][outer_fold][inner_fold].set_outer_fold(outer_fold)
                if os.path.exists(temporary_path):
                    rmtree(temporary_path)
                copytree(paths[model_name][outer_fold][inner_fold], temporary_path, dirs_exist_ok=True)
                models[model_name][outer_fold][inner_fold].load_model_weights(sequences, temporary_path)

    return models

def load_test_nn(experiment:str):
    """Loads a model from a checkpoint path

    Args:
        experiment (str): experiment_path up untill the date stamp
        outer_fold: outer fold from which to retrieve the model
        inner_fold: inner fold from which to retrieve the model
    """
    settings = _read_config_file(experiment)
    print(settings['data']['pipeline'])
    paths = _crawl_test_paths(experiment)
    temporary_path = '../experiments/temp_checkpoints/plotter/'

    pipeline = PipelineMaker(settings)
    sequences, labels, indices, id_dictionary = pipeline.build_data()

    xval = XValMaker(settings)
    model = xval.get_model()

    models = {}
    for outer_fold in paths:
        models[outer_fold] = model(settings)
        models[outer_fold].set_outer_fold(outer_fold)
        print(paths[outer_fold])
        copytree(paths[outer_fold], temporary_path, dirs_exist_ok=True)
        models[outer_fold].load_model_weights(sequences, temporary_path)

    return models

if __name__ == '__main__':
    experiment = '/Users/cock/kDrive/PhD/Projects/Labs/beerslaw-lab/experiments/nested/lstm/field_colourbreak/binconcepts/lstm/raw_full/2022_01_24_0'
    models = load_all_nn(experiment)



