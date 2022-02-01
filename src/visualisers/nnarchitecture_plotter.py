import os
import yaml
import pickle

from ml import load_tf_models as model_loader

class NNArchitecturePlotter:

    def __init__(self, settings):
        self._name = 'architecture plotter'
        self._notation = 'arcpltr'
        self._settings = settings

    def _test_models(self, experiment:str) -> dict:
        """Crawl Tests Paths to retrieve the test models

        Args:
            experiment (str): path up untill the date

        Return:
            models: dictionary (fold: model)
        """
        models = model_loader.load_test_nn(experiment)
        return models

    def _plot_single_architecture(self, model):
        """experiments for 

        Args:
            experiment (str): [description]
        """
        layer = model._retrieve_attentionlayer()
        