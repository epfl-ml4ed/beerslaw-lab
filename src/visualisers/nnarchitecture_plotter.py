import os
import yaml
import pickle

from ml import load_tf_models as model_loader

class NNArchitecturePlotter:

    def __init__(self, settings):
        self._name = 'architecture plotter'
        self._notation = 'arcpltr'
        self._settings = settings