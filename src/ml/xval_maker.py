import yaml
import logging
import numpy as np
import pandas as pd
from typing import Tuple

from ml.models.model import Model
from ml.models.classifiers.random_forest import RandomForest
from ml.models.classifiers.simple_nn import SimpleNN
from ml.models.classifiers.scikit_nn import ScikitNN
from ml.models.classifiers.svm import SVMClassifier
from ml.models.classifiers.sgd import SGDModel
from ml.models.classifiers.knn import KNNModel
from ml.models.classifiers.adaboost import ADABoostModel
from ml.models.classifiers.lstm import LSTMModel
from ml.models.classifiers.lstmcnn import LSTMCNNModel
from ml.models.classifiers.prior_lstmcnn import PriorLSTMCNNModel
from ml.models.classifiers.cnnlstm import CNNLSTMModel
from ml.models.classifiers.prior_cnnlstm import PriorCNNLSTMModel
from ml.models.classifiers.ssan import SSANModel
from ml.models.classifiers.stateaction_ssan import SASSANModel
from ml.models.classifiers.ssanlstm import SSANLSTMModel
from ml.models.classifiers.stateaction_ssanlstm import SASSANLSTMModel
from ml.models.classifiers.prior_ssan import PriorSSANModel
from ml.models.classifiers.prior_lstm import PriorLSTMModel
from ml.models.classifiers.rnn_attention import RNNAttentionModel
from ml.models.classifiers.rnnattention_concat import RNNAttentionConcatModel
from ml.models.classifiers.priorlast_attention import PriorLastAttentionModel
from ml.models.classifiers.lasttimestep_attention import LastTimestepAttentionModel
from ml.models.classifiers.timestep_attention import TimestepAttentionModel
from ml.models.modellers.pairwise_skipgram import PWSkipgram

from ml.samplers.sampler import Sampler
from ml.samplers.no_sampler import NoSampler
from ml.samplers.random_oversampler import RandomOversampler

from ml.scorers.scorer import Scorer
from ml.scorers.binaryclassification_scorer import BinaryClfScorer
from ml.scorers.multiclassification_scorer import MultiClfScorer

from ml.splitters.splitter import Splitter
from ml.splitters.stratified_kfold import StratifiedKSplit

from ml.xvalidators.xvalidator import XValidator
from ml.xvalidators.nested_xval import NestedXVal
from ml.xvalidators.unsup_nested_xval import UnsupNestedXVal
# from ml.xvalidators.early_nested_xval import EarlyNestedXVal
from ml.xvalidators.checkpoint_xval import CheckpointXVal
from ml.xvalidators.ranking_xval import RankingXVal
from ml.xvalidators.ranking_early_nested_xval import RankingEarlyNestedXVal
from ml.xvalidators.rankingseed_xval import RankingSeedXVal
from ml.xvalidators.rankingseed_early_xval import SeedRankingEarlyNestedXVal
from ml.xvalidators.nonnested_xval import NonNestedRankingXVal
from ml.xvalidators.ranking_early_nonnested_xval import RankingEarlyNonNestedXVal

from ml.gridsearches.gridsearch import GridSearch
from ml.gridsearches.supervised_gridsearch import SupervisedGridSearch
from ml.gridsearches.unsupervised_gridsearch import UnsupervisedGridSearch
from ml.gridsearches.checkpoint_gridsearch import CheckpointGridsearch
from ml.gridsearches.permutation_gridsearch import PermutationGridSearch

from ml.splitters.flat_stratified import FlatStratified
from ml.splitters.one_fold import OneFoldSplit

class XValMaker:
    """This script assembles the machine learning component and creates the training pipeline according to:
    
        - splitter
        - sampler
        - model
        - xvalidator
        - scorer
    """
    
    def __init__(self, settings:dict):
        logging.debug('initialising the xval')
        self._name = 'training maker'
        self._notation = 'trnmkr'
        self._settings = dict(settings)
        self._experiment_root = self._settings['experiment']['root_name']
        self._experiment_name = settings['experiment']['name']
        self._pipeline_settings = self._settings['ML']['pipeline']
        
        self._build_pipeline()
        
    # def _choose_splitter(self):
    #     if self._pipeline_settings['splitter'] == 'stratkf':
    #         self._splitter = StratifiedKSplit

    def get_gridsearch_splitter(self):
        return self._gs_splitter

    def get_sampler(self):
        return self._sampler

    def get_scorer(self):
        return self._scorer

    def get_model(self):
        return self._model

    def _choose_splitter(self, splitter:str) -> Splitter:
        if splitter == 'stratkf':
            return StratifiedKSplit
        if splitter == 'flatstrat':
            return FlatStratified
        if splitter == '1kfold':
            return OneFoldSplit
    
    def _choose_inner_splitter(self):
        self._inner_splitter = self._choose_splitter(self._pipeline_settings['inner_splitter'])

    def _choose_outer_splitter(self):
        self._outer_splitter = self._choose_splitter(self._pipeline_settings['outer_splitter'])

    def _choose_gridsearch_splitter(self):
        self._gs_splitter = self._choose_splitter(self._pipeline_settings['gs_splitter'])
            
    def _choose_sampler(self):
        if self._pipeline_settings['sampler'] == 'nosplr':
            self._sampler = NoSampler
            
        elif self._pipeline_settings['sampler'] == 'rdmos':
            self._sampler = RandomOversampler
            
    def _choose_model(self):
        logging.debug('model: {}'.format(self._pipeline_settings['model']))
        if self._pipeline_settings['task'] == 'modelling':
            if self._pipeline_settings['model'] == 'pwsg':
                self._model = PWSkipgram
            
        elif self._pipeline_settings['task'] == 'classification':
            if self._pipeline_settings['model'] == 'rf':
                self._model = RandomForest
                gs_path = './configs/gridsearch/gs_rf.yaml'
                
            elif self._pipeline_settings['model'] == '1nn':
                self._model = SimpleNN
                gs_path = './configs/gridsearch/gs_1nn.yaml'
            
            elif self._pipeline_settings['model'] == 'sknn':
                self._model = ScikitNN
                gs_path = './configs/gridsearch/gs_sknn.yaml'
            
            elif self._pipeline_settings['model'] == 'svc':
                self._model = SVMClassifier
                gs_path = './configs/gridsearch/gs_svc.yaml'
                
            elif self._pipeline_settings['model'] == 'sgd':
                self._model = SGDModel
                gs_path = './configs/gridsearch/gs_sgd.yaml'
                
            elif self._pipeline_settings['model'] == 'knn':
                self._model = KNNModel
                gs_path = './configs/gridsearch/gs_knn.yaml'
                
            elif self._pipeline_settings['model'] == 'adaboost':
                self._model = ADABoostModel
                gs_path = './configs/gridsearch/gs_ada.yaml'
            
            elif self._pipeline_settings['model'] == 'lstm':
                self._model = LSTMModel
                gs_path = './configs/gridsearch/gs_LSTM.yaml'

            elif self._pipeline_settings['model'] == 'prior_lstm':
                self._model = PriorLSTMModel
                gs_path = './configs/gridsearch/gs_LSTM.yaml'

            elif self._pipeline_settings['model'] == 'priorlast_attention':
                self._model = PriorLastAttentionModel
                gs_path = './configs/gridsearch/gs_LSTM.yaml'

            elif self._pipeline_settings['model'] == 'rnn_attention':
                self._model = RNNAttentionModel
                gs_path = './configs/gridsearch/gs_LSTM.yaml'
            elif self._pipeline_settings['model'] == 'rnnattention_concat':
                self._model = RNNAttentionConcatModel
                gs_path = './configs/gridsearch/gs_LSTM.yaml'

            elif self._pipeline_settings['model'] == 'lastts_attention':
                self._model = LastTimestepAttentionModel
                gs_path = './configs/gridsearch/gs_LSTM.yaml'

            elif self._pipeline_settings['model'] == 'ts_attention':
                self._model = TimestepAttentionModel
                gs_path = './configs/gridsearch/gs_LSTM.yaml'

            elif self._pipeline_settings['model'] == 'rnnattention_concat':
                self._model = RNNAttentionConcatModel
                gs_path = './configs/gridsearch/gs_LSTM.yaml'


            elif self._pipeline_settings['model'] == 'lstmcnn':
                self._model = LSTMCNNModel
                gs_path = './configs/gridsearch/gs_lstmcnn.yaml'
            elif self._pipeline_settings['model'] == 'prior_lstmcnn':
                self._model = PriorLSTMCNNModel
                gs_path = './configs/gridsearch/gs_lstmcnn.yaml'
            elif self._pipeline_settings['model'] == 'cnnlstm':
                self._model = CNNLSTMModel
                gs_path = './configs/gridsearch/gs_lstmcnn.yaml'
            elif self._pipeline_settings['model'] == 'prior_cnnlstm':
                self._model = PriorCNNLSTMModel
                gs_path = './configs/gridsearch/gs_lstmcnn.yaml'

            elif self._pipeline_settings['model'] == 'ssan':
                self._model = SSANModel
                gs_path = './configs/gridsearch/gs_ssan.yaml'
            elif self._pipeline_settings['model'] == 'prior_ssan':
                self._model = PriorSSANModel
                gs_path = './configs/gridsearch/gs_ssan.yaml'
            elif self._pipeline_settings['model'] == 'sassan':
                self._model = SASSANModel
                gs_path = './configs/gridsearch/gs_ssan.yaml'
            elif self._pipeline_settings['model'] == 'ssanlstm':
                self._model = SSANLSTMModel
                gs_path = './configs/gridsearch/gs_ssan.yaml'
            elif self._pipeline_settings['model'] == 'sassanlstm':
                self._model = SASSANLSTMModel
                gs_path = './configs/gridsearch/gs_ssan.yaml'

                
            if self._settings['ML']['pipeline']['gridsearch'] != 'nogs':
                with open(gs_path, 'r') as fp:
                    gs = yaml.load(fp, Loader=yaml.FullLoader)
                    self._settings['ML']['xvalidators']['nested_xval']['param_grid'] = gs
                    print(gs)
                    
    def _choose_scorer(self):
        if self._pipeline_settings['scorer'] == '2clfscorer':
            self._scorer = BinaryClfScorer
        elif self._pipeline_settings['scorer'] == 'multiclfscorer':
            self._scorer = MultiClfScorer
            
    def _choose_gridsearcher(self):
        if self._pipeline_settings['gridsearch'] == 'supgs':
            self._gridsearch = SupervisedGridSearch
        elif self._pipeline_settings['gridsearch'] == 'unsupgs':
            self._gridsearch = UnsupervisedGridSearch
        elif self._pipeline_settings['gridsearch'] == 'ckptgs':
            self._gridsearch = CheckpointGridsearch
        elif self._pipeline_settings['gridsearch'] == 'permgs':
            self._gridsearch = PermutationGridSearch
                
    def _choose_xvalidator(self):
        if 'nested' in self._pipeline_settings['xvalidator']:
            self._choose_gridsearcher()
        if self._pipeline_settings['xvalidator'] == 'nested_xval':
            self._xval = NestedXVal
        if self._pipeline_settings['xvalidator'] == 'unsup_nested_xval':
            self._xval = UnsupNestedXVal
        # if self._pipeline_settings['xvalidator'] == 'early_nested_xval':
        #     self._xval = EarlyNestedXVal
        if self._pipeline_settings['xvalidator'] == 'ckpt_xval':
            self._choose_gridsearcher()
            self._xval = CheckpointXVal
        if self._pipeline_settings['xvalidator'] == 'ranking_xval':
            self._choose_gridsearcher()
            self._xval = RankingXVal
        if self._pipeline_settings['xvalidator'] == 'ranking_earlynested':
            self._choose_gridsearcher()
            self._xval = RankingEarlyNestedXVal
        if self._pipeline_settings['xvalidator'] == 'rankingseed':
            self._choose_gridsearcher()
            self._xval = RankingSeedXVal
        if self._pipeline_settings['xvalidator'] == 'rankingseed_early':
            self._choose_gridsearcher()
            self._xval = SeedRankingEarlyNestedXVal
        if self._pipeline_settings['xvalidator'] == 'nonnested':
            self._gridsearch = {}
            self._xval = NonNestedRankingXVal
        if self._pipeline_settings['xvalidator'] == 'early_nonnested':
            self._gridsearch = {}
            self._xval = RankingEarlyNonNestedXVal
        self._xval = self._xval(self._settings, self._gridsearch, self._inner_splitter, self._gs_splitter, self._outer_splitter, self._sampler, self._model, self._scorer)
                
    def _build_pipeline(self):
        # self._choose_splitter()
        self._choose_inner_splitter()
        self._choose_outer_splitter()
        self._choose_gridsearch_splitter()
        self._choose_sampler()
        self._choose_model()
        self._choose_scorer()
        self._choose_xvalidator()
        
    def train(self, X:list, y:list, indices:list):
        results = self._xval.xval(X, y, indices)
        

