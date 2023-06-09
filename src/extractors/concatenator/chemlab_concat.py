import os
import yaml
import pickle

import numpy as np

class ChemlabConcatenate:
    """In the case where students used the simulation several time for different purposes, we can concatenate the sequences together to create the final id_dictionary
    """
    def __init__(self, path: str, tasks:list, limit: int):
        self._name = 'chemlab concatenator'
        self._notation = 'chemconcat'
        
        self._path = path # path where to find the sequenced_simulations
        self._tasks = tasks # tasks to concatenate
        self._limit = limit # minimum sequence size
        
    def concatenate(self):
        """Create the new id_dictionary with the said concatenated sequences
        """
        idds = {}
        for t in self._tasks:
            with open(self._path + 'id_dictionary' + str(t) + '.pkl', 'rb') as fp:
                idds[str(t)] = pickle.load(fp)

        id_dictionary = {
            'sequences': {},
            'index': {}
        }
        self._tasks.sort()
        for i in range(len(idds[str(t)]['sequences'])):
            sim_dict = {
                'sequence': [],
                'begin': [],
                'end': [],
                'permutation': '',
                'last_timestamp': '',
                'learner_id': ''
            }
            lt = 0
            
            third = False
            for n_task in self._tasks:
                p = idds[str(n_task)]['sequences'][i]['path']
                if 'data/beerslaw' not in p:
                    p = p.replace('/data/', '/data/beerslaw/')
                with open(p, 'rb') as fp:
                    sequenced = pickle.load(fp)
                sim_dict['sequence'] = sim_dict['sequence'] + sequenced['sequence']
                sim_dict['begin'] = sim_dict['begin'] + list(np.array(sequenced['begin']) + lt)
                sim_dict['end'] = sim_dict['end'] + list(np.array(sequenced['end']) + lt)
                
                sim_dict['permutation'] = sequenced['permutation']
                sim_dict['last_timestamp'] = sequenced['last_timestamp'] + lt
                sim_dict['learner_id'] = sequenced['learner_id']
                sim_dict['gender'] = sequenced['gender']
                sim_dict['year'] = sequenced['year']
                
                lt += sequenced['last_timestamp']
                
            path = '{}p_{}_lid{}_t{}_sequenced.pkl'.format(self._path, sim_dict['permutation'], sim_dict['learner_id'], str(''.join(self._tasks)))
            # path = self._path + 'p_' + sim_dict['permutation'] + '_lid' + sim_dict['learner_id'] + '_t' + str(''.join(self._tasks)) + '_sequenced.pkl'
            with open(path, 'wb') as fp:
                pickle.dump(sim_dict, fp)
            
            if len(sim_dict['sequence']) >= self._limit:
                id_dictionary['sequences'][i] = {
                    'path': path,
                    'length': len(sim_dict['sequence']),
                    'learner_id': sim_dict['learner_id']
                }
                id_dictionary['index'][sim_dict['learner_id']] = i
                
        with open(self._path + 'id_dictionary.pkl', 'wb') as fp:
            pickle.dump(id_dictionary, fp)
                
            
            
            