import pickle

class ChemLabFilter:
    """This class implements filters to prune some students from the original dataset
    """

    def __init__(self, settings:dict, id_dictionary: dict):
        self._name = 'filter'
        self._notation = 'flt'

        self._settings = dict(settings)
        self._filter_settings = dict(settings['data']['filters'])
        self._id_dictionary = id_dictionary

    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation

    def filter_data(self):
        id_dictionary = {
            'sequences': {},
            'index': {}
        }

        if 'learner_ids' in self._filter_settings: # Ids to exclude
            with open(self._filter_settings['learner_ids'], 'rb') as fp:
                learner_ids = pickle.load(fp)

        if self._filter_settings['interactionlimit'] == 10:
            with open('../data/beerslaw/experiment_keys/over10.pkl', 'rb') as fp:
                over10 = pickle.load(fp)

        index = 0
        for i, idx in enumerate(self._id_dictionary['sequences']):
            with open(self._id_dictionary['sequences'][idx]['path'], 'rb') as fp:
                sim_dict = pickle.load(fp)
            if 'genders' in self._filter_settings:
                if sim_dict['gender'] not in self._filter_settings['genders']:
                    continue
            
            if 'years' in self._filter_settings:
                if sim_dict['year'] not in self._filter_settings['years']:
                    continue
            
            if 'learner_ids' in self._filter_settings:
                if sim_dict['learner_id'] not in learner_ids:
                    continue

            if 'permutations' in self._filter_settings:
                if sim_dict['permutation'] not in self._filter_settings['permutations']:
                    continue

            if 'timelimit' in self._filter_settings:
                if sim_dict['last_timestamp'] < self._filter_settings['timelimit']:
                    continue
            
            if 'interactionlimit' in self._filter_settings:
                if self._filter_settings['interactionlimit'] == 10:
                    if sim_dict['learner_id'] not in over10:
                        continue
                elif len(sim_dict['sequence']) < self._filter_settings['interactionlimit']:
                    continue

            id_dictionary['sequences'][index] = {
                'path': self._id_dictionary['sequences'][idx]['path'],
                'length': len(sim_dict['sequence']),
                'learner_id': sim_dict['learner_id']
            }
            id_dictionary['index'][sim_dict['learner_id']] = index
            index += 1

        return id_dictionary
            




            
                
