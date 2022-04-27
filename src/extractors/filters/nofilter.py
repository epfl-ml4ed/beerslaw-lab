class NoFilter:
    """No pruning
    """

    def __init__(self, settings:dict, id_dictionary: dict):
        self._name = 'nofilter'
        self._notation = 'noflt'
        self._id_dictionary = id_dictionary

    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation

    def filter_data(self):
        return self._id_dictionary
