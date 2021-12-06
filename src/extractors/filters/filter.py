class Filter:
    """This class implements filters to prune some students from the original dataset
    """

    def __init__(self, settings:dict, id_dictionary: dict):
        self._name = 'filter'
        self._notation = 'flt'

    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation

    def filter_data(self):
        raise NotImplementedError
