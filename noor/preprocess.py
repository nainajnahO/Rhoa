from pandas.api.extensions import register_dataframe_accessor

@register_dataframe_accessor("preprocess")
class PreprocessAccessor:
    """Accessor for DataFrame preprocessing methods."""
    def __init__(self, pandas_obj):
        self._df = pandas_obj
