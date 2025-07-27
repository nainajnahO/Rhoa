from pandas.api.extensions import register_dataframe_accessor

@register_dataframe_accessor("preprocess")
class PreprocessAccessor:
    """Accessor for DataFrame preprocessing methods."""
    def __init__(self, pandas_obj):
        self._df = pandas_obj

    def drop_na(self):
        """Drop rows with NA values from the DataFrame."""
        return self._df.dropna()
