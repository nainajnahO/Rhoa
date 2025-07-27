from pandas.api.extensions import register_dataframe_accessor

@register_dataframe_accessor("indicators")
class IndicatorsAccessor:
    """Accessor for DataFrame technical indicator methods."""
    def __init__(self, pandas_obj):
        self._df = pandas_obj

    def sma(self, window=30):
        """Calculate simple moving average for each column in the DataFrame."""
        return self._df.rolling(window=window).mean()
