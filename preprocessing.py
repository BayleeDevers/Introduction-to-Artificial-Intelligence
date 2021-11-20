import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone

class PctChange(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.pct_change().replace({np.inf: 0, np.nan: 0})

class Densify(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.toarray()

sentiment_words = ['bull',
                    'bear',
                    'moon',
                    'lambo',
                    'buy',
                    'sell',
                    'crash',
                    'pump',
                    'alt',
                    'altcoin',
                    'shitcoin',
                    'ethereum',
                    'dip',
                    'fee',
                    'fees',
                    'scam',
                    'fud',
                    'fomo',
                    'bubble',
                    'late',
                    'early',
                    'worry']

def stack_windows(timeseries, window_len, units_ahead, pct_change=False):
    '''
    Returns a tuple of numpy arrays:
    The first being X with shape: (len(timeseries) - window_len, window_len)
    Each row of the ndarray corresponds to window_len consecutive days immidiately before
    the day used as a label
    The second being y with shape: (len(timeseries) - window_len,)
    '''
    X = np.stack([np.roll(timeseries, -1*i, axis=0) for i in range(window_len)], axis=1)
    X = X[:-(window_len + units_ahead),:]

    y = timeseries[:,0]
    if pct_change:
        y = np.stack([np.roll(y, -1*i, axis=0) for i in range(window_len, window_len + units_ahead + 1)], axis=1).mean(axis=1)
    else:
        y = np.roll(y, -(window_len + units_ahead))
    y = y[:-(window_len + units_ahead)]
    return X, y




# Data Pipeline used for preprocessing
price_pipeline = Pipeline([('pct_change', PctChange()), 
                            ('scaler', StandardScaler())
                            ])
other_pipeline = clone(price_pipeline)

text_pipeline = Pipeline([('bagowords', CountVectorizer(vocabulary=sentiment_words)),
                            ('densify', Densify()),
                            ('scaler', StandardScaler())
                            ])

full_pipeline = ColumnTransformer([('price_pipeline', price_pipeline, ['Open']),
                                    ('other_pipeline', other_pipeline, ['Volume_(BTC)', 'size', 'transaction_count']),
                                    ('textual', text_pipeline, 'title')
                                    ], remainder='drop')