from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np


class ProdCompOHE(BaseEstimator, TransformerMixin):
    def __init__(self, column_name='production_company'):
        self.column_name = column_name
        self.prod = None

    def fit(self, X, y=None):

        self.prod = pd.cut(X.value_counts(),
                        bins=[0, 1, 5, 20, 50, 500],
                        include_lowest=True,
                        labels=[1, 2, 3, 4, 5])
        return self

    def transform(self, X):
        # create columns by each unique genre
        return self
