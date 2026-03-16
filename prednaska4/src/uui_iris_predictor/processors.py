import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MeanInputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y = None):
        self.means_ = X[self.variables].mean().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X.fillna({variable: self.means_[variable]}, inplace=True)
        return X
