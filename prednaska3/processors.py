import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables, na_label="Missing"):
        self.variables = variables
        self.na_label = na_label

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[variable].fillna(self.na_label, inplace=True)
        return X

class Mapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables, mappings):
        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[variable].map(self.mappings)
        return X

class MeanInputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y = None):
        self.means_ = X[self.variables].mean().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[variable].fillna(self.means_[variable], inplace=True)
        return X

class RareCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables, rare_tol=0.01, rare_label="Rare"):
        self.variables = variables
        self.rare_tol = rare_tol
        self.rare_label = rare_label

    def fit(self, X, y = None):
        self.freq_values_ = {}

        for variable in self.variables:
            counts = pd.Series(X[variable]).value_counts(normalize=True)
            self.freq_values_[variable] = list(counts[counts > self.rare_tol].index)
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = np.where(
                X[variable].isin(self.freq_values_[variable]),
                X[variable],
                self.rare_label
            )
        return X
