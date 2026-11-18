"""
Klasy użyte w Twoim pipeline. Model zapisano z klasą QuantileClipper zdefiniowaną
w skrypcie/notebooku treningowym (moduł __main__), więc aplikacja musi ją znać,
żeby wczytać plik. Parametry (lower_, upper_) są odczytywane z pliku modelu,
a ta klasa tylko je stosuje.

Jeśli Twoja oryginalna klasa robiła coś innego niż obcięcie wartości do
kwantyli – wklej tu jej prawdziwy kod.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)   # NaN pozostają NaN
