"""Uniwersalny model regresji logistycznej do klasyfikacji binarnej.

Nie zawiera żadnych założeń o konkretnej bazie danych.
"""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold,
    cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Podzial:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    grupy_train: pd.Series | None = None
    grupy_test: pd.Series | None = None


def podziel_dane(X, y, grupy=None, test_size=0.2, random_state=42):
    """Zwraca podział train/test; opcjonalnie separuje całe grupy (np. firmy)."""
    X = pd.DataFrame(X).copy()
    y = pd.Series(y, index=X.index).copy()
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError('X i y muszą mieć zgodne indeksy i długości.')
    if y.isna().any() or not set(y.unique()).issubset({0, 1}) or y.nunique() != 2:
        raise ValueError('Zmienna y musi zawierać obie klasy: 0 i 1, bez braków.')
    if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
        raise ValueError('Wszystkie cechy muszą być liczbowe; dane kategoryczne zakoduj wcześniej.')
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError('Dane zawierają +/-inf; należy je obsłużyć przed treningiem.')

    if grupy is None:
        ix_train, ix_test = train_test_split(np.arange(len(X)), test_size=test_size,
                                              stratify=y, random_state=random_state)
        g_train = g_test = None
    else:
        grupy = pd.Series(grupy, index=X.index)
        if grupy.isna().any():
            raise ValueError('Identyfikatory grup nie mogą zawierać braków.')
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        ix_train, ix_test = next(splitter.split(X, y, groups=grupy))
        g_train, g_test = grupy.iloc[ix_train], grupy.iloc[ix_test]

    X_train, X_test = X.iloc[ix_train].copy(), X.iloc[ix_test].copy()
    y_train, y_test = y.iloc[ix_train].copy(), y.iloc[ix_test].copy()
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError('W jednej części podziału brakuje klasy. Zmień losowanie lub proporcje podziału.')
    return Podzial(X_train, X_test, y_train, y_test, g_train, g_test)


def zbuduj_model(C=1.0, class_weight='balanced', random_state=42):
    """Imputacja medianą -> standaryzacja -> regresja logistyczna."""
    if C <= 0:
        raise ValueError('C musi być dodatnie.')
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=C, class_weight=class_weight,
                                           solver='lbfgs', max_iter=2000,
                                           random_state=random_state)),
    ])


def ocen_model(model, X_test, y_test, prog=0.5):
    """Metryki dla niezależnego testu; ROC-AUC i PR-AUC liczone z prawdopodobieństw."""
    if not 0 < prog < 1:
        raise ValueError('Próg musi należeć do (0, 1).')
    prawdopodobienstwa = model.predict_proba(X_test)[:, 1]
    predykcje = (prawdopodobienstwa >= prog).astype(int)
    return {
        'accuracy': accuracy_score(y_test, predykcje),
        'balanced_accuracy': balanced_accuracy_score(y_test, predykcje),
        'precision': precision_score(y_test, predykcje, zero_division=0),
        'recall': recall_score(y_test, predykcje, zero_division=0),
        'f1': f1_score(y_test, predykcje, zero_division=0),
        'roc_auc': roc_auc_score(y_test, prawdopodobienstwa),
        'pr_auc': average_precision_score(y_test, prawdopodobienstwa),
        'confusion_matrix': confusion_matrix(y_test, predykcje, labels=[0, 1]),
    }
