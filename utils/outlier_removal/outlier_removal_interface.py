from abc import ABC, abstractmethod
from numpy import typing as npt
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemovalInterface(ABC, BaseEstimator, TransformerMixin):
    def __init__(self, top=True, bottom=True) -> None:
        self._top = top
        self._bottom = bottom
    
    @abstractmethod
    def fit(self, X: npt.ArrayLike):
        return self
    
    @abstractmethod
    def transform(self, X: npt.ArrayLike):
        pass