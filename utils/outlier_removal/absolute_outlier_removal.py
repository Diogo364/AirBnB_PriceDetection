from typing import Union
import numpy as np
from numpy import typing as npt
from .outlier_removal_interface import OutlierRemovalInterface

class AbsoluteOutlierRemoval(OutlierRemovalInterface):
    def __init__(self, top: float=None, bottom= None) -> None:
        super().__init__(top, bottom)
    
    def fit(self, X: npt.ArrayLike):
        self.lower_filter_ = self._bottom if self._bottom is not None else np.min(X)
        self.top_filter_ = self._top if self._top is not None else np.max(X)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        return np.where(
            (X >= self.lower_filter_) &
            (X <= self.top_filter_)
        )

if __name__ == '__main__':
    a = np.asarray([1, 2, 1, 2, 3, 2, 20, 30])
    iqr_removal = AbsoluteOutlierRemoval(bottom=2)
    iqr_removal.fit(a)

    print(a)
    print(a[iqr_removal.transform(a)])