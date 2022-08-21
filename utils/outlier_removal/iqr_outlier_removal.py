import numpy as np
from scipy.stats import iqr
from numpy import typing as npt
from .outlier_removal_interface import OutlierRemovalInterface

class IQROutlierRemoval(OutlierRemovalInterface):
    def __init__(self, top: float=0.75, bottom: float=0.25, wiskers_ratio: float=1.5) -> None:
        super().__init__(top, bottom)
        self.wiskers_ratio_ = wiskers_ratio
    
    def fit(self, X: npt.ArrayLike):
        self.bottom_quantile_ = np.quantile(X, self._bottom)
        self.top_quantile_ = np.quantile(X, self._top)
        self.iqr_ = iqr(X)
        self.lower_filter_ = self.bottom_quantile_ - (self.iqr_ * self.wiskers_ratio_)
        self.top_filter_ = self.top_quantile_ + (self.iqr_ * self.wiskers_ratio_)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        return np.where(
            (X >= self.lower_filter_) &
            (X <= self.top_filter_)
        )

if __name__ == '__main__':
    a = np.asarray([1, 2, 1, 2, 3, 2, 20, 30])
    iqr_removal = IQROutlierRemoval()
    iqr_removal.fit(a)

    print(a)
    print(a[iqr_removal.transform(a)])