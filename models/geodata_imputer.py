import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class KNNGeoDataImputer(BaseEstimator, TransformerMixin):
    def __init__(
                self, 
                n_neighbors: int= 3,
                geographic_cat_info: list=[
                    'neighbourhood', 
                    'country', 
                    'country_code'
                ],
                geographic_coordinates: list=[
                    'lat', 'long']
                    ):
        self.geographic_cat_info = geographic_cat_info
        self.geographic_coordinates = geographic_coordinates
        self.n_neighbors = n_neighbors
        self.__models = {
            geo_info: KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
                for geo_info in geographic_cat_info
        }
    
    def fit(self, X: pd.DataFrame):
        assert self.__validate_columns(X, self.geographic_coordinates), f'Columns not present in X dataframe. Columns: {self.geographic_coordinates}'
        assert self.__validate_columns(X, self.geographic_cat_info), f'Columns not present in X dataframe. Columns: {self.geographic_cat_info}'

        for geo_info in self.geographic_cat_info:
            self.__models[geo_info].fit(X[self.geographic_coordinates].values, X[geo_info].values)

        return self
    
    def transform(self, X: pd.DataFrame):
        X = X[self.geographic_coordinates].values.reshape(-1, 2)
        return np.asarray(
            [self.__models[geo_info].predict(X) for geo_info in self.geographic_cat_info]
            ).T

    def __validate_columns(self, X: pd.DataFrame, column_list: list) -> bool:
        return all([col in X.columns for col in column_list])
