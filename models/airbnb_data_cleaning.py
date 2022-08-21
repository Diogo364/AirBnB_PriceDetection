import pandas as pd
from typing import Callable
from utils import parse_currency_str_to_float
from utils.outlier_removal import IQROutlierRemoval, AbsoluteOutlierRemoval
from utils.outlier_removal.outlier_removal_interface import OutlierRemovalInterface

class AirBnBDataCleaner:
    def __init__(self, 
                data: pd.DataFrame, 
                target_feature: str='price',
                numerical_features: list=[
                                        'construction_year', 
                                        'service_fee', 
                                        'minimum_nights', 
                                        'number_of_reviews', 
                                        'reviews_per_month', 
                                        'review_rate_number', 
                                        'availability_365',
                                        'calculated_host_listings_count'],
                categorical_features: list=[
                                        'host_identity_verified', 
                                        'neighbourhood', 'country', 
                                        'instant_bookable', 
                                        'cancellation_policy', 
                                        'room_type',
                                        'has_rules'],
                str_currency_columns: list=[
                                        'price', 
                                        'service_fee'],
                fillna_kw: dict={
                                        'reviews_per_month': 0,
                                        'service_fee': '$0'},
                currency_parser: Callable=parse_currency_str_to_float,
                data_filterining_features: list=[
                                        'minimum_nights',
                                        'price',
                                        'availability_365'],
                data_filtering_pipeline: list[OutlierRemovalInterface]=[
                                        IQROutlierRemoval(bottom=0.1),
                                        AbsoluteOutlierRemoval(bottom=50),
                                        AbsoluteOutlierRemoval(bottom=0, top=365)]
                ):
        self._target_feature = target_feature
        self._numerical_features = numerical_features
        self._categorical_features = categorical_features
        self._str_currency_columns = str_currency_columns
        self._fillna_kw = fillna_kw
        self.__currency_parser = currency_parser
        self._data_filterining_features = data_filterining_features
        self.__data_filtering_pipeline = data_filtering_pipeline
        
        self._all_features = numerical_features + categorical_features + [target_feature]

        self.__clean_data(data)
    
    def __clean_data(self, data: pd.DataFrame):
        data = self.__normalize_column_names(data)
        data = (
            data
            .drop_duplicates()
            .dropna(subset=[self._target_feature])
            .fillna(self._fillna_kw)
        )
        data = self.__parse_currency_str_to_float(data)
        data = self.__generate_has_rules_feature(data)
        data = self.__filter_data(data)
        self.__data = data[self._all_features].dropna()
            
    def get_data(self) -> pd.DataFrame:
        return self.__data

    @staticmethod
    def __normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
        data.columns = map(lambda x: x.replace(' ', '_').lower(), data.columns)
        return data

    @staticmethod
    def __has_column(data: pd.DataFrame, column: str) -> bool:
        return column in data.columns

    def __parse_currency_str_to_float(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.__currency_parser is None:
            return data
        
        assert isinstance(self.__currency_parser, Callable), 'Currency_parser parameter should be a function or None'
        for currency_col in self._str_currency_columns:
            data.loc[:, currency_col] = data.loc[:, currency_col].apply(parse_currency_str_to_float)
        return data
    
    def __generate_has_rules_feature(self, data: pd.DataFrame):
        if self.__has_column(data, 'house_rules'):
            data.loc[:, 'has_rules'] = (data.loc[:, 'house_rules'].notna()) & (data.loc[:, 'house_rules'] != '#NAME?')
        return data
    
    def __filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        for feature, out_removal in zip(self._data_filterining_features, self.__data_filtering_pipeline):
            data = data.dropna(subset=feature)
            stay_idx = out_removal.fit_transform(data[feature].values)
            data = data.iloc[stay_idx]
        return data
