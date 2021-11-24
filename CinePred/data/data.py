import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from datetime import date
from CinePred.data.utils import *
import numpy as np
from currency_converter import CurrencyConverter


class Data:
    '''
        class for cleaning,preprocessing and managing Data
    '''

    def __init__(self, link):
        '''
        init function for preprocessing data

        Parameters
        ----------
        link : str
            path to the csv file
        '''
        self.link = link
        self.dataframe = None

    def import_data(self):
        '''
        read the CSV file located in self.link
        '''
        self.dataframe = pd.read_csv(self.link, low_memory=False)
        return self

    def keep_columns(self, columns_names):
        '''
        keep columns specified in columns_names

        Parameters
        ----------
        columns_names : str array
            list of columns names to keep

        Examples
        --------
        >>> keep_columns(['budget','actors'])
        '''
        self.dataframe = self.dataframe[columns_names]
        return self

    def remove_na_rows(self):
        '''
        remove empy or NA rows
        '''
        self.dataframe = self.dataframe.dropna()
        return self

    def convert_income(self, column_name):
        '''
        convert income colomn in value $1000 -> 1000

        Parameters
        ----------
        column_name : str
            name of the column to convert
        '''

        self.dataframe[column_name] = self.dataframe[column_name].str.split()
        self.dataframe[column_name] = self.dataframe[column_name].apply(
            lambda x: x[1])
        self.dataframe[column_name] = self.dataframe[column_name].astype(
            str).astype(int)
        return self


    def convert_budget_column(self,column_name='budget', out_currency = 'USD', min_rows=45):
        '''
        convert budget column in USD value converted in int

        Parameters
        ----------
        columns_name : str , default 'budget'
            name of the budget column

        out_currency : str , default 'USD'
            the output of the currency conversion

        min_rows : int , default 45
            drop lines if number of line with this currency is under min_rows
        '''

        # supprime les espaces à la fin et au début
        self.dataframe[column_name] = self.dataframe[column_name].str.strip()

        # split la string en mots
        self.dataframe[column_name] = self.dataframe[column_name].str.split()

        # split in two columns
        self.dataframe['currency'] = self.dataframe[column_name].apply(
            lambda x: x[0])
        self.dataframe[column_name] = self.dataframe[column_name].apply(
            lambda x: x[1]).astype('int64')

        # select only rows with more than min_rows
        self.dataframe = self.dataframe.groupby('currency').filter(lambda x : len(x)>min_rows)

        c = CurrencyConverter()
        self.dataframe[column_name] = self.dataframe[[column_name,'currency']]\
            .apply(lambda x: convert(x[column_name], x['currency'], 'USD',converter = c), axis=1)
        self.dataframe = self.dataframe.drop(columns='currency')

        return self

    def filter_categories(self, column_name, nb=5):
        '''
        filter only nb(5) categories
        '''
        # separate all actors into lists
        category_list = self.dataframe[column_name].str.split(', ').tolist()
        #return top 5 actors
        self.dataframe[column_name]= [', '.join(category[:nb]) for category in category_list]

        return self

    def one_hot_encode(self,column_names):
        '''
        for cell with multiple categories, one hot encode for each column, each categories

        Parameters
        ----------
        columns_name : array str
            name list of the columns to encode
        '''
        for column_name in column_names :
            self.dataframe = one_hot_encode_multiple(
                self.dataframe, column_name)
        return self

    def convert_to_int(self,column_name):
        '''
        convert column to integer

        Parameters
        ----------
        column_name : str
            name of the column to convert
        '''
        self.dataframe[column_name] = self.dataframe[column_name].astype(
            str).astype(int)
        return self

    def convert_to_date(self, column_name, date_format='%Y-%m-%d'):
        '''
        convert column to datetime

        Parameters
        ----------
        column_name : str
            name of the column to convert

        date_format : str , default '%Y-%m-%d'
            format of the dates in the column
        '''
        self.dataframe[column_name] = pd.to_datetime(
            self.dataframe[column_name], format=date_format)
        return self

    def reset_index(self):
        '''
        reset index to clean dataframe
        '''
        self.dataframe = self.dataframe.reset_index()
        self.dataframe = self.dataframe.drop(columns='index')

        return self

    def add_sin_cos_features(self, column_name):
        '''
        seasonality: add sin & cos column for each month
        '''
        self.dataframe[column_name] = pd.DatetimeIndex(self.dataframe['date_published']).month
        months = 12
        self.dataframe["sin_MoPub"] = np.sin(2 * np.pi * self.dataframe.Month_published / months)
        self.dataframe["cos_MoPub"] = np.cos(2 * np.pi * self.dataframe.Month_published /months)

        return self


    def add_prod_company_category(self,existing_column_name, new_column_name):
        '''
        Categorize production company in 5 categories ranging from 1 to 5
        '''
        prod = pd.cut(self.dataframe[existing_column_name].value_counts(),
                      bins=[0, 1, 5, 20, 50, 500],
                      include_lowest=True,
                      labels=[1, 2, 3, 4, 5])
        self.dataframe[new_column_name] = self.dataframe[existing_column_name].map(lambda x: prod[str(x)])
        return self


def example():
    '''
    reset index to clean dataframe
    '''
    print('----- init Data -----')
    data = Data('raw_data/IMDb movies.csv')

    print('----- import Data -----')
    data.import_data()

    print('----- keep columns -----')
    data.keep_columns(columns_names=[
        'imdb_title_id', 'title', 'year', 'date_published', 'genre',
        'duration', 'country', 'director', 'writer', 'production_company',
        'actors', 'budget', 'worlwide_gross_income'
    ])

    print('----- remove na rows -----')
    data.remove_na_rows()

    print('----- convert budget -----')
    data.convert_budget_column(column_name='budget',min_rows=45, out_currency='USD')
    print('----- filter categories -----')
    data.filter_categories("actors", nb=2)

    print('----- convert income column -----')
    data.convert_income(column_name='worlwide_gross_income')

    print('----- one hot encode -----')
    #data.one_hot_encode(column_names=['director','genre'])
    #data.one_hot_encode(column_names=['genre'])
    #data.one_hot_encode(column_names=['director'])

    print('----- convert to int -----')
    data.convert_to_int('year')
    data.convert_to_int('duration')

    print('----- convert to date -----')
    data.convert_to_date('date_published')

    print('----- seasonality Sin/Cos -----')
    data.add_sin_cos_features('Month_published')

    print('----- categorize production company -----')
    data.add_prod_company_category("production_company", "production_weight")

    print('----- reset index -----')
    data.reset_index()

    print('----- data_shape -----')
    print(data.dataframe.shape)


if __name__ == "__main__":
    example()
