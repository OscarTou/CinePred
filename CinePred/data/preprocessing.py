from CinePred.data.importing import import_data
import pandas as pd
import numpy as np
import os.path
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
from pandas.api.types import is_object_dtype

# --------------------------------------- #
# -------       preprocess        ------- #
# --------------------------------------- #

def keep_columns(df, column_names):
    '''
    keep columns specified in columns_names

    Parameters
    ----------
    df : DataFrame
        the source dataframe
    columns_names : str array
        list of columns names to keep
    '''
    return df[column_names]

def remove_na_rows(df):
    '''
    remove empy or NA rows
    '''
    df = df.dropna()
    df = df.reset_index()
    df = df.drop(columns='index')
    return df


def convert_income(df):
    '''
    convert income colomn in value $1000 -> 1000

    Parameters
    ----------
    column_name : str
        name of the column to convert
    '''

    df = df.iloc[:, 0].str.split()
    df = df.apply(lambda x: x[1])
    df = df.astype(str).astype(int)
    return df

def convert_to_int(df):
    '''
        convert column to integer

        Parameters
        ----------
        column_name : str
            name of the column to convert
        '''
    df = df.astype(str).astype(int)
    return pd.DataFrame(df)

def convert_to_date(df, date_format='%Y-%m-%d'):
    '''
    convert column to datetime

    Parameters
    ----------
    column_name : str
        name of the column to convert

    date_format : str , default '%Y-%m-%d'
        format of the dates in the column
    '''
    df = df.apply(lambda x: pd.to_datetime(x, format=date_format))
    return pd.DataFrame(df)

def convert_in_usd(val, cur, df_currencies):
    currency = df_currencies.loc[df_currencies['in'] == cur, 'value']
    if (len(currency) >= 1):
        return int(currency.item() * val)
    else:
        return 0

def reduce_column_type(df, nb_max=5):
    # separate all types into list
    df_copy = df.copy()
    types_list = df_copy.iloc[:, 0].str.split(',')
    df_copy = [', '.join(types[:nb_max]) for types in types_list]
    return pd.DataFrame(df_copy)


def convert_budget_column(df):
    '''
        convert budget column in USD value converted in int
    '''
    if (not is_numeric_dtype(df.iloc[:,0])):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../../raw_data/currencies.csv')
        df_currencies = import_data(filename)
        df_copy = df.copy()
        budget = df_copy.iloc[:, 0].str.split()
        df_copy['budget_cur'] = budget.apply(lambda x: x[0])
        df_copy['budget_val'] = budget.apply(lambda x: x[1]).astype('int64')

        result = pd.DataFrame({
            'budget_val': df_copy['budget_val'],
            'budget_cur': df_copy['budget_cur']
        }).apply(lambda x: convert_in_usd(x['budget_val'], x['budget_cur'],
                                        df_currencies),
                axis=1)

        return pd.DataFrame(result)
    return df

def log_transformation(df):
    df = np.log(df)/np.log(10)
    return pd.DataFrame(df)


def one_hot_encode_multiple(data, column_name, remove_column=True):
    # separate all genres into one list, considering comma + space as separators
    genre = data[column_name].str.split(', ').tolist()

    # flatten the list
    flat_genre = [item for sublist in genre for item in sublist]

    # convert to a set to make unique
    set_genre = set(flat_genre)

    # back to list
    unique_genre = list(set_genre)

    # create columns by each unique genre
    data = data.reindex(data.columns.tolist() + unique_genre,
                        axis=1,
                        fill_value=0)

    # for each value inside column, update the dummy
    for index, row in data.iterrows():
        for val in row[column_name].split(', '):
            if val != 'NA':
                data.loc[index, val] = 1

    data.drop(column_name, axis=1, inplace=True)
    return data


def preprocess_example(path='../raw_data/IMDb_movies.csv'):
    print('----- import Data -----')
    df = import_data(path)

    print('----- keep columns -----')
    df = keep_columns(df,
                      column_names=[
                          'imdb_title_id', 'title', 'year', 'date_published',
                          'genre', 'duration', 'country', 'director', 'writer',
                          'production_company', 'actors', 'budget',
                          'worlwide_gross_income'
                      ])

    print('----- remove na rows -----')
    df = remove_na_rows(df)

    print('----- convert budget -----')
    df['budget'] = convert_budget_column(df[['budget']])

    print('----- reduce column type -----')
    # df['actors'] = reduce_column_type(df[['actors']])

    print('----- convert income column -----')
    df['worlwide_gross_income'] = convert_income(df[['worlwide_gross_income']])

    print('----- convert to int -----')
    df['year'] = convert_to_int(df[['year']])
    df['duration'] = convert_to_int(df[['duration']])

    print('----- convert to date -----')
    df['date_published'] = convert_to_date(df[['date_published']])

    print('----- log transform -----')
    df['worlwide_gross_income'] = log_transformation(
        df[['worlwide_gross_income']])

    print('----- data_shape -----')

    return df





def reduce_column_type(data, column_name, nb_max=5):
    # separate all actors into lists
    actor_list = data[column_name].str.split(', ').tolist()
    #return top 5 actors
    data[column_name] = [','.join(actor[:nb_max]) for actor in actor_list]
    return data


if __name__ == "__main__":
    preprocess_example()
