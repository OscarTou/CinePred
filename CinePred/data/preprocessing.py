from CinePred.data.importing import import_data
import pandas as pd
import numpy as np

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
    return df.dropna()

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
    types_list = df.iloc[:, 0].str.split(', ').tolist()
    #return top 5 actors
    df = [', '.join(types[:nb_max]) for types in types_list]
    return pd.DataFrame(df)

def convert_budget_column(df):
    '''
        convert budget column in USD value converted in int
    '''

    df_currencies = import_data('raw_data/currencies.csv')
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

def log_transformation(df):
    df = np.log(df)/np.log(10)
    return pd.DataFrame(df)

def reset_index(df):
    '''
    reset index to clean dataframe
    '''
    df = df.reset_index()
    df = df.drop(columns='index')

    return df

def preprocess_example(path='raw_data/IMDb movies.csv'):
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
    df['actors'] = reduce_column_type(df[['actors']], nb_max=2)

    print('----- convert income column -----')
    df['worlwide_gross_income'] = convert_income(df[['worlwide_gross_income']])

    print('----- convert to int -----')
    df['year'] = convert_to_int(df[['year']])
    df['duration'] = convert_to_int(df[['duration']])

    print('----- convert to date -----')
    df['date_published'] = convert_to_date(df[['date_published']])

    print('----- reset index -----')
    df = reset_index(df)

    print('----- log transform -----')
    df['worlwide_gross_income'] = log_transformation(
        df[['worlwide_gross_income']])

    print('----- data_shape -----')

    return df

def import_clean_df():
    # IMPORT DF
    data = Data('raw_data/IMDb movies.csv')
    data.import_data()

    # CLEANING
    data.keep_columns(columns_names=[
        'imdb_title_id', 'title', 'year', 'date_published', 'genre',
        'duration', 'country', 'director', 'writer', 'production_company',
        'actors', 'budget', 'worlwide_gross_income'
    ])
    data.remove_na_rows()
    data.convert_income(column_name='worlwide_gross_income')
    data.convert_to_date(column_name='date_published')
    data.dataframe.sort_values(by='date_published', inplace=True)
    data.dataframe.reset_index(inplace=True)

    return data

if __name__ == "__main__":
    preprocess_example()
