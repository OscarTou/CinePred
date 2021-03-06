from os import path
from numpy.lib.shape_base import column_stack
import pandas as pd
import numpy as np

from CinePred.data.importing import import_data
from CinePred.data.preprocessing import preprocess_example
import cpi
cpi.update()

# --------------------------------------- #
# -------        featuring        ------- #
# --------------------------------------- #

def add_sin_features(df):
    '''
    seasonality: add sin column for each month
    '''
    df_copy = df.copy()
    months = df_copy.iloc[:,0].apply(lambda x: x.month)
    result = np.sin(2 * np.pi * months / 12)
    return pd.DataFrame(result)

def add_cos_features(df):
    '''
    seasonality: add cos column for each month
    '''
    df_copy = df.copy()
    months = df_copy.iloc[:, 0].apply(lambda x: x.month)
    result = np.cos(2 * np.pi * months / 12)
    return pd.DataFrame(result)


def add_sin_cos_features(df):
    '''
    seasonality: add sin & cos column for each month
    '''
    df_copy = df.copy()
    months = df_copy.iloc[:, 0].apply(lambda x: x.month)
    result = pd.DataFrame({
        'sin': np.sin(2 * np.pi * months / 12),
        'cos': np.cos(2 * np.pi * months / 12)
    })
    return result

def add_director_category(df):
    '''
    Categorize director in 3 categories ranging from 1 to 3
    '''
    df_copy = df.copy()
    prod = pd.cut(df_copy.iloc[:, 0].value_counts(),
                  bins=[0, 2, 10, 50],
                  include_lowest=True,
                  labels=[1, 2, 3])
    return pd.DataFrame(df_copy.iloc[:, 0].apply(lambda x: prod[str(x)]))


def add_prod_company_category(df):
    '''
    Categorize production company in 5 categories ranging from 1 to 5
    '''
    df_copy = df.copy()
    prod = pd.cut(df_copy.iloc[:, 0].value_counts(),
                  bins=[0, 1, 5, 20, 50, 500],
                  include_lowest=True,
                  labels=[1, 2, 3, 4, 5])
    return pd.DataFrame(df_copy.iloc[:, 0].apply(lambda x: prod[str(x)]))


def add_writer_category(df):
    '''
    Categorize writer in 5 categories ranging from 1 to 3
    '''
    df_copy = df.copy()
    prod = pd.cut(df_copy.iloc[:, 0].value_counts(),
                  bins=[0, 1, 5, 40],
                  include_lowest=True,
                  labels=[1, 2, 3])
    return pd.DataFrame(df_copy.iloc[:, 0].apply(lambda x: prod[str(x)]))


def prod_count_times(df):
    df_copy = df.copy()
    count_times = df_copy.iloc[:, 0].value_counts()
    result = df_copy.iloc[:, 0].apply(lambda x: count_times[str(x)])
    return pd.DataFrame(result)

def writer_count_times(df):
    df_copy = df.copy()
    count_times = df_copy.iloc[:, 0].value_counts()
    result = df_copy.iloc[:, 0].apply(lambda x: count_times[str(x)])
    return pd.DataFrame(result)

def director_count_times(df):
    df_copy = df.copy()
    count_times = df_copy.iloc[:, 0].value_counts()
    result = df_copy.iloc[:, 0].apply(lambda x: count_times[str(x)])
    return pd.DataFrame(result)


def add_cum_budget_per_production_company(df):
    df_copy = df.copy()
    cum_bpc = df_copy.groupby(by=df_copy.iloc[:, 0]).sum().reset_index()
    result = df_copy.iloc[:, 0].apply(
        lambda x: cum_bpc[cum_bpc.iloc[:, 0] == x].iloc[0, 1])
    return pd.DataFrame(result)


def add_inflation(df, column):
    df["year_2"] = df["year"].apply(lambda x: 2018 if x > 2018 else x)
    df[column] = df.apply(lambda x: cpi.inflate(x[column], x["year_2"]),
                          axis=1)
    df.drop(columns='year_2', inplace=True)
    return df

def one_hot_encode(df, column_names):
    '''
    for cell with multiple categories, one hot encode a list of column, for each categories
    for cell with multiple categories, one hot encode for each column, each categories

    Parameters
    ----------
    columns_name : array str
        name list of the columns to encode
    '''

    for column_name in column_names:
        df = one_hot_encode_multiple(df, column_name)
    return df

def famour_or_not_famous(df):
    if (df['budget'] >= 100_000_000) & (df['ratio'] >= 3):
        return 1
    return 0

def add_success_movies_per_actors(df, path = "raw_data/cat_acteur.csv"):
    '''
    Function that count how many success movies an actor did in his timeline. Add weight in function of the times
    '''
    acteurs_df = pd.read_csv(path)
    acteurs_df['ratio'] = acteurs_df['income']/acteurs_df['budget']
    acteurs_df.sort_values('year', ascending=True, inplace= True)
    acteurs_df['connu'] = acteurs_df.apply(famour_or_not_famous, axis = 1)
    acteurs_df['nbsucces'] = acteurs_df['connu']
    new_df = acteurs_df
    new_df['nbsuccess'] = acteurs_df.groupby(by  ='acteur_name')['connu'].cumsum(axis = 0)
    new_df = new_df.sort_values('year', ascending=True)
    new_df.drop(columns='nbsucces', inplace = True)
    new_df['connu2'] = new_df['nbsuccess'].apply(lambda x : 1 if x >=1 else 0)
    new_df.drop(columns="connu", inplace = True)
    new_df.rename(columns={"connu2" : "connu"}, inplace=True)
    new_df['totalsuccess'] = new_df.groupby(by = 'title').cumsum()['shifted']
    total_success = pd.DataFrame(new_df.groupby(['title'], sort = False)['shifted'].max())
    total_success.reset_index(inplace = True)
    df = df.merge(right=total_success, on=, how = "right")
    return df


def Add_Ones(df):
    df['Ones'] = 1
    return df

def Remove_Ones(df):
    df.drop(columns = "Ones", inplace = True)
    return df

def Add_number_of_movies_per_prod_company_in_Timeline(df):
    df['Nb_actuals_movie_directors_company'] = df.groupby(by = "production_company").cumsum()['Ones']
    return df


def Add_number_of_movies_per_directors_in_Timeline(df):
    df['Nb_actuals_movie_directors'] = df.groupby(by = "director").cumsum()['Ones']
    return df


def Add_number_of_movies_per_writer_in_Timeline(df):
    df['Nb_actuals_movie_directors_writer'] = df.groupby(by = "writer").cumsum()['Ones']
    return df




def example():

    print('----- import Data -----')
    df = import_data(path = 'raw_data/IMDb movies.csv')

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
    df = convert_budget_column(df,
                               column_name='budget',
                               min_rows=45,
                               out_currency='USD')

    print('----- filter categories -----')
    df = filter_categories(df, "actors", nb=2)

    print('----- convert income column -----')
    df = convert_income(df, column_name='worlwide_gross_income')

    print('----- one hot encode -----')
    #df = one_hot_encode(df, column_names=['director','genre'])
    df = one_hot_encode(df, column_names=['genre'])
    #df = one_hot_encode(df, column_names=['director'])

    print('----- convert to int -----')
    df = convert_to_int(df, 'year')
    df = convert_to_int(df, 'duration')

    print('----- convert to date -----')
    df = convert_to_date(df, 'date_published')

    print('----- seasonality Sin/Cos -----')
    df = add_sin_cos_features(df, 'Month_published')

    print('----- categorize production company -----')
    df = add_prod_company_category(df, "production_company",
                                   "production_weight")

    print('----- categorize director -----')
    df = add_director_category(df, 'director', 'cat_director')

    print('----- categorize writer -----')
    df = add_writer_category(df, "production_company", "production_weight")


    print('----- data_shape -----')
    print(df.shape)
    return df

def feature_example(df):
    print('----- add_sin_features -----')
    df["sin"] = add_sin_features(df[['date_published']])

    print('----- add_sin_features -----')
    df["cos"] = add_cos_features(df[['date_published']])

    print('----- add_sin_cos_features -----')
    df[["sin2","cos2"]] = add_sin_cos_features(df[['date_published']])

    print('----- add_director_category -----')
    df["director_cat"] = add_director_category(df[["director"]])

    print('----- add_prod_company_category -----')
    df["production_cat"] = add_prod_company_category(df[["production_company"]])

    print('----- add_writer_category -----')
    df["writer_cat"] = add_writer_category(df[["writer"]])

    print('----- prod_count_times -----')
    df["production_count"] = prod_count_times(df[["production_company"]])

    print('----- writer_count_times -----')
    df["writer_count"] = writer_count_times(df[["writer"]])

    print('----- director_count_times -----')
    df["director_count"] = director_count_times(df[["director"]])

    print('----- add_cum_budget_per_production_company -----')
    df["cum_budget_prod"] = add_cum_budget_per_production_company(
        df[["production_company","budget"]])

    #print('----- add_inflation_budget -----')
    #df["inflation_budget"] = add_inflation_budget(df[["budget"]])

    print('----- add column inflated-----')
    df = add_inflation(df, "budget")

    print('----- add income inflated-----')
    df = add_inflation(df, "worlwide_gross_income")
    # print('----- one_hot_encode -----')
    # df["production"] = one_hot_encode(df["production_company"])

    # print('----- famour_or_not_famous -----')
    # df["production"] = famour_or_not_famous(df["production_company"])

    # print('----- add_success_movies_per_actors -----')
    # df["production"] = add_success_movies_per_actors(df["production_company"])

    # print('----- add_prod_company_category -----')
    # df["production"] = add_prod_company_category(df["production_company"])

    return df


if __name__ == "__main__":
    print("\n----  PREPROCESSING -----\n")
    df = preprocess_example(path='raw_data/IMDb movies.csv')

    print("\n----  FEATURING -----\n")
    df = feature_example(df)
