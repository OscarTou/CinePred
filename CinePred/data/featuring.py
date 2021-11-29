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

def add_inflation(df, column):
    df["year_2"] = df["year"].apply(lambda x: 2018 if x > 2018 else x)
    df[column] = df.apply(lambda x: cpi.inflate(x[column], x["year_2"]),
                          axis=1)
    df.drop(columns='year_2', inplace=True)
    return df


def famour_or_not_famous(df):
    if (df['budget'] >= 100_000_000) & (df['ratio'] >= 3):
        return 1
    return 0

def add_success_movies_per_actors(df):
    '''
    Function that count how many success movies an actor did in his timeline. Add weight in function of the times
    '''
    df2 = pd.read_csv('../raw_data/cat_acteur.csv')
    df2['ratio'] = df2['worlwide_gross_income'] / df2['budget']
    # df2_cat = df2.loc[(df2['ratio']>=5) & (df2['budget'] >= 100_000_000)]
    # len(df2_cat['acteur_name'].unique())
    # df2_cat[['acteur_name','budget']].groupby(by='acteur_name').sum().sort_values(by="budget", ascending=False).head(159)
    # top_tier_actor_df = df2.groupby(by="acteur_name").sum()[['budget']].sort_values("budget", ascending=False)
    df2.sort_values('year', ascending=True, inplace= True)
    df2['connu'] = df2.apply(famour_or_not_famous, axis = 1)
    df2['nbsucces'] = df2['connu']
    new_df = df2
    new_df['nbsuccess'] = df2.groupby(by  ='acteur_name')['connu'].cumsum(axis = 0)
    new_df = new_df.sort_values('year', ascending=True)
    new_df.drop(columns='nbsucces', inplace = True)
    new_df['connu2'] = new_df['nbsuccess'].apply(lambda x : 1 if x >=1 else 0)
    new_df.drop(columns="connu", inplace = True)
    new_df.rename(columns={"connu2" : "connu"}, inplace=True)
    new_df['totalsuccess'] = new_df.groupby(by = 'title').cumsum()['shifted']
    total_success = pd.DataFrame(new_df.groupby(['title'], sort = False)['shifted'].max())
    total_success.reset_index(inplace = True)
    df.merge(right=total_success, on='title', how = "right")
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
    df = import_data('raw_data/IMDb movies.csv')

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
