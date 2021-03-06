#!/usr/bin/env python
# -*- coding: utf-8 -*-

from CinePred.data.importing import import_data
from CinePred.data.preprocessing import *
from CinePred.data.featuring import *
from CinePred.data.transformers import GenreOHE
from google.cloud import storage

from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBRegressor, plot_importance
import numpy as np
from joblib import dump, load

def preproc(df, path="gs://wagon-data-722-cinepred/data/cat_acteur.csv"):
    '''
        Clean the dataframe

        Input: dataframe
        Output: dataframe cleaned and sorted by budget
    '''
    # NA & columns:
    df = add_success_movies_per_actors(df, path=path)
    df = add_number_of_movies_actor1_in_Timeline(df, path=path)
    df = add_number_of_movies_actor2_in_Timeline(df, path=path)
    df = add_number_of_movies_actor3_in_Timeline(df, path=path)
    df = add_total_income_of_last_movie_of_actors_in_Timeline(df, path=path)

    df = keep_columns(df,
                      column_names=[
                          'worlwide_gross_income', 'year', 'date_published',
                          'genre', 'duration', 'budget', 'production_company',
                          'director', 'writer', 'shifted', 'nb_movies_actor1',
                          'nb_movies_actor2', 'nb_movies_actor3',
                          'last income', 'imdb_title_id','actors',
                          'description','avg_vote','country','title'
                      ])
    df = remove_na_rows(df)

    # date_published
    df['date_published'] = convert_to_date(df[['date_published']])

    # day of the year
    df['date_sin'] = add_sin_features(df[['date_published']])
    df['date_cos'] = add_cos_features(df[['date_published']])
    df.drop(columns='date_published', inplace=True)

    # year, duration
    df['year'] = convert_to_int(df[['year']])
    df['duration'] = convert_to_int(df[['duration']])

    # genre
    ohe = GenreOHE()
    ohe.fit(df) # la colonne 'genre' est spécifié dans la classe
    df = ohe.transform(df)

    # budget
    df['budget'] = convert_budget_column(df[['budget']])
    df = df[df['budget'] > 100]
    df = add_inflation(df, 'budget')
    df['budget'] = log_transformation(df[['budget']])

    # income
    df['worlwide_gross_income'] = convert_income(df[['worlwide_gross_income']])
    df = add_inflation(df, 'worlwide_gross_income')
    df['worlwide_gross_income'] = log_transformation(df[['worlwide_gross_income']])

    # Cumsum
    df = Add_Ones(df)
    df = Add_number_of_movies_per_prod_company_in_Timeline(df)
    df = Add_number_of_movies_per_directors_in_Timeline(df)
    df = Add_number_of_movies_per_writer_in_Timeline(df)
    df = Remove_Ones(df)

    # sort & index:
    df.sort_values('budget', inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)

    return df

def feature_importance(df):
    X = df.drop(columns=['worlwide_gross_income'])
    y = df['worlwide_gross_income']
    model = XGBRegressor(learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    print(model.feature_importances_)
    plot_importance(model)


def preproc_demoday(df, path = "gs://wagon-data-722-cinepred/data/cat_acteur.csv"):
    # df = add_success_movies_per_actors(df, path=path)
    # df = add_number_of_movies_actor1_in_Timeline(df, path=path)
    # df = add_number_of_movies_actor2_in_Timeline(df, path=path)
    # df = add_number_of_movies_actor3_in_Timeline(df, path=path)
    # df = add_total_income_of_last_movie_of_actors_in_Timeline(df, path=path)

    # df = remove_na_rows(df)


    # df['date_published'] = convert_to_date(df[['date_published']])

    # # day of the year
    # df['date_sin'] = add_sin_features(df[['date_published']])
    # df['date_cos'] = add_cos_features(df[['date_published']])
    # df.drop(columns='date_published', inplace=True)

    # year, duration
    # df['year'] = convert_to_int(df[['year']])
    # df['duration'] = convert_to_int(df[['duration']])

    # genre
    # ohe = GenreOHE()
    # ohe.fit(df) # la colonne 'genre' est spécifié dans la classe
    # df = ohe.transform(df)

    # budget
    # df['budget'] = convert_budget_column(df[['budget']])
    # df = df[df['budget'] > 100]
    # df = add_inflation(df, 'budget')
    df['budget'] = log_transformation(df[['budget']])

    # # income
    # df['worlwide_gross_income'] = convert_income(df[['worlwide_gross_income']])
    # df = add_inflation(df, 'worlwide_gross_income')
    # df['worlwide_gross_income'] = log_transformation(df[['worlwide_gross_income']])

    # Cumsum
    # df = Add_Ones(df)
    # df = Add_number_of_movies_per_prod_company_in_Timeline(df)
    # df = Add_number_of_movies_per_directors_in_Timeline(df)
    # df = Add_number_of_movies_per_writer_in_Timeline(df)
    # df = Remove_Ones(df)

    # sort & index:
    # df.sort_values('budget', inplace=True)
    # df.reset_index(inplace=True)
    # df.drop(columns=['index'], inplace=True)

    return df



def get_best_params(model, X, y):
    # Inspect all pipe components parameters to find the one you want to gridsearch
    # Instanciate grid search
    grid_search = GridSearchCV(
        model,
        param_grid={
            # Access any component of the pipeline, as far back as you want
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200, 300],
            'max_depth': [2]
        },
        cv=5,
        scoring="neg_mean_absolute_error",
        verbose=2)

    grid_search.fit(X, y)
    return grid_search.best_params_

def get_mae(df):
    X = df.drop(columns=['worlwide_gross_income'])
    y = df['worlwide_gross_income']
    model = XGBRegressor(learning_rate=0.1, max_depth=2)

    return cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')

def predict(df):
    '''
        Input: a preprocessed df
        Output: 2 scores MAE scores '''

    mid = int(len(df) / 2)
    df1 = df.iloc[:mid].copy()
    df2 = df.iloc[mid:].copy()

    score1 = np.mean(np.abs(get_mae(df1))) # {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 200}
    score2 = np.mean(np.abs(get_mae(df2))) # {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 300}
    return [round(score1, 3), round(score2, 3)]

def predict2(df):
    ''' Get the mae on the full dataset (crossvalidated) '''

    score = np.abs(get_mae(df))  # {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 300}
    score = round(np.mean(score),2)
    return(score)

def predict_fromX(model, df):
    prediction = model.predict(df)
    return 10 ** prediction[0]

def save_model(fitted_model, file_name="model.joblib"):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, 'models/'+file_name)
    dump(fitted_model, filepath)

def load_model(file_name="model.joblib"):
    #dirname = os.path.dirname(__file__)
    #filepath = os.path.join(dirname, 'models/'+file_name)

    # filepath = 'gs://wagon-data-722-cinepred/model/' + file_name

    BUCKET_NAME = "wagon-data-722-cinepred"
    storage_filename = 'model/model.joblib'
    local_filename = "model.joblib"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_filename)
    blob.download_to_filename(local_filename)

    return load(local_filename)


def get_fitted_model(df):
    # Select only film with most income
    mid = int(len(df) / 2)
    df1 = df.iloc[:mid].copy()
    df2 = df.iloc[mid:].copy()

    # Create X and y
    X = df2.drop(columns=['worlwide_gross_income'])
    y = df2['worlwide_gross_income']
    model = XGBRegressor(learning_rate=0.1, max_depth=2)
    model.fit(X,y)

    score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f'mae = {round(np.mean(np.abs(score)),2)}')
    return model


if __name__ == '__main__':
    # Import
    print("----- IMPORT DATA ------")
    #df = import_data(link = 'gs://wagon-data-722-cinepred/data/IMDb_movies.csv')
    #df_preproc = preproc(df)
    df_preproc = import_data(link='gs://wagon-data-722-cinepred/data/preprocessed.csv')

    # Prepare
    print("----- CLEAN DATA ------")
    df_preproc = df_preproc.drop(columns=['production_company', 'director', 'writer'])
    df_preproc = df_preproc.drop(columns=['imdb_title_id','actors','description','avg_vote','country','title'])

    # Predict
    print("----- PREDICT DATA ------")
    #print(predict(df_preproc))

    print("----- GET FITTED MODEL ------")
    model = get_fitted_model(df_preproc)

    print("----- SAVE MODEL ------")
    save_model(model, "model.joblib")

    print("----- LOAD MODEL ------")
    model = load_model("model.joblib")

    print("----- PREDICT MODEL ------")
    prediction = predict_fromX(
        model,df_preproc.head(1).drop(columns=['worlwide_gross_income']))
    print(prediction)
    #print(predict(df))
    #print(predict2(df))
    #get_mae(df)
