from sklearn.pipeline import _name_estimators
from CinePred.data.importing import *
from CinePred.data.preprocessing import *
from CinePred.data.featuring import *
from CinePred.data.transformers import GenreOHE

from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBRegressor
import numpy as np

def preproc(df):
    '''
        Clean the dataframe

        Input: dataframe
        Output: dataframe cleaned and sorted by budget
    '''
    # NA & columns:
    df = keep_columns(df,
                      column_names=[
                          'year', 'date_published', 'genre', 'duration',
                          'budget', 'worlwide_gross_income'
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
    df['budget'] = log_transformation(df[['budget']])

    # income
    df['worlwide_gross_income'] = convert_income(df[['worlwide_gross_income']])
    df['worlwide_gross_income'] = log_transformation(
        df[['worlwide_gross_income']])

    # sort & index:
    df.sort_values('budget', inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns='index', inplace=True)

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
    mid = int(df.shape[0] / 2)
    df1 = df.iloc[:mid].copy()
    df2 = df.iloc[mid:].copy()

    score1 = np.mean(np.abs(get_mae(df1))) # {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 200}
    score2 = np.mean(np.abs(get_mae(df2))) # {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 300}
    #return [score1, score2]
    return [round(score1, 2), round(score2, 2)]

if __name__ == '__main__':
    # Import
    df = import_data('raw_data/IMDb movies.csv')

    # Prepare
    df = preproc(df)

    # Predict
    print(predict(df))
