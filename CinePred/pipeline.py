# IMPORT
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from currency_converter import CurrencyConverter
from CinePred.data.utils import convert_budget_column, convert_to_int, \
    add_sin_features, add_cos_features
from CinePred.data.data import Data
from CinePred.data.genre_ohe import GenreOHE
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV



def import_clean_df():
    # IMPORT DF
    data = Data('raw_data/IMDb movies.csv')
    data.import_data()

    # CLEANING
    print(data.dataframe.shape)
    data.remove_na_rows()
    print(data.dataframe.shape)
    data.convert_income(column_name='worlwide_gross_income')
    data.convert_to_date(column_name='date_published')
    data.dataframe.sort_values(by='date_published', inplace=True)
    data.dataframe.reset_index(inplace=True)

    return data


def create_pipeline():
    # PIPELINE
    sin_transformer = FunctionTransformer(add_sin_features)
    cos_transformer = FunctionTransformer(add_cos_features)

    int_transformer = FunctionTransformer(convert_to_int)
    time_pipeline = make_pipeline(int_transformer, RobustScaler())

    budget_transformer = FunctionTransformer(convert_budget_column)
    genre_transformer = make_pipeline(GenreOHE())

    preproc_basic = make_column_transformer(
        (time_pipeline, ['year', 'duration']),
        (budget_transformer, ['budget']),
        (sin_transformer, ['date_published']),
        (cos_transformer, ['date_published']),
        (genre_transformer, ['genre']))

    pipeline = make_pipeline(preproc_basic, GradientBoostingRegressor())

    return pipeline


# FIT & PREDICT
def baseline(pipeline, X, y):
    """ Returns a list of 5 mae scores"""
    mae = []
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mae.append(mean_absolute_error(y_test, y_pred))
    print("MAE: ", mae[-1])
    return mae


def get_best_params(pipeline):
    # Inspect all pipe components parameters to find the one you want to gridsearch
    # Instanciate grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid={
            # Access any component of the pipeline, as far back as you want
            'gradientboostingregressor__learning_rate': [0.001, 0.01,
                                                         0.1], # 0.1
            'gradientboostingregressor__n_estimators': [10, 100, 200, 500], # 200
            'gradientboostingregressor__max_depth': [2, 3, 4] # 2
        },
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_absolute_error")

    grid_search.fit(X, y)
    return grid_search


if __name__=='__main__':
    # DECLARE X & Y
    data = import_clean_df()
    X = data.dataframe[[
        'budget', 'genre', 'duration', 'year', 'date_published',
        'production_company'
    ]]
    y = data.dataframe['worlwide_gross_income']
    y = np.log(y) / np.log(10)

    pipeline = create_pipeline()
    #mae = baseline(pipeline, X, y)

    #grid_search = get_best_params(pipeline)
    #print("Best params for GBD: ", grid_search.best_params_)
