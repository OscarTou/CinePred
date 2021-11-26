from CinePred.data.importing import *
from CinePred.data.preprocessing import *
from CinePred.data.featuring import *
from CinePred.data.transformers import *

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer, RobustScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from currency_converter import CurrencyConverter
from xgboost import XGBRegressor


def import_clean_df():
    # IMPORT DF
    df = import_data('raw_data/IMDb movies.csv')

    # CLEANING
    df = keep_columns(df,column_names=[
        'imdb_title_id', 'title', 'year', 'date_published', 'genre',
        'duration', 'country', 'director', 'writer', 'production_company',
        'actors', 'budget', 'worlwide_gross_income'
    ])
    df = remove_na_rows(df)
    df['worlwide_gross_income'] = convert_income(df[['worlwide_gross_income']])
    df['date_published'] = convert_to_date(df[['date_published']])
    df = df.sort_values(by='date_published')
    df = df.reset_index()

    return df


def create_pipeline():
    # PIPELINE
    sin_transformer = FunctionTransformer(add_sin_features)
    cos_transformer = FunctionTransformer(add_cos_features)

    int_transformer = FunctionTransformer(convert_to_int)
    time_pipeline = make_pipeline(int_transformer, RobustScaler())

    budget_transformer = FunctionTransformer(convert_budget_column)
    genre_transformer = make_pipeline(GenreOHE())
    prod_transformer = FunctionTransformer(prod_count_times)
    writer_transformer = FunctionTransformer(writer_count_times)
    director_transformer = FunctionTransformer(director_count_times)


    preproc_basic = make_column_transformer(
        (time_pipeline, ['year', 'duration']),
        (budget_transformer, ['budget']),
        (sin_transformer, ['date_published']),
        (cos_transformer, ['date_published']),
        (genre_transformer, ['genre']),
        (prod_transformer, ['production_company']),
        (writer_transformer, ['writer']),
        (director_transformer, ['director']),
    )

    pipeline = make_pipeline(
        preproc_basic,
        XGBRegressor(max_depth=10, n_estimators=100,
                     learning_rate=0.1))  # GradientBoostingRegressor

    return pipeline


# FIT & PREDICT
def fit_and_score(pipeline, X, y):
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
    df = import_clean_df()
    X = df[[
        'budget', 'genre', 'duration', 'year', 'date_published',
        'production_company', 'writer', 'director'
    ]]
    y = df['worlwide_gross_income']
    y = np.log(y) / np.log(10)

    pipeline = create_pipeline()
    mae = fit_and_score(pipeline, X, y)

    #grid_search = get_best_params(pipeline)
    #print("Best params for GBD: ", grid_search.best_params_)
