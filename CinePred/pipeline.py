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


def import_clean_df(path = "raw_data/IMDb movies.csv"):
    # IMPORT DF
    df = import_data(path)

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
        (sin_transformer, ['date_published']),
        (cos_transformer, ['date_published']),
        (budget_transformer, ['budget']),
        (genre_transformer, ['genre']))


    # without (sin_transformer, ['date_published']),
    #  and    (cos_transformer, ['date_published']), # 0.745
    # without: (time_pipeline, ['year', 'duration']), # 0.733 (!)
    # 0.742
    # with (prod_transformer, ['production_company']), # 0.736
    # with (director_transformer, ['director']), # 0.732
    # with (int_transformer, ['shifted']) # 0.742
    # with (writer_transformer, ['writer'])) # 0.738
    return preproc_basic


# FIT & PREDICT
def fit_and_score(pipeline, X, y):
    """ Returns a list of 5 mae scores"""
    print(X.columns)
    mae = []
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Transform train&test
        X_train_preproc = pipeline.fit_transform(X_train)
        X_test_preproc = pipeline.transform(X_test)

        # Split train into train/split (to get best_iteration):
        X_train2_preproc = X_train_preproc[:-400]
        X_val2_preproc = X_train_preproc[-400:]
        y_train2_preproc = y_train[:-400]
        y_val2_preproc = y_train[-400:]

        model_xgb = XGBRegressor(max_depth=5, n_estimators=1000, learning_rate=0.1)
        model_xgb.fit(X_train2_preproc,
                      y_train2_preproc,
                      verbose=False,
                      eval_set=[(X_train2_preproc, y_train2_preproc),
                                (X_val2_preproc, y_val2_preproc)],
                      eval_metric=["mae"],
                      early_stopping_rounds=5)

        best_iter = model_xgb.best_iteration

        # Re-fit our XGBr with best_iter
        model_xgb2 = XGBRegressor(max_depth=5,
                                  n_estimators=best_iter,
                                  learning_rate=0.1)
        model_xgb2.fit(X_train_preproc, y_train)

        # Prediction
        y_pred = model_xgb2.predict(X_test_preproc)

        # Score
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
            'gradientboostingregressor__learning_rate': [0.001, 0.01, 0.1],
            'gradientboostingregressor__n_estimators': [10, 100, 200, 500],
            'gradientboostingregressor__max_depth': [2, 3, 4]
        },
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_absolute_error")

    grid_search.fit(X, y)
    return grid_search


if __name__=='__main__':
    # DECLARE X & Y
    df = import_clean_df()
    df = add_success_movies_per_actors(df)
    X = df[[
        'budget', 'genre', 'duration', 'year', 'date_published',
        'production_company', 'writer', 'director', 'shifted'
    ]]
    y = df['worlwide_gross_income']
    y = np.log(y) / np.log(10)

    pipeline = create_pipeline()
    mae = fit_and_score(pipeline, X, y)

    #grid_search = get_best_params(pipeline)
    #print("Best params for GBD: ", grid_search.best_params_)
