from CinePred.data.importing import *
from CinePred.data.preprocessing import *
from CinePred.data.featuring import *
from CinePred.data.transformers import *
from CinePred.pipeline import *

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_validate, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer, RobustScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from currency_converter import CurrencyConverter
from xgboost import XGBRegressor

def create_baseline_pipeline():

    # PIPELINE
    budget_transformer = FunctionTransformer(convert_budget_column)
    genre_transformer = make_pipeline(GenreOHE())
    sin_transformer = FunctionTransformer(add_sin_features)
    cos_transformer = FunctionTransformer(add_cos_features)

    preproc_basic = make_column_transformer(
        (budget_transformer, ['budget']),
        (sin_transformer, ['date_published']),
        (cos_transformer, ['date_published']),
        (genre_transformer, ['genre'])
        )

    pipeline = make_pipeline(preproc_basic, LinearRegression())

    return pipeline

def cross_val(pipeline, X, y):
    cv = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(cv)
    return cv[-1]


# FIT & PREDICT
def fit_and_score(pipeline, X, y):
    """ Returns a list of 5 mae scores"""
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

        model_xgb = XGBRegressor(max_depth=5,
                                 n_estimators=1000,
                                 learning_rate=0.1)
        model_xgb.fit(X_train2_preproc,
                      y_train2_preproc,
                      verbose=False,
                      eval_set=[(X_train2_preproc, y_train2_preproc),
                                (X_val2_preproc, y_val2_preproc)],
                      eval_metric=["mae"],
                      early_stopping_rounds=5)

        best_iter = model_xgb.best_iteration

        # Re-fit our XGBr with best_iter
        model = LinearRegression()
        model.fit(X_train_preproc, y_train)

        # Prediction
        y_pred = model_xgb2.predict(X_test_preproc)

        # Score
        mae.append(mean_absolute_error(y_test, y_pred))
    print("MAE: ", mae[-1])
    return mae


if __name__ == '__main__':

    # Init DataFrame
    print("---- Init Data ----")
    df = import_data('raw_data/IMDb movies.csv')
    df = keep_columns(df,column_names=[
        'budget',
        'genre',
        'date_published',
        'worlwide_gross_income'])

    # Cleaning DataFrame
    print("---- Cleaning Data ----")
    df = remove_na_rows(df)
    df['date_published'] = convert_to_date(df[['date_published']])
    df['worlwide_gross_income'] = convert_income(df[['worlwide_gross_income']])
    df['worlwide_gross_income'] = log_transformation(df[['worlwide_gross_income']])

    # X and Y Creation
    print("---- X and Y Creation ----")
    X = df[[
        'budget',
        'genre',
        'date_published',
    ]]
    y = df['worlwide_gross_income']

    # Pipeline and fit
    print("---- Pipeline Creation ----")
    pipeline = create_baseline_pipeline()
    print(cross_val(pipeline, X, y))
