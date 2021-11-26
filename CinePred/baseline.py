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

    int_transformer = FunctionTransformer(convert_to_int)
    time_pipeline = make_pipeline(int_transformer, RobustScaler())

    genre_transformer = make_pipeline(GenreOHE())
    sin_transformer = FunctionTransformer(add_sin_features)
    cos_transformer = FunctionTransformer(add_cos_features)

    preproc_basic = make_column_transformer(
        (time_pipeline, ['year', 'duration']),
        (budget_transformer, ['budget']),
        (sin_transformer, ['date_published']),
        (cos_transformer, ['date_published']),
        (genre_transformer, ['genre'])
        )

    pipeline = make_pipeline(preproc_basic, LinearRegression())

    return pipeline

def cross_val(pipeline, X, y):
    cv = cross_val_score(pipeline, X, y, cv=TimeSeriesSplit(n_splits=5))
    print(cv)


if __name__ == '__main__':

    # Init DataFrame
    print("---- Init Data ----")
    df = import_data('raw_data/IMDb movies.csv')
    df = keep_columns(df,column_names=[
        'budget',
        'duration',
        'genre',
        'year',
        'date_published',
        'worlwide_gross_income'])

    # Cleaning DataFrame
    print("---- Cleaning Data ----")
    df = remove_na_rows(df)
    df['date_published'] = convert_to_date(df[['date_published']])
    df['worlwide_gross_income'] = convert_income(df[['worlwide_gross_income']])
    df['worlwide_gross_income'] = log_transformation(df[['worlwide_gross_income']])
    df.sort_values(by='date_published', inplace=True)
    df = reset_index(df)

    # X and Y Creation
    print("---- X and Y Creation ----")
    X = df[[
        'budget',
        'duration',
        'genre',
        'year',
        'date_published',
    ]]
    y = df['worlwide_gross_income']

    # Pipeline and fit
    print("---- Pipeline Creation ----")
    pipeline = create_baseline_pipeline()
    print("Pipeline created")
    #pipeline.fit(X)
    cross_validate(pipeline, X, y)
    print("Cross val done")
    #cross_val(pipeline, X, y)
