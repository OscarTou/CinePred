from CinePred.data.importing import *
from CinePred.data.preprocessing import *
from CinePred.data.featuring import *
from CinePred.data.transformers import *

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_validate, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer, RobustScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

def preproc(df):
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

    # budget
    df['budget'] = convert_budget_column(df[['budget']])
    #df['budget'] = log_transformation(df[['budget']])

    # income
    df['worlwide_gross_income'] = convert_income(df[['worlwide_gross_income']])
    df['worlwide_gross_income'] = log_transformation(
        df[['worlwide_gross_income']])

    return df

def get_mae(df):
    X = df.drop(columns=['worlwide_gross_income'])
    y = df['worlwide_gross_income']
    model = XGBRegressor()
    return cross_val_score(model, X, y, cv=5)


if __name__ == '__main__':
    # DECLARE X & Y
    df = import_data('raw_data/IMDb movies.csv')
    ohe = GenreOHE()
    df = preproc(df)
    ohe.fit(df)
    df = ohe.transform(df)

    df.sort_values('budget', inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns='index', inplace=True)

    mid = int(df.shape[0] / 2)
    df1 = df.iloc[:mid].copy()
    df2 = df.iloc[mid:].copy()

    print(get_mae(df1))
    print(get_mae(df2))
