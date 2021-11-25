from CinePred.params import *
from CinePred.pipeline import import_clean_df, fit_and_score


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

    pipeline = make_pipeline(preproc_basic, LinearRegression())

    return pipeline



if __name__ == '__main__':
    # DECLARE X & Y
    data = import_clean_df()
    X = data.dataframe[[
        'budget', 'genre', 'duration', 'year', 'date_published',
    ]]
    y = data.dataframe['worlwide_gross_income']
    y = np.log(y) / np.log(10)

    pipeline = create_pipeline()
    mae = fit_and_score(pipeline, X, y)
