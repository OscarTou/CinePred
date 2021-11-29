from sklearn.pipeline import _name_estimators
from CinePred.data.importing import *
from CinePred.data.preprocessing import *
from CinePred.data.featuring import *
from CinePred.data.transformers import GenreOHE

from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBRegressor
import numpy as np
from joblib import dump, load

def preproc(df, path = "../raw_data/cat_acteur.csv"):
    '''
        Clean the dataframe

        Input: dataframe
        Output: dataframe cleaned and sorted by budget
    '''
    print("preproc")
    # NA & columns:
    df = add_success_movies_per_actors(df, path = path)

    df = keep_columns(df,
                      column_names=[
                          'year', 'date_published', 'genre', 'duration',
                          'budget', 'worlwide_gross_income',
                          'production_company', 'director', 'writer', 'shifted'
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
    df = df[df['budget'] != 0]
    df = add_inflation(df, 'budget')
    df['budget'] = log_transformation(df[['budget']])

    # income
    df['worlwide_gross_income'] = convert_income(df[['worlwide_gross_income']])
    #df = add_inflation(df, 'worlwide_gross_income')
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
    df.drop(columns=['index', 'production_company', 'director', 'writer'],
            inplace=True)

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
    params = get_best_params(model, X, y)
    print(params)
    #return cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')

def predict(df):
    '''
        Input: a preprocessed df
        Output: 2 scores MAE scores '''
    mid = int(df.shape[0] / 2)
    df1 = df.iloc[:mid].copy()
    df2 = df.iloc[mid:].copy()

    score1 = np.mean(np.abs(get_mae(df1))) # {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 200}
    score2 = np.mean(np.abs(get_mae(df2))) # {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 300}
    return [round(score1, 2), round(score2, 2)]

def predict2(df):
    ''' Get the mae on the full dataset (crossvalidated) '''

    score1 = np.abs(get_mae(
        df))  # {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 300}
    print(score1)
    return(np.mean(np.round(score1, 2)))


def predict_fromX(model, df):
    prediction = model.predict(df)
    return 10 ** prediction[0]

def save_model(fitted_model, file_name="model.joblib"):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, 'models/'+file_name)
    dump(fitted_model, filepath)

def load_model(file_name="model.joblib"):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, 'models/'+file_name)
    return load(filepath)

def get_fitted_model(df):
    X = df.drop(columns=['worlwide_gross_income'])
    y = df['worlwide_gross_income']
    model = XGBRegressor(learning_rate=0.1, max_depth=2)
    model.fit(X,y)
    return model



if __name__ == '__main__':
    # Import
    df = import_data(link = '../raw_data/IMDb_movies.csv')


    # Prepare
    print("----- CLEAN DATA ------")
    df_preproc = preproc(df)

    # Predict
    print("----- PREDICT DATA ------")
    #print(predict(df_preproc))
    #print(predict2(df_preproc))

    print("----- GET FITTED MODEL ------")
    model = get_fitted_model(df_preproc)

    print("----- SAVE MODEL ------")
    save_model(model, "model.joblib")

    print("----- LOAD MODEL ------")
    model = load_model("model.joblib")

    print("----- PREDICT MODEL ------")
    prediction = predict_fromX(
        model,df_preproc.head(1).drop(columns='worlwide_gross_income'))
    print(prediction)
    #print(predict(df))
    #print(predict2(df))
    #get_mae(df)
