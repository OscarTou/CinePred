# from os import link
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from CinePred.data.importing import import_data
from CinePred.data.preprocessing import *
# from CinePred.new_model import preproc

import pandas as pd
import numpy as np
# from deep_translator import GoogleTranslator

df = import_data()
df = preprocess_example(path='raw_data/IMDb_movies.csv')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"App Name": "CinePred"}


@app.get("/search_movie")
def search_movie(title):
    movie_dic = {}
    for i in range(7000,8000):

        movie_dic[df['title'].iloc[0:df.shape[0]][i]] = {'Actors' : df['actors'].iloc[0:df.shape[0]][i],
                                                         'Country' : df['country'].iloc[0:df.shape[0]][i],
                                                         'imdb_title_id' : df['imdb_title_id'].iloc[0:df.shape[0]][i],
                                                         'Income' : np.round(10**(df['worlwide_gross_income'].iloc[0:df.shape[0]][i]),2)
                                                        #  'Numbers of blockbuster' : df['shifted'].iloc[0:df.shape[0]][i]
                                                         }

        # en dehors du form movie_dic['title'] = movie_dic.keys
    movie_dic['Movie title'] = movie_dic
    return movie_dic[title]

# Il monello
@app.get("/movies")
def movies():
    movie_dic = {}
    for i in range(7000,8000):

        movie_dic[df['title'].iloc[0:df.shape[0]][i]] = {'Actors' : df['actors'].iloc[0:df.shape[0]][i],
                                                         'Country' : df['country'].iloc[0:df.shape[0]][i],
                                                         'imdb_title_id' : df['imdb_title_id'].iloc[0:df.shape[0]][i],
                                                         'Income' : np.round(10**(df['worlwide_gross_income'].iloc[0:df.shape[0]][i]),2)
                                                        #  'Numbers of blockbuster' : df['shifted'].iloc[0:df.shape[0]][i]
                                                         }

        # en dehors du form movie_dic['title'] = movie_dic.keys
    # movie_dic['Movie title'] = movie_dic
    return movie_dic

@app.get("/search_actor")
def search_actors(name):
    movie_list = df[df['actors'].str.contains(name)]['title'].values.tolist()
    actor_dict =  {}
    list_index = np.arange(0,len(movie_list)).tolist()

    for movie,indexs in zip(movie_list,list_index):
        tmpy_dic = {}
        tmpy_dic = {indexs : movie}
        actor_dict.update(tmpy_dic)

    return actor_dict

# @app.get("/predict")
# def prediction(X):

#     # X = []

#     return predict(X)
