#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from os import link
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from CinePred.data.importing import import_data
# from CinePred.data.preprocessing import *
from CinePred.new_model import load_model, predict_fromX, preproc

import pandas as pd
import numpy as np
import json
from pydantic import BaseModel

# from deep_translator import GoogleTranslator
df = import_data()
df_preproc = preproc(df)

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
        movie_dic[df_preproc['title'].iloc[0:df_preproc.shape[0]][i]] = {'Actors' : df_preproc['actors'].iloc[0:df_preproc.shape[0]][i],
                                                        'Country' : df_preproc['country'].iloc[0:df_preproc.shape[0]][i],
                                                        'imdb_title_id' : df_preproc['imdb_title_id'].iloc[0:df_preproc.shape[0]][i],
                                                        'Income' : np.round((10**(df_preproc['worlwide_gross_income'].iloc[0:df_preproc.shape[0]][i])),2)}

        # en dehors du form movie_dic['title'] = movie_dic.keys
    movie_dic['Movie title'] = movie_dic
    return movie_dic[title]

# # Il monello
@app.get("/movies")
def movies():
    movie_dic = {}
    for i in range(7000,8000):

        movie_dic[df_preproc['title'].iloc[0:df_preproc.shape[0]][i]] = {'Actors' : df_preproc['actors'].iloc[0:df_preproc.shape[0]][i],
                                                         'Country' : df_preproc['country'].iloc[0:df_preproc.shape[0]][i],
                                                         'imdb_title_id' : df_preproc['imdb_title_id'].iloc[0:df_preproc.shape[0]][i],
                                                         'Income' : np.round(10**(df_preproc['worlwide_gross_income'].iloc[0:df_preproc.shape[0]][i]),2)
                                                        #  'Numbers of blockbuster' : df['shifted'].iloc[0:df.shape[0]][i]
                                                         }

        # en dehors du form movie_dic['title'] = movie_dic.keys
    # movie_dic['Movie title'] = movie_dic
    return movie_dic

@app.get("/search_actor")
def search_actors(name):
    movie_list = df_preproc[df_preproc
                            ['actors'].str.contains(name)]['title'].values.tolist()
    actor_dict =  {}
    list_index = np.arange(0,len(movie_list)).tolist()

    for movie,indexs in zip(movie_list,list_index):
        tmpy_dic = {}
        tmpy_dic = {indexs : movie}
        actor_dict.update(tmpy_dic)

    return actor_dict

@app.get("/predict")
def prediction():
    # model = load_model("../models/model.joblib")
    # X = df_clean.head(1).drop(columns='worlwide_gross_income')
    # return predict_fromX(model,X)
    # Prepare
    #
    print("----- CLEAN DATA ------")


    print("----- LOAD MODEL ------")
    model = load_model("model.joblib")

    print("----- PREDICT MODEL ------")
    prediction = predict_fromX(
        model,df_preproc.head(1).drop(columns=['imdb_title_id','actors','description','avg_vote','country','title','worlwide_gross_income']))
    return prediction

# 'year', 'date_published', 'genre', 'duration','budget','production_company', 'director', 'writer', 'shifted'

@app.get("/test")
def test(director,
         year,
         main_actor,
         second_actor,
         third_actor,
         writer,
         production_company,
         date_published,
         genre,
         duration,
         budget,
         title=''):

    #----   Init Dataframe ----#
    df = pd.DataFrame({'year': [year]})
    df['year'] = year
    df['duration'] = duration
    df['budget'] = budget * 1000000
    df['genre'] = genre
    df['director'] = director
    df['writer'] = writer
    df['production_company'] = production_company
    df['actors'] = f'{main_actor}, {second_actor}, {third_actor}'
    df['date_published'] = date_published

    #----   preproc   ----#
    df['year'] = convert_to_int(df[['year']])
    df['duration'] = convert_to_int(df[['duration']])
    df['budget'] = log_transformation(df[['budget']])

    actors_1 = df_preproc[['shifted', 'actors']][df_preproc[[
        'shifted', 'actors'
    ]]['actors'].str.contains(main_actor)].max()['shifted']
    actors_2 = df_preproc[['shifted', 'actors']][df_preproc[[
        'shifted', 'actors'
    ]]['actors'].str.contains(second_actor)].max()['shifted']
    actors_3 = df_preproc[['shifted', 'actors']][df_preproc[[
        'shifted', 'actors'
    ]]['actors'].str.contains(third_actor)].max()['shifted']
    shifted = actors_1 + actors_2 + actors_3
    df['shifted'] = shifted

    df['date_published'] = convert_to_date(df[['date_published']])
    df['date_sin'] = add_sin_features(df[['date_published']])
    df['date_cos'] = add_cos_features(df[['date_published']])
    df.drop(columns='date_published', inplace=True)

    ohe = GenreOHE()
    ohe.fit(df)  # la colonne 'genre' est spécifié dans la classe
    df = ohe.transform(df)

    df['Nb_actuals_movie_directors_company'] = df_preproc[[
        'Nb_actuals_movie_directors_company', 'production_company'
    ]][df_preproc[[
        'Nb_actuals_movie_directors_company', 'production_company'
    ]]['production_company'].str.contains(
        production_company)].max()['Nb_actuals_movie_directors_company']
    df['Nb_actuals_movie_directors'] = df_preproc[[
        'Nb_actuals_movie_directors', 'director'
    ]][df_preproc[[
        'Nb_actuals_movie_directors', 'director'
    ]]['director'].str.contains(director)].max()['Nb_actuals_movie_directors']
    df['Nb_actuals_movie_directors_writer'] = df_preproc[[
        'Nb_actuals_movie_directors_writer', 'writer'
    ]][df_preproc[['Nb_actuals_movie_directors_writer',
                   'writer']]['writer'].str.contains(
                       writer)].max()['Nb_actuals_movie_directors_writer']

    df = df.drop(columns=['actors'])

    df = df.drop(columns=['production_company', 'director', 'writer'])
    #----   Prediction   ----#
    model = load_model("model.joblib")
    result = predict_fromX(model, df)

    return result
