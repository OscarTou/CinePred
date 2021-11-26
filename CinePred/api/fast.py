from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from CinePred.data.importing import *
from CinePred.data.preprocessing import *
import numpy as np

df = preprocess_example("raw_data/IMDb movies.csv")

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


@app.get("/actors")
def actors():
    movie_dic = {}
    for i in range(5000):

        movie_dic[df['title'].iloc[0:df.shape[0]][i]] = {'Actors' : df['actors'].iloc[0:df.shape[0]][i],
                                                            'Country' : df['country'].iloc[0:df.shape[0]][i],
                                                            'imdb_title_id' : df['imdb_title_id'].iloc[0:df.shape[0]][i],
                                                         'Income' : np.round(10**(df['worlwide_gross_income'].iloc[0:df.shape[0]][i]),2),
                                                        #  'Numbers of blockbuster' : df['shifted'].iloc[0:df.shape[0]][i]
                                                         }

        # en dehors du form movie_dic['title'] = movie_dic.keys

    return movie_dic
