#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import os

# --------------------------------------- #
# -------        Data init        ------- #
# --------------------------------------- #

def import_data(link = "gs://wagon-data-722-cinepred/data/IMDb_movies.csv"):
    '''
    read the CSV file located in link
    Parameters
    ----------
    link : str
        path of the CSV file
    '''

    return pd.read_csv(link, low_memory=False, encoding = "utf8" )


if __name__ == "__main__":
    print('----- import Data -----')
    df = import_data()
