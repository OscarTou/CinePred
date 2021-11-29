import pandas as pd
import os

# --------------------------------------- #
# -------        Data init        ------- #
# --------------------------------------- #

def import_data(link = "raw_data/IMDb_movies.csv"):
    '''
    read the CSV file located in link
    Parameters
    ----------
    link : str
        path of the CSV file
    '''

    return pd.read_csv(link, low_memory=False)


if __name__ == "__main__":
    print('----- import Data -----')
    df = import_data()
