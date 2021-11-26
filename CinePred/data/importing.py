import pandas as pd

# --------------------------------------- #
# -------        Data init        ------- #
# --------------------------------------- #

def import_data(link):
    '''
    read the CSV file located in link
    Parameters
    ----------
    link : str
        path of the CSV file
    '''

    return pd.read_csv(link)


if __name__ == "__main__":
    print('----- import Data -----')
    df = import_data('../raw_data/IMDb movies.csv')
