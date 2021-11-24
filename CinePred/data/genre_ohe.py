from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class GenreOHE(OneHotEncoder):
    def transform(self, *args, **kwargs):
        return pd.DataFrame(super().transform(*args, **kwargs),
                            columns=self.get_feature_names())


def one_hot_encode(self, column_names):
    '''
        for cell with multiple categories, one hot encode for each column, each categories

        Parameters
        ----------
        columns_name : array str
            name list of the columns to encode
        '''
    for column_name in column_names:
        self.dataframe = one_hot_encode_multiple(self.dataframe, column_name)
    return self
