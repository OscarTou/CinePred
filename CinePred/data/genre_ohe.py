from sklearn.base import TransformerMixin, BaseEstimator

class GenreOHE(BaseEstimator, TransformerMixin):
    '''
    Specific Transformer for Multiple Categories
    Multiple One Hot Encoder
    '''
    def __init__(self, column_name='genre', format_separator=', '):
        '''
        init function for MOHE transformer

        Parameters
        ----------
        column_name : str
            name of the column to MOHE

        format_separator : str
            separator format for the split
            default : ', '
        '''
        self.column_name=column_name
        self.format_separator = format_separator
        self.list_unique_categories = None

    def fit(self, X, y=None):
        '''
        fit function that create the list of unique categories

        Parameters
        ----------
        X : input dataframe
            imput of the column to fit
        '''
        # separate all categories into one list, considering comma + space as separators
        categories = X[self.column_name].str.split(self.format_separator).tolist()

        # flatten the list
        flat_categories = [item for sublist in categories for item in sublist]

        # convert to a set to make unique
        set_categories = set(flat_categories)

        # back to list
        self.list_unique_categories = list(set_categories)
        return self

        return self

    def transform(self, X):
        '''
        transform function that create columns for each categories and OHE each rows

        Parameters
        ----------
        X : input dataframe
            imput of the column to transform
        '''
        # create columns by each unique categories
        data = X.reindex(X.columns.tolist() + self.list_unique_categories,
                         axis=1,
                         fill_value=0)

        # for each value inside column, update the dummy
        for index, row in data.iterrows():
            for val in row[self.column_name].split(', '):
                if val != 'NA':
                    data.loc[index, val] = 1

        data = data.drop(self.column_name, axis=1)
        return data
