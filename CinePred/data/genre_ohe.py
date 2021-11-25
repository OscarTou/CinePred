from sklearn.base import TransformerMixin, BaseEstimator

class GenreOHE(BaseEstimator, TransformerMixin):

    def __init__(self, column_name='genre'):
        self.column_name=column_name
        self.list_unique_genre = None

    def fit(self, X, y=None):
        # separate all genres into one list, considering comma + space as separators
        genre = X[self.column_name].str.split(', ').tolist()

        # flatten the list
        flat_genre = [item for sublist in genre for item in sublist]

        # convert to a set to make unique
        set_genre = set(flat_genre)

        # back to list
        self.list_unique_genre = list(set_genre)

        return self

    def transform(self, X):
        # create columns by each unique genre
        data = X.reindex(X.columns.tolist() + self.list_unique_genre,
                         axis=1,
                         fill_value=0)

        # for each value inside column, update the dummy
        for index, row in data.iterrows():
            for val in row[self.column_name].split(', '):
                if val != 'NA':
                    data.loc[index, val] = 1

        data = data.drop(self.column_name, axis=1)
        return data
