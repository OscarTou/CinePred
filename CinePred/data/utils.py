from CinePred.params import *


def convert(value, in_currency, out_currency, converter=None):
    if converter is None :
        converter = CurrencyConverter()
    if in_currency == '$':
        in_currency = 'USD'
    elif in_currency == 'DEM' :
        return value*0.576
    elif in_currency == 'FRF' :
        return value*0.172

    try:
        result = converter.convert(value, in_currency, out_currency)
    except:
        return value # Unrecognized currency: Not converted !
    return result

def one_hot_encode_multiple (data,column_name, remove_column = True):
    # separate all genres into one list, considering comma + space as separators
    genre = data[column_name].str.split(', ').tolist()

    # flatten the list
    flat_genre = [item for sublist in genre for item in sublist]

    # convert to a set to make unique
    set_genre = set(flat_genre)

    # back to list
    unique_genre = list(set_genre)

    # create columns by each unique genre
    data = data.reindex(data.columns.tolist() + unique_genre, axis=1, fill_value=0)

    # for each value inside column, update the dummy
    for index, row in data.iterrows():
        for val in row[column_name].split(', '):
            if val != 'NA':
                data.loc[index, val] = 1

    data.drop(column_name, axis = 1, inplace = True)
    return data

def reduce_column_type(data, column_name, nb_max=5):
    # separate all actors into lists
    actor_list = data[column_name].str.split(', ').tolist()
    #return top 5 actors
    data[column_name] = [','.join(actor[:nb_max]) for actor in actor_list]
    return data

# Functions from the class Data:
def convert_to_int(df):
    '''
        convert column to integer

        Parameters
        ----------
        column_name : str
            name of the column to convert
        '''
    df = df.astype(str).astype(int)
    return df

def convert_budget_column(df, column_name='budget', out_currency='USD'):
    '''
        convert budget column in USD value converted in int
    '''
    # supprime les espaces à la fin et au début
    df[column_name] = df[column_name].str.strip()

    # split la string en mots
    df[column_name] = df[column_name].str.split()

    # split in two columns
    df['currency'] = df[column_name].apply(lambda x: x[0])
    df[column_name] = df[column_name].apply(lambda x: x[1]).astype('int64')

    c = CurrencyConverter()
    df[column_name] = df[[column_name,'currency']]\
        .apply(
        lambda x: convert(x[column_name], x['currency'], 'USD', converter = c), axis=1)
    df = df.drop(columns='currency')
    return df

def convert_to_date(df, date_format='%Y-%m-%d'):
    '''
    convert column to datetime

    Parameters
    ----------
    column_name : str
        name of the column to convert

    date_format : str , default '%Y-%m-%d'
        format of the dates in the column
    '''
    df = df.apply(lambda x: pd.to_datetime(x, format=date_format))
    return df


def add_sin_features(df):
    '''
    seasonality: add sin column for each month
    '''
    months = df.apply(lambda x: pd.DatetimeIndex(x).month)
    result =  np.sin(2 * np.pi * months / 12)
    return result

def add_cos_features(df):
    '''
    seasonality: add cos column for each month
    '''
    months = df.apply(lambda x: pd.DatetimeIndex(x).month)
    return np.cos(2 * np.pi * months / 12)

def add_director_category(df):
    '''
    Categroize director in 3 categories ranging from 1 to 3
    '''
    prod = pd.cut(df.value_counts(),
                  bins=[0, 2, 10, 50],
                  include_lowest=True,
                  labels=[1, 2, 3])
    return df.apply(lambda x: prod[str(x)])

def add_prod_company_category(df):
    '''
    Categorize production company in 5 categories ranging from 1 to 5
    '''
    prod = pd.cut(df.value_counts(),
                    bins=[0, 1, 5, 20, 50, 500],
                    include_lowest=True,
                    labels=[1, 2, 3, 4, 5])
    return df.apply(lambda x: prod[str(x)])


def add_writer_category(df):
    '''
    Categorize writer in 5 categories ranging from 1 to 3
    '''
    prod = pd.cut(df.value_counts(),
                  bins=[0, 1, 5, 40],
                  include_lowest=True,
                  labels=[1, 2, 3])
    return df.apply(lambda x: prod[str(x)])


def prod_count_times(data):
    df = data.copy()
    count_times = df['production_company'].value_counts()
    df['production_company'] = df['production_company'].apply(
        lambda x: count_times[str(x)])
    return df

def writer_count_times(data):
    df = data.copy()
    count_times = df['writer'].value_counts()
    df['writer'] = df['writer'].apply(lambda x: count_times[str(x)])
    return df

def director_count_times(data):
    df = data.copy()
    count_times = df['director'].value_counts()
    df['director'] = df['director'].apply(lambda x: count_times[str(x)])
    return df

def add_actor_nbmovie(df):
    #TODO Ruben
    pass

def log_transformation(df):
    df = np.log(df)
    return df

def add_cum_budget_per_production_company(prod_comp, budget):
    cum_bpc = budget.groupby(by=prod_comp).sum()
    return prod_comp.apply(lambda x: cum_bpc[(x)])
