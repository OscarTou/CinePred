from currency_converter import CurrencyConverter


def convert (value,in_currency, out_currency):
    c = CurrencyConverter()
    if in_currency == '$':
        in_currency = 'USD'
    elif in_currency == 'DEM' :
        return value*0.576
    elif in_currency == 'FRF' :
        return value*0.172

    result = c.convert(value,in_currency,out_currency)
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
