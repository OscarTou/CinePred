from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score


def get_baseline_scores(model, X, y):
    """
        Baseline model

        Input:
            X, y: sorted by year (ascending), cleaning and scaled
            model: LinearRegression (for example)

        Returns:
            r2: a list of 5 r2 scores
    """

    r2 = []
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2.append(r2_score(y_test, y_pred))
    print("R2 scores: ", r2)
    return r2
