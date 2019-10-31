import numpy as np
import pandas as pd
import sys
from collections import OrderedDict


class LogisticRegressionOvrPredict(object):
    def __init__(self, eta=5e-5, n_iter=30000):
        self.eta = eta
        self.n_iter = n_iter

    def _scaling(self, x):
        for i in range(len(x)):
            x[i] = (x[i] - x.mean()) / x.std()
        return x

    def _processing(self, df):
        df = df.iloc[:, 5:]
        df = df.dropna()
        df_features = np.array(df)

        np.apply_along_axis(self._scaling, 0, df_features)
        return df_features

    def _predict_one(self, x, weights):
        return max((x.dot(w), c) for w, c in weights)[1]

    def predict(self, x, weights):
        x = self._processing(x)
        return [self._predict_one(i, weights) for i in np.insert(x, 0, 1, axis=1)]


if __name__ == "__main__":
    df_test = pd.read_csv(sys.argv[1], index_col="Index")
    logreg = LogisticRegressionOvrPredict()
    predictions = logreg.predict(df_test, np.load(sys.argv[2],allow_pickle=True ))
    print("Predictions saved to houses.csv :", predictions)
    houses = pd.DataFrame(OrderedDict({'Index': range(len(predictions)), 'Hogwarts House': predictions}))
    houses.to_csv('houses.csv', index=False)