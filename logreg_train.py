import numpy as np
import pandas as pd
import sys


class LogisticRegressionOvrTrain(object):
    def __init__(self, w=None, eta=5e-5, n_iter=30000):
        if w is None:
            w = []
        self.eta = eta
        self.n_iter = n_iter
        self.w = w

    def _scaling(self, x):
        for i in range(len(x)):
            x[i] = (x[i] - x.mean()) / x.std()
        return x

    def _preprocessing(self, df):
        df = df.dropna()
        df_features = np.array((df.iloc[:, 5:]))
        df_labels = np.array(df.loc[:, "Hogwarts House"])
        print(df_labels)
        np.apply_along_axis(self._scaling, 0, df_features)
        return df_features, df_labels

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, df):
        x, y = self._preprocessing(df)
        x = np.insert(x, 0, 1, axis=1)
        m = x.shape[0]

        for i in np.unique(y):
            y_copy = np.where(y == i, 1, 0)
            w = np.ones(x.shape[1])

            for _ in range(self.n_iter):
                output = x.dot(w)
                errors = y_copy - self._sigmoid(output)
                gradient = np.dot(x.T, errors)
                w += self.eta * gradient

            self.w.append((w, i))
        return self.w

    def _predict_one(self, x):
        return max((x.dot(w), c) for w, c in self.w)[1]

    def predict(self, x):
        return [self._predict_one(i) for i in np.insert(x, 0, 1, axis=1)]

    def score(self, df):
        x, y = self._preprocessing(df)
        return sum(self.predict(x) == y) / len(y)


def main():
    df_train = pd.read_csv(sys.argv[1], index_col="Index")
    logreg = LogisticRegressionOvrTrain()
    weights = logreg.fit(df_train)
    np.save('weights', weights)
    print('Weights saved in weights.npy.')
    print(f'Accuracy : {logreg.score(df_train)}')


if __name__ == "__main__":
    main()
