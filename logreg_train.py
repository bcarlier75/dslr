import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from logreg_tools import plot_confusion_matrix, metrics, confusion_matrix, score


class LogisticRegressionOvrGd(object):
    def _scaling(self, x):
        for i in range(len(x)):
            x[i] = (x[i] - x.mean()) / x.std()
        return x

    def preprocessing(self, df: pd.DataFrame):
        df_features = df.iloc[:, 5:]
        df_features = df_features.fillna(df.mean())
        df_features = np.array(df_features)
        df_labels = np.array(df.loc[:, "Hogwarts House"])
        np.apply_along_axis(self._scaling, 0, df_features)
        return df_features, df_labels

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, classes, thetas, x):
        x = np.insert(x, 0, 1, axis=1)
        preds = [np.argmax([self._sigmoid(np.dot(xi, theta))
                            for theta in thetas]) for xi in x]
        return [classes[p] for p in preds]

    def _loss(self, h, y):
        e = 1e-6
        loss = 1 / len(y) * np.sum(-y * np.log(h + e) - (1 - y) * np.log(1 - h + e))
        return loss

    def fit(self, x, y, num_iter=5000, alpha=0.01, verbose=False):
        x = np.insert(x, 0, 1, axis=1)
        thetas = []
        classes = np.unique(y)
        # one vs. rest binary classification
        for c in classes:
            if verbose:
                print(f'\nClass {c} vs all:')
            binary_y = np.where(y == c, 1, 0)
            theta = np.zeros(x.shape[1])
            for epoch in range(num_iter):
                h = self._sigmoid(np.dot(x, theta))
                grad = 1 / len(binary_y) * np.dot(x.T, (h - binary_y))
                theta -= alpha * grad
                if verbose and epoch % (num_iter / 10) == 0:
                    print(f'epoch {epoch:<6}: loss {self._loss(h, binary_y):.15f}')
            thetas.append(theta)
        return thetas


if __name__ == "__main__":
    verb = False
    if len(sys.argv) > 2 and sys.argv[2] == '-v':
        verb = True
    df_train = pd.read_csv(sys.argv[1], index_col="Index")
    logreg = LogisticRegressionOvrGd()
    x_train, y_train = logreg.preprocessing(df_train)
    weights = logreg.fit(x_train, y_train, num_iter=5000, alpha=0.01, verbose=verb)
    np.save('weights', weights)
    print('Weights saved to weights.npy.')
    if verb:
        u_classes = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        y_pred = logreg.predict(u_classes, weights, x_train)
        final_cm = confusion_matrix(u_classes, y_train, y_pred)
        final_metrics = metrics(final_cm, u_classes, debug=False)
        print(f'\n-------- Metrics on train dataset --------'
              f'\n. . . . . . . . .\nAccuracy: {score(y_pred, y_train):.5f}'
              f'\n. . . . . . . . .\nConfusion matrix:\n{final_cm}'
              f'\n. . . . . . . . .\nMetrics:\n{final_metrics}'
              f'\n. . . . . . . . .\n------------------------------------------\n')
        y_pred = logreg.predict(u_classes, weights, x_train)
        np.set_printoptions(precision=3)
        # Plot normalized confusion matrix.
        # Change normalize to False for non normalized version.
        plot_confusion_matrix(y_train, y_pred, classes=u_classes, cm=final_cm, normalize=True)
        plt.show()
