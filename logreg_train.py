import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from plot_cm import plot_confusion_matrix


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

    def preprocessing(self, df: pd.DataFrame):
        df_features = df.iloc[:, 5:]
        df_features = df_features.fillna(df.mean())
        df_features = np.array(df_features)
        df_labels = np.array(df.loc[:, "Hogwarts House"])
        np.apply_along_axis(self._scaling, 0, df_features)
        return df_features, df_labels

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _cost(self, theta, x, y):
        epsilon = 1e-5

        h = self._sigmoid(np.dot(x, theta))
        m = len(y)
        cost = (1 / m) * np.sum(-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon))
        grad = 1 / m * (np.dot((y - h), x))
        return cost, grad

    def fit(self, x, y, max_iter=5000, alpha=0.1, verbose=False):
        x = np.insert(x, 0, 1, axis=1)
        thetas = []
        classes = np.unique(y)
        costs = np.zeros(max_iter)

        for c in classes:
            if verbose:
                print(f'Class {c} vs all:')
            # one vs. rest binary classification
            binary_y = np.where(y == c, 1, 0)

            theta = np.zeros(x.shape[1])
            for epoch in range(max_iter):
                costs[epoch], grad = self._cost(theta, x, binary_y)
                theta += alpha * grad
                if verbose and epoch % (max_iter / 10) == 0:
                    print(f'epoch {epoch:<6}: loss is {costs[epoch]:.15f}')
            thetas.append(theta)
            if verbose is True:
                print()
        return thetas

    def predict(self, classes, thetas, x):
        x = np.insert(x, 0, 1, axis=1)
        preds = [np.argmax(
            [self._sigmoid(np.dot(xi, theta)) for theta in thetas]
        ) for xi in x]
        return [classes[p] for p in preds]

    def score(self, classes, theta, x, y):
        return (self.predict(classes, theta, x) == y).mean()

    def confusion_matrix(self, classes, theta, x, y):
        predictions = self.predict(classes, theta, x)
        cm_matrix = pd.DataFrame(0, columns=classes, index=classes)
        for i in range(len(predictions)):
            for c in cm_matrix.columns:
                sublist = [x for x in classes if x != c]
                if predictions[i] == c and y[i] == c:
                    cm_matrix.loc[c][c] += 1
                for j in range(len(sublist)):
                    if predictions[i] == c and y[i] == sublist[j]:
                        cm_matrix.loc[sublist[j]][c] += 1
        assert cm_matrix.values.sum() == len(predictions)
        return cm_matrix

    def metrics(self, cm, classes, debug=False):
        m_classes = classes + ['--avg--']
        metrics = pd.DataFrame(0, index=m_classes, columns=['precision', 'recall', 'f1-score', 'N Obs'])
        metrics = metrics.astype({'precision': float, 'recall': float, 'f1-score': float, 'N Obs': int})
        for c in cm.columns:
            tp = tn = fp = fn = 0
            sublist = [x for x in classes if x != c]
            tp = cm.loc[c, c]
            for j in range(len(sublist)):
                tn += cm.loc[sublist[j], sublist[j]]
                fp += cm.loc[sublist[j], c]
                fn += cm.loc[c, sublist[j]]
            precision, recall, f1score, n_obs = \
                tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn), tp + fp
            if debug:
                print(f'\nClass: {c}\n\t'
                      f'True positives: {tp}\n\t'
                      f'True negatives: {tn}\n\t'
                      f'False positives: {fp}\n\t'
                      f'False negatives: {fn}')
            metrics.loc[c, 'precision'], metrics.loc[c, 'recall'], metrics.loc[c, 'f1-score'], metrics.loc[c, 'N Obs'] \
                = precision, recall, f1score, n_obs
        metrics.loc['--avg--', 'precision'] = metrics.loc[classes[0]:classes[len(classes) - 1], 'precision'].mean()
        metrics.loc['--avg--', 'recall'] = metrics.loc[classes[0]:classes[len(classes) - 1], 'recall'].mean()
        metrics.loc['--avg--', 'f1-score'] = metrics.loc[classes[0]:classes[len(classes) - 1], 'f1-score'].mean()
        metrics.loc['--avg--', 'N Obs'] = metrics.loc[classes[0]:classes[len(classes) - 1], 'N Obs'].sum()
        return metrics


if __name__ == "__main__":
    verb = False
    if len(sys.argv) > 2 and sys.argv[2] == '-v':
        verb = True
    df_train = pd.read_csv(sys.argv[1], index_col="Index")
    logreg = LogisticRegressionOvrTrain()
    x_train, y_train = logreg.preprocessing(df_train)
    weights = logreg.fit(x_train, y_train, max_iter=5000, alpha=0.01, verbose=verb)
    np.save('weights', weights)
    print('Weights saved in weights.npy.')
    if verb:
        u_classes = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        final_cm = logreg.confusion_matrix(u_classes, weights, x_train, y_train)
        final_metrics = logreg.metrics(final_cm, u_classes, debug=False)
        print(f'\n-------- Metrics on train dataset --------\n. . . . . . . . .'
              f'\nAccuracy: {logreg.score(u_classes, weights, x_train, y_train):.5f}')
        print(f'. . . . . . . . .\nConfusion matrix:\n{final_cm}')
        print(f'. . . . . . . . .\nMetrics:\n{final_metrics}')
        print(f'. . . . . . . . .\n------------------------------------------\n')
        y_pred = logreg.predict(u_classes, weights, x_train)
        np.set_printoptions(precision=3)
        # Plot normalized confusion matrix.
        # Change normalize to False for non normalized version.
        plot_confusion_matrix(y_train, y_pred, classes=u_classes, cm=final_cm, normalize=True)
        plt.show()
