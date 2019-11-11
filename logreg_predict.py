import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from collections import OrderedDict
from plot_cm import plot_confusion_matrix


class LogisticRegressionOvrPredict(object):
    def __init__(self, eta=5e-5, n_iter=30000):
        self.eta = eta
        self.n_iter = n_iter

    def _scaling(self, x):
        for i in range(len(x)):
            x[i] = (x[i] - x.mean()) / x.std()
        return x

    def preprocessing(self, df):
        df_features = df.iloc[:, 5:]
        df_features = df_features.fillna(df.mean())
        df_features = np.array(df_features)
        np.apply_along_axis(self._scaling, 0, df_features)
        return df_features

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, classes, thetas, x):
        x = np.insert(x, 0, 1, axis=1)
        preds = [np.argmax(
            [self._sigmoid(np.dot(xi, theta)) for theta in thetas]
        ) for xi in x]
        return [classes[p] for p in preds]

    def score(self, classes, theta, x, y):
        return (self. predict(classes, theta, x) == y).mean()

    def confusion_matrix(self, classes, y_t, y_p):
        cm_matrix = pd.DataFrame(0, columns=classes, index=classes)
        for i in range(len(y_p)):
            for c in cm_matrix.columns:
                sublist = [x for x in classes if x != c]
                if y_p[i] == c and y_t[i] == c:
                    cm_matrix.loc[c][c] += 1
                for j in range(len(sublist)):
                    if y_p[i] == c and y_t[i] == sublist[j]:
                        cm_matrix.loc[sublist[j]][c] += 1
        assert cm_matrix.values.sum() == len(y_p)
        return cm_matrix

    def metrics(self, cm, classes, debug=False):
        m_classes = classes + ['--avg--']
        metrics = pd.DataFrame(0, index=m_classes, columns=['precision', 'recall', 'f1-score', 'N Obs'])
        metrics = metrics.astype({'precision': float, 'recall': float, 'f1-score': float, 'N Obs': int})
        for c in cm.columns:
            tp = tn = fp = fn = 0
            sublist = [x for x in classes if x != c]
            tp = cm.loc[c][c]
            for j in range(len(sublist)):
                tn += cm.loc[sublist[j], sublist[j]]
                fp += cm.loc[sublist[j], c]
                fn += cm.loc[c, sublist[j]]
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1score = 2 * tp / (2 * tp + fp + fn)
            n_obs = tp + fp
            if debug is True:
                print(f'\nClass: {c}\n\t'
                      f'True positives: {tp}\n\t'
                      f'True negatives: {tn}\n\t'
                      f'False positives: {fp}\n\t'
                      f'False negatives: {fn}')
            metrics.loc[c, 'precision'] = precision
            metrics.loc[c, 'recall'] = recall
            metrics.loc[c, 'f1-score'] = f1score
            metrics.loc[c, 'N Obs'] = n_obs
        metrics.loc['--avg--', 'precision'] = metrics.loc[classes[0]:classes[len(classes)-1], 'precision'].mean()
        metrics.loc['--avg--', 'recall'] = metrics.loc[classes[0]:classes[len(classes)-1], 'recall'].mean()
        metrics.loc['--avg--', 'f1-score'] = metrics.loc[classes[0]:classes[len(classes)-1], 'f1-score'].mean()
        metrics.loc['--avg--', 'N Obs'] = metrics.loc[classes[0]:classes[len(classes)-1], 'N Obs'].sum()
        return metrics


if __name__ == "__main__":
    verb = False
    if len(sys.argv) > 3 and sys.argv[3] == '-v':
        verb = True
    df_test = pd.read_csv(sys.argv[1], index_col="Index")
    u_classes = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    logreg = LogisticRegressionOvrPredict()
    x_test = logreg.preprocessing(df_test)
    y_pred = logreg.predict(u_classes, np.load(sys.argv[2], allow_pickle=True), x_test)
    houses = pd.DataFrame(OrderedDict({'Index': range(len(y_pred)), 'Hogwarts House': y_pred}))
    houses.to_csv('houses.csv', index=False)
    print("Predictions saved to houses.csv.")
    if verb:
        df_truth = pd.read_csv('datasets/dataset_truth.csv', index_col="Index")
        y_true = df_truth.loc[:, 'Hogwarts House']
        final_cm = logreg.confusion_matrix(u_classes, y_true, y_pred)
        final_metrics = logreg.metrics(final_cm, u_classes, debug=False)
        print(f'\n-------- Metrics on test dataset --------\n'
              f'. . . . . . . . .\nAccuracy: {(y_pred == y_true).mean():.5f}\n'
              f'. . . . . . . . .\nConfusion matrix:\n{final_cm}\n'
              f'. . . . . . . . .\nMetrics:\n{final_metrics}\n'
              f'. . . . . . . . .\n------------------------------------------\n')
        np.set_printoptions(precision=3)
        # Plot normalized confusion matrix.
        # Change normalize to False for non normalized version.
        plot_confusion_matrix(y_true, y_pred, classes=u_classes, cm=final_cm, normalize=True)
        plt.show()
