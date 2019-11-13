import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv
from collections import OrderedDict
from logreg_tools import plot_confusion_matrix, metrics, confusion_matrix, score


class LogisticRegressionOvrPredict(object):
    def _normalize(self, x):
        for i in range(len(x)):
            x[i] = (x[i] - x.mean()) / x.std()
        return x

    def preprocessing(self, df: pd.DataFrame):
        # Features wrangling
        df_features = df.iloc[:, 5:]
        df_features = df_features.fillna(df.mean())
        df_features = np.array(df_features)
        np.apply_along_axis(self._normalize, 0, df_features)
        return df_features

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, classes, thetas, x):
        x = np.insert(x, 0, 1, axis=1)  # adding interception feature
        preds = [np.argmax([self._sigmoid(np.dot(xi, theta))
                            for theta in thetas]) for xi in x]
        return np.array([classes[p] for p in preds])


if __name__ == "__main__":
    verbose = False
    if len(argv) > 3 and argv[3] == '-v':
        verbose = True
    # Initialization and data wrangling
    df_test = pd.read_csv(argv[1], index_col="Index")
    logreg = LogisticRegressionOvrPredict()
    x_test = logreg.preprocessing(df_test)
    u_classes = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    # Compute predictions and save it to houses.csv
    y_pred = logreg.predict(u_classes, np.load(argv[2], allow_pickle=True), x_test)
    houses = pd.DataFrame(OrderedDict({'Index': range(len(y_pred)), 'Hogwarts House': y_pred}))
    houses.to_csv('houses.csv', index=False)
    print("Predictions saved to houses.csv.")

    if verbose:
        df_truth = pd.read_csv('datasets/dataset_truth.csv', index_col="Index")
        y_true = df_truth.loc[:, 'Hogwarts House']
        final_cm = confusion_matrix(u_classes, y_true, y_pred)
        final_metrics = metrics(final_cm, u_classes, debug=False)
        print(f'\n-------- Metrics on test dataset --------'
              f'\n. . . . . . . . .\nAccuracy: {score(y_true, y_pred):.5f}'
              f'\n. . . . . . . . .\nConfusion matrix:\n{final_cm}'
              f'\n. . . . . . . . .\nMetrics:\n{final_metrics}'
              f'\n. . . . . . . . .\n------------------------------------------\n')
        # Plot confusion matrix.
        # Change normalize to False for non-normalized version. (False by default)
        plot_confusion_matrix(y_true, y_pred, classes=u_classes, cm=final_cm, normalize=True)
        plt.show()
