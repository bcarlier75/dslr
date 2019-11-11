import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from collections import OrderedDict
from logreg_tools import plot_confusion_matrix, metrics, confusion_matrix, score


class LogisticRegressionOvrPredict(object):
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
        preds = [np.argmax([self._sigmoid(np.dot(xi, theta))
                            for theta in thetas]) for xi in x]
        return [classes[p] for p in preds]


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
        final_cm = confusion_matrix(u_classes, y_true, y_pred)
        final_metrics = metrics(final_cm, u_classes, debug=False)
        print(f'\n-------- Metrics on test dataset --------'
              f'\n. . . . . . . . .\nAccuracy: {score(y_pred, y_true):.5f}'
              f'\n. . . . . . . . .\nConfusion matrix:\n{final_cm}'
              f'\n. . . . . . . . .\nMetrics:\n{final_metrics}'
              f'\n. . . . . . . . .\n------------------------------------------\n')
        np.set_printoptions(precision=3)
        # Plot normalized confusion matrix.
        # Change normalize to False for non normalized version.
        plot_confusion_matrix(y_true, y_pred, classes=u_classes, cm=final_cm, normalize=True)
        plt.show()
