import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv, exit
from logreg_tools import plot_confusion_matrix, plot_cost_history, metrics, confusion_matrix, score


class LogisticRegressionOvrMiniBatch(object):
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
        # Labels wrangling
        df_labels = np.array(df.loc[:, "Hogwarts House"])
        return df_features, df_labels

    def iterate_minibatches(self, x, y, batch_size, shuffle=False):
        assert x.shape[0] == y.shape[0]
        if shuffle:
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, x.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, x.shape[0])
            if shuffle:
                excerpt = indices[start_idx:end_idx]
            else:
                excerpt = slice(start_idx, end_idx)
            yield x[excerpt], y[excerpt]

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, classes, thetas, x):
        x = np.insert(x, 0, 1, axis=1)  # adding interception feature
        preds = [np.argmax([self._sigmoid(np.dot(xi, theta))
                            for theta in thetas]) for xi in x]
        return np.array([classes[p] for p in preds])

    def _loss(self, h, y, m):
        e = 1e-15   # epsilon value to avoid log(0) errors
        loss = -1 / m * np.sum(y * np.log(h + e) + (1 - y) * np.log(1 - h + e))
        return loss

    def fit(self, x: np.array, y: np.array, num_iter=1000, eta0=0.01,
            verbose=True, learning_rate='optimal', alpha=0.0001,
            batch_size=40, shuffle=True):

        x = np.insert(x, 0, 1, axis=1)  # adding interception feature

        # Initialize lists
        theta_list = []
        loss_list = []
        classes = np.unique(y)

        # Initialize optimal_init or eta according to the learning rate
        if learning_rate == 'optimal':
            optimal_init = 1.0 / (eta0 * alpha)
        elif learning_rate == 'constant' or learning_rate == 'invscaling':
            eta = eta0
        else:
            exit("Unknow parameter for learning_rate : available parameters"
                 "are 'optimal', 'invscaling or 'constant'")
        for c in classes:
            if verbose:
                print(f'\nClass {c} vs all:')
            binary_y = np.where(y == c, 1, 0)
            theta = np.zeros(x.shape[1])
            # Loop for number of iterations = 'num_iter'
            for epoch in range(num_iter):
                # Loop on all batches of size = 'batch_size'
                t = 1
                for batch in self.iterate_minibatches(x, binary_y, batch_size, shuffle):
                    x_batch, y_batch = batch
                    m = len(y_batch)
                    h = self._sigmoid(np.dot(x_batch, theta))
                    gradient = np.dot(x_batch.T, (h - y_batch))
                    if learning_rate == 'optimal':
                        eta = 1.0 / (alpha * (optimal_init + t - 1))
                    elif learning_rate == 'invscaling':
                        eta = eta0 / np.power(t, 0.5)
                    theta -= eta * (1 / m) * gradient
                    t += 1
                if verbose:
                    loss = self._loss(h, y_batch, m)
                    loss_list.append(loss)
                    if epoch % (num_iter / 10) == 0:
                        print(f'epoch {epoch:<6}: loss {loss:.15f}')

            theta_list.append(theta)
        return np.array(theta_list), np.array(loss_list)


class LogisticRegressionOvrBatch(object):
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
        # Labels wrangling
        df_labels = np.array(df.loc[:, "Hogwarts House"])
        return df_features, df_labels

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, classes, thetas, x):
        x = np.insert(x, 0, 1, axis=1)  # adding interception feature
        preds = [np.argmax([self._sigmoid(np.dot(xi, theta))
                            for theta in thetas]) for xi in x]
        return np.array([classes[p] for p in preds])

    def _loss(self, h, y, m):
        e = 1e-15   # epsilon value to avoid log(0) errors
        loss = -1 / m * np.sum(y * np.log(h + e) + (1 - y) * np.log(1 - h + e))
        return loss

    def fit(self, x: np.array, y: np.array, num_iter=5000, eta0=0.01,
            verbose=False, learning_rate='optimal', alpha=0.0001):

        x = np.insert(x, 0, 1, axis=1)  # adding interception feature

        # Initialize lists and m
        theta_list = []
        loss_list = []
        classes = np.unique(y)
        m = len(y)

        # Initialize optimal_init or eta according to the learning rate
        if learning_rate == 'optimal':
            optimal_init = 1.0 / (eta0 * alpha)
        elif learning_rate == 'constant' or learning_rate == 'invscaling':
            eta = eta0
        else:
            exit("Unknow parameter for learning_rate : available parameters"
                 "are 'optimal', 'invscaling or 'constant'")

        for c in classes:
            if verbose:
                print(f'\nClass {c} vs all:')
            binary_y = np.where(y == c, 1, 0)
            theta = np.zeros(x.shape[1])
            # Loop for number of iterations = 'num_iter'
            for t in range(num_iter):
                h = self._sigmoid(np.dot(x, theta))
                gradient = np.dot(x.T, (h - binary_y))
                if learning_rate == 'optimal':
                    eta = 1.0 / (alpha * (optimal_init + (t + 1) - 1))
                elif learning_rate == 'invscaling':
                    eta = eta0 / np.power((t + 1), 0.5)
                theta -= eta * (1 / m) * gradient
                if verbose:
                    loss = self._loss(h, binary_y, m)
                    loss_list.append(loss)
                    if t % (num_iter / 10) == 0:
                        print(f'epoch {t:<6}: loss {loss:.15f}')
            theta_list.append(theta)
        return np.array(theta_list), np.array(loss_list)


if __name__ == "__main__":
    # Setting hyperparameters here
    verbose = False
    if len(argv) > 2 and argv[2] == '-v':
        verbose = True
    num_iter = 1000         # number of iterations
    eta0 = 0.01             # learning rate
    batch_size = 40         # batch size for mini-batch GD
    shuffle = True          # introduce stochasticity to the model if set to True
    # Initialization and data wrangling
    df_train = pd.read_csv(argv[1], index_col="Index")
    logreg = LogisticRegressionOvrBatch()   # Batch GD
    # logreg = LogisticRegressionOvrMiniBatch() #Mini-batch GD and SGD
    x_train, y_train = logreg.preprocessing(df_train)

    # Retrieve weights + cost history (for the later, only if verbose set to True)
    # -- Batch GD fit
    weights, cost_hist = logreg.fit(x_train, y_train, num_iter, eta0, verbose, 'optimal')

    # -- Mini-batch GD or SGD fit
    # weights, cost_hist = logreg.fit(x_train, y_train, num_iter, eta0, verbose,
    #                                 'optimal', 0.0001, batch_size, shuffle)

    np.save('weights', weights)
    print('Weights saved to weights.npy.')

    if verbose:
        u_classes = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        y_pred = logreg.predict(u_classes, weights, x_train)
        final_cm = confusion_matrix(u_classes, y_train, y_pred)
        final_metrics = metrics(final_cm, u_classes, debug=False)
        print(f'\n-------- Metrics on train dataset --------'
              f'\n. . . . . . . . .\nAccuracy: {score(y_train, y_pred):.5f}'
              f'\n. . . . . . . . .\nConfusion matrix:\n{final_cm}'
              f'\n. . . . . . . . .\nMetrics:\n{final_metrics}'
              f'\n. . . . . . . . .\n------------------------------------------\n')
        # Plot cost history for class of index = ' class_i' in u_classes.
        plot_cost_history(num_iter, cost_hist, class_i=0)
        # Plot confusion matrix.
        # Change normalize to False for non-normalizeda version. (False by default)
        plot_confusion_matrix(y_train, y_pred, classes=u_classes, cm=final_cm, normalize=True)
        plt.show()
