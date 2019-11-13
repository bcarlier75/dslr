import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_cost_history(num_iter: int, cost_hist: np.array, class_i: int):
    """
    This function plots the cost history.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel(f'Cross-Entropy loss for class index {class_i}')
    _ = ax1.plot(range(num_iter), cost_hist[num_iter*class_i:num_iter*(class_i+1)], 'b.')
    return ax1


def plot_confusion_matrix(y_true: np.array, y_pred: np.array, classes: list, cm: pd.DataFrame,
                          normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=3)
    cm = np.array(cm)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax2 = plt.subplots()
    im = ax2.imshow(cm, interpolation='nearest', cmap=cmap)
    ax2.figure.colorbar(im, ax=ax2)
    # We want to show all ticks...
    ax2.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax2


def score(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculate the accuracy of our predictions.
    :param y_pred: numpy array of true outputs
    :param y_true: numpy array of predicted outputs
    :return:
    """
    return (y_pred == y_true).mean()


def confusion_matrix(classes: list, y_true: np.array, y_pred: np.array) -> pd.DataFrame:
    """
    Retrive the confuson matrix as a dataframe.
    :param classes: list of unique classes
    :param y_true: numpy array of true outputs
    :param y_pred: numpy array of predicted outputs
    :return: a dataframe corresponding to the confusion matrix
    """
    cm_tab = pd.DataFrame(0, columns=classes, index=classes)
    for i in range(len(y_pred)):
        for c in cm_tab.columns:
            sublist = [x for x in classes if x != c]
            # Diagonal values, true positives
            if y_pred[i] == c and y_true[i] == c:
                cm_tab.loc[c][c] += 1
            # Other values, type I and type II errors
            for j in range(len(sublist)):
                if y_pred[i] == c and y_true[i] == sublist[j]:
                    cm_tab.loc[sublist[j]][c] += 1
    # Simple unit test to ensure we retrieve all samples
    assert cm_tab.values.sum() == len(y_pred)
    return cm_tab


def metrics(cm: pd.DataFrame, classes: list, debug=False) -> pd.DataFrame:
    """
    Return a dataframe with metrics informations for each class such as : 
        - precision : tp / (tp + fp)
        - recall    : tp / (tp + fn)
        - f1score   : 2 * tp / (2 * tp + fp + fn) == 2 * (precision * recall) / (precision + recall)
        - Number of observations per class predicted
    :param cm: dataframe of the confusion matrix
    :param classes: list of unique classes
    :param debug: if set to True print tp, tn, fp and fn for each class
    :return: 
    """
    m_classes = classes + ['--avg--']
    m_tab = pd.DataFrame(0, index=m_classes, columns=['precision', 'recall', 'f1-score', 'N Obs'])
    m_tab = m_tab.astype({'precision': float, 'recall': float, 'f1-score': float, 'N Obs': int})
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
        m_tab.loc[c, 'precision'], m_tab.loc[c, 'recall'], m_tab.loc[c, 'f1-score'], m_tab.loc[c, 'N Obs'] \
            = precision, recall, f1score, n_obs
    m_tab.loc['--avg--', 'precision'] = m_tab.loc[classes[0]:classes[len(classes) - 1], 'precision'].mean()
    m_tab.loc['--avg--', 'recall'] = m_tab.loc[classes[0]:classes[len(classes) - 1], 'recall'].mean()
    m_tab.loc['--avg--', 'f1-score'] = m_tab.loc[classes[0]:classes[len(classes) - 1], 'f1-score'].mean()
    m_tab.loc['--avg--', 'N Obs'] = m_tab.loc[classes[0]:classes[len(classes) - 1], 'N Obs'].sum()
    return m_tab
