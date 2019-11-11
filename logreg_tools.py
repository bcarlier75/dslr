import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, classes, cm: pd.DataFrame,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = np.array(cm)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def score(y_p, y_t):
    return (y_p == y_t).mean()


def confusion_matrix(classes, y_t, y_p):
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


def metrics(cm, classes, debug=False):
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
