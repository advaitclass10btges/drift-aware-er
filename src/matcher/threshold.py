import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(y_true, scores):

    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.linspace(0,1,101):

        preds = (
            scores >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            preds
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1
