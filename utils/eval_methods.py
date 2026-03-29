# -*- coding: utf-8 -*-
import numpy as np
import sklearn.metrics


def get_adjusted_composite_metrics(score, label):
    score = -score
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])


    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true=label, y_score=score, drop_intermediate=False)
    auroc = sklearn.metrics.auc(fpr, tpr)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true=label, probas_pred=score)

    f1 = np.max(2 * precision * recall / (precision + recall + 1e-5))
    ap = sklearn.metrics.average_precision_score(y_true=label, y_score=score, average=None)
    return auroc, ap, f1, precision, recall, fpr, tpr
