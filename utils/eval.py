# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from auto_rpGCN.GCN.GCN_PyTorch.utils.spot import SPOT
import pandas as pd
import sklearn

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """

    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)

    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)

    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False,
                    ):
    # """
    # Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    #
    # Args:
    #     score (np.ndarray): The anomaly score
    #     label (np.ndarray): The ground-truth label
    #     threshold (float): The threshold of anomaly score.
    #         A point is labeled as "anomaly" if its score is lower than the threshold.
    #     pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
    #     calc_latency (bool):
    #
    # Returns:
    #     np.ndarray: predict labels
    # """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:

            predict = score>threshold

    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict



def pot_eval(init_score, score, label, q):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            For `OmniAnomaly`, it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            For `OmniAnomaly`, it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t

    Returns:
        dict: pot result dict
    """
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)  # data import, like M_train M_test
    s.initialize()  # initialization step
    ret = s.run()  # run
    s.plot(ret)  # plot
    plt.xlabel("step", fontsize=20)
    plt.ylabel("error", fontsize=20)

    plt.show()

    print(len(ret['alarms']))
    print(len(ret['thresholds']))
    pot_th = np.mean(ret['thresholds'])


    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    print('POT result: ', p_t, pot_th, p_latency)
    return {
        'pot-f1': p_t[0],
        'pot-precision': p_t[1],
        'pot-recall': p_t[2],
        'pot-TP': p_t[3],
        'pot-TN': p_t[4],
        'pot-FP': p_t[5],
        'pot-FN': p_t[6],
        'pot-threshold': pot_th,
        'pot-latency': p_latency
    }


def pot_eval_new(score, label,data_name,pot_th):

    pred, p_latency = adjust_predicts(score, label,data_name, pot_th, calc_latency=True)
    p_t,precision, recall, TP, TN, FP, FN = calc_point2point(pred, label)

    return p_t,precision, recall, TP, TN, FP, FN

def optimal_judgment(y_pred,y_test,data_name,strat_th,end_th,step_th):


    pot_th = np.arange(strat_th, end_th, step_th)


    f1_all=[]
    for pot_thod in pot_th:
        pot_result,precision, recall, TP, TN, FP, FN = pot_eval_new(y_pred, y_test[-len(y_test):],data_name,pot_thod)
        f1_all.append(pot_result)

    dic = dict(zip(pot_th,f1_all))
    dic_sort = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    max_f = dic_sort[0][1]
    key_f = dic_sort[0][0]

    pot_result,precision, recall, TP, TN, FP, FN = pot_eval_new(y_pred, y_test[-len(y_test):],data_name,key_f)
    print("best precision{}_".format(precision))
    print("best recall{}_".format(recall))
    return {'f1': pot_result, 'precision': precision,
            'recall': recall, 'TP': TP,
            'TN': TN, 'FP': FP, 'FN': FN,
            'pot-threshold': key_f}
