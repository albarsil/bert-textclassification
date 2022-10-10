
import warnings
from collections import Counter

import numpy as np
import sklearn.metrics as metrics
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import make_scorer

"""
    This script contains a series of metrics that can be extracted from any machine learning algorithms
    Mostly of the methods receive the <y_true> that is the gold and <y_pred> which is the algorithm predictions.
"""

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def metric_accuracy(y_true, y_pred, pos_label=True):
    return metrics.accuracy_score(y_true, y_pred)

def metric_f1(y_true, y_pred, pos_label=True):
    return metrics.f1_score(y_true, y_pred, average='weighted', pos_label=pos_label)

def metric_f1_micro(y_true, y_pred, pos_label=True):
    return metrics.f1_score(y_true, y_pred, average='micro', pos_label=pos_label)

def metric_f1_macro(y_true, y_pred, pos_label=True):
    return metrics.f1_score(y_true, y_pred, average='macro', pos_label=pos_label)

def metric_precision(y_true, y_pred, pos_label=True):
    return metrics.precision_score(y_true, y_pred, average='binary', pos_label=pos_label)

def metric_recall(y_true, y_pred, pos_label=True):
    return metrics.recall_score(y_true, y_pred, average='binary', pos_label=pos_label)

def metric_auc(y_true, y_pred, pos_label=True):
    try:
        fpr, tpr, _threshold = metrics.roc_curve(
            y_true,
            y_pred,
            pos_label=pos_label
        )
        return metrics.auc(fpr, tpr)
    except:
        return None

def metric_kappa(y_true, y_pred):
    return metrics.cohen_kappa_score(y_true, y_pred)

def metric_roc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred, average='weighted')

def metric_log_loss(y_true, y_pred):
    try:
        return metrics.log_loss(y_true, y_pred)
    except:
        return None

def metric_true_positive(y_true, y_pred):
    _tn, _fp, _fn, tp = _confusion_matrix(y_true, y_pred)
    return tp

def metric_true_negative(y_true, y_pred):
    tn, _fp, _fn, _tp = _confusion_matrix(y_true, y_pred)
    return tn

def metric_false_positive(y_true, y_pred):
    _tn, fp, _fn, _tp = _confusion_matrix(y_true, y_pred)
    return fp

def metric_false_negative(y_true, y_pred):
    _tn, _fp, fn, _tp = _confusion_matrix(y_true, y_pred)
    return fn

def metric_positive_predictive_value(y_true, y_pred):
    _tn, fp, _fn, tp = _confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def metric_negative_predictive_value(y_true, y_pred):
    tn, _fp, fn, _tp = _confusion_matrix(y_true, y_pred)
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0

def metric_sensitivity(y_true, y_pred):
    _tn, _fp, fn, tp = _confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def metric_specificity(y_true, y_pred):
    tn, fp, _fn, _tp = _confusion_matrix(y_true, y_pred)
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def metric_expected_no(y_true, y_pred): #pylint: disable=W0613
    return Counter(y_true)[0]

def metric_expected_yes(y_true, y_pred): #pylint: disable=W0613
    return Counter(y_true)[1]

def metric_diff_expected(y_true, y_pred):
    return abs(Counter(y_true)[1] - Counter(y_pred)[1])

def _confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return int(tn), int(fp), int(fn), int(tp)

def smape(y, y_hat):
  return (1/y.size * np.sum(np.abs(y-y_hat) / (np.abs(y) + np.abs(y_hat))*100))

def corr(y, y_hat):
  return np.corrcoef(y, y_hat)[0][1]

def rmse(y, y_hat):
  return np.sqrt(np.mean(np.square(y - y_hat)))

def quantiles(y, y_hat, p):
  try:
    if isinstance(p, float):
      return np.mean((np.quantile(y_hat, p)-np.quantile(y, p))/np.quantile(y, p))
    else:
      return np.mean((np.sum(np.quantile(y_hat, p))-np.sum(np.quantile(y, p)))/np.sum(np.quantile(y, p)))
  except:
    return np.Inf

def evaluate_binary_classification(y_true: list, y_pred: list) -> dict:
    """
    Extract metrics from predicted and expected

    Parameters:
        y_true (list): A list of ground truth (correct) target values
        y_pred (list): A list of estimated targets

    Returns:
        dict: A dicionary with the metrics
    """
    
    tn, fp, fn, tp = _confusion_matrix(y_true, y_pred)

    dct = {
        'ACCURACY': metric_accuracy(y_true, y_pred),
        'F1': metric_f1(y_true, y_pred),
        'F1_MICRO': metric_f1_micro(y_true, y_pred),
        'F1_MACRO': metric_f1_macro(y_true, y_pred),
        'PRECISION': metric_precision(y_true, y_pred),
        'RECALL': metric_recall(y_true, y_pred),
        'AUC': metric_auc(y_true, y_pred),
        'KAPPA': metric_kappa(y_true, y_pred),
        'TRUE_POSITIVE': tp,
        'TRUE_NEGATIVE': tn,
        'FALSE_POSITIVE': fp,
        'FALSE_NEGATIVE': fn,
        'POS_PRED_VALUE': metric_positive_predictive_value(y_true, y_pred),
        'NEG_PRED_VALUE': metric_negative_predictive_value(y_true, y_pred),
        'SENSITIVITY': metric_sensitivity(y_true, y_pred),
        'SPECIFICITY': metric_specificity(y_true, y_pred)
    }
    return {k: 0.0 if not v else v for k, v in dct.items()}

def evaluate_regression(y_true: list, y_pred: list, greater_than_zero:bool = True) -> dict:
    """
    Extract metrics from predicted and expected

    Parameters:
        y_true (list): A list of ground truth (correct) target values
        y_pred (list): A list of estimated targets
        greater_than_zero (bool): (Default True) A boolean flag meaning if should consider just values greater than zero.

    Returns:
        dict: A dicionary with the metrics
    """

    y = np.array(list(zip(y_true,y_pred)))

    if greater_than_zero:
        y = y[(y[:,0] > 0) & (y[:,1] > 0)]
        
    y_true, y_pred = y[:,0],y[:,1]

    dct = {
        'MAPE': metrics.mean_absolute_percentage_error(y_true, y_pred),
        'SMAPE': smape(y_true, y_pred),
        'CORRELATION': corr(y_true, y_pred),
        'MAE': metrics.mean_absolute_error(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MSLE': metrics.mean_squared_log_error(y_true, y_pred),
        'MEAN-SE': metrics.mean_squared_error(y_true, y_pred),
        'MEDIAN-SE': metrics.median_absolute_error(y_true, y_pred),
        'R2': metrics.r2_score(y_true, y_pred),
        'EVS': metrics.explained_variance_score(y_true, y_pred),
        'MAXE': metrics.max_error(y_true, y_pred),
        'MPD': metrics.mean_poisson_deviance(y_true, y_pred),
        'MGD': metrics.mean_gamma_deviance(y_true, y_pred),
        'MTD': metrics.mean_tweedie_deviance(y_true, y_pred),
        'P50': quantiles(y_true, y_pred, 0.5),
        'P75': quantiles(y_true, y_pred, 0.75),
        'P90': quantiles(y_true, y_pred, 0.9),
        'P95': quantiles(y_true, y_pred, 0.95),
        'P99': quantiles(y_true, y_pred, 0.99),
        'POVERALL': quantiles(y_true, y_pred, [0.5,0.75,0.9,0.95,0.99]),
    }

    return {k: 0.0 if not v else v for k, v in dct.items()}
