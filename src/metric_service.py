
import warnings
from collections import Counter
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

def evaluate_binary_classification(y_true, y_pred):
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

def cross_validation_scorers():
    return {
        'ACCURACY': make_scorer(metric_accuracy, greater_is_better=True),
        'F1': make_scorer(metric_f1, greater_is_better=True),
        'F1_MICRO': make_scorer(metric_f1_micro, greater_is_better=True),
        'F1_MACRO': make_scorer(metric_f1_macro, greater_is_better=True),
        'PRECISION': make_scorer(metric_precision, greater_is_better=True),
        'RECALL': make_scorer(metric_recall, greater_is_better=True),
        'AUC': make_scorer(metric_auc, greater_is_better=True),
        'KAPPA': make_scorer(metric_kappa, greater_is_better=True),
        'roc': make_scorer(metric_roc, greater_is_better=True),
        'LOG_LOSS': make_scorer(metric_log_loss, greater_is_better=False),
        'TRUE_POSITIVE': make_scorer(metric_true_positive, greater_is_better=True),
        'TRUE_NEGATIVE': make_scorer(metric_true_negative, greater_is_better=True),
        'FALSE_POSITIVE': make_scorer(metric_false_positive, greater_is_better=False),
        'FALSE_NEGATIVE': make_scorer(metric_false_negative, greater_is_better=False),
        'POS_PRED_VALUE': make_scorer(metric_positive_predictive_value, greater_is_better=True),
        'NEG_PRED_VALUE': make_scorer(metric_negative_predictive_value, greater_is_better=True),
        'SENSITIVITY': make_scorer(metric_sensitivity, greater_is_better=True),
        'SPECIFICITY': make_scorer(metric_specificity, greater_is_better=True)
    }

def evaluate_model(y_true, y_pred):
    """
    Some metrics to evaluate the models.

    Args:
        y_true: a vector (list or array) with the ground truth.
        y_pred: a vector (list or array) with the predictions.

    Returns:
        A dictionary with results of some metrics (check bellow).
    """
    
    tn, fp, fn, tp = _confusion_matrix(y_true, y_pred)

    return {
        'ACCURACY': metric_accuracy(y_true, y_pred),
        'F1': metric_f1(y_true, y_pred),
        'F1_MICRO': metric_f1_micro(y_true, y_pred),
        'F1_MACRO': metric_f1_macro(y_true, y_pred),
        'PRECISION': metric_precision(y_true, y_pred),
        'RECALL': metric_recall(y_true, y_pred),
        'TRUE_POSITIVE': tp,
        'TRUE_NEGATIVE': tn,
        'FALSE_POSITIVE': fp,
        'FALSE_NEGATIVE': fn
    }
