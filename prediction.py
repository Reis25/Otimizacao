from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt
from errno import EEXIST
from os import makedirs, path
from scipy import interp


def predict_model(base, model, features=None, random_state=0, k_folds=10, n_repeats=5):
    ''' Train and test a prediction model with crossvalidation
    
    Parameters
    ----------
    base : tuple (<PANDAS DF>, <PANDAS SERIES>)
        where the first value is a pandas dataframe with non class features of a
        dataset, the second value correspond to class value for each sample.
    model : classification model instance
    features : list
        A binary list of features 1 refers to presence of a feature, 0 refers to
        it's absense
    sampling : sampling class instance
    kfolds : int
    n_repeats : int
    Returns
    -------
    dict[<BASE_NAME>][<MODEL_NAME>]
        [<FEATURES>] : Pandas dataframe
            A dataframe with features of the model prediction
        [<KFOLDS>][<KFOLD>]
            [<Y_TRUE>] : list
                True values of test kfold samples
            [<Y_PRED>] : list
                Predicted values of test kfold samples
            [<IMPORTANCES>] : Pandas dataframe
                Gini importance of features for each kfold
 
    '''

    random = np.random.RandomState(random_state) if random_state is int() else random_state
    rskf = RepeatedStratifiedKFold(n_splits=k_folds, random_state=random, n_repeats=n_repeats)
    # sampling = SMOTE(random_state=random)
    X, y = base
    if features is None:
        features = [1 for val in X[0]]
    selected_features = [True if val == 1 else False for val in features]
    X = X[:, selected_features]
    predictions_dict = {'kfolds': {}}
    start_time = time.time()

    for i, (train, test) in enumerate(rskf.split(X, y)):
        X_train, y_train = (X[train], y[train])
        X_test, y_test = (X[test], y[test])
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        predictions_dict['kfolds']['rep%dkf%d' % (i // n_repeats, i)] = {
            'y_true': y_test,
            'y_pred': y_pred
        }

    classification_time = time.time() - start_time
    return predictions_dict


def metrics_by_prediction(y_true, y_pred):
    ''' Applies metrics to comparisson of predicted and true values
    
    Parameters
    ----------
    y_true : list
        True binary labels or binary label indicators.
    y_pred : list
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions.
    
    Returns
    -------
    dict
        a dict of int, float, list, representing each calculated metric
    '''
    metrics = {}
    y_bin = [1 if feature >= 0.5 else 0 for feature in y_pred]

    accuracy = accuracy_score(y_true, y_bin)
    average_precision = average_precision_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)
    fpr_roc, tpr_roc, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr_roc, tpr_roc)

    tn, fp, fn, tp = conf_matrix.ravel()

    metrics['accuracy'] = accuracy
    metrics['average_precision'] = average_precision
    metrics['f1'] = f1
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp
    metrics['tn'] = tn
    metrics['ppv'] = tp / (tp + fp)
    metrics['tpr'] = tp / (tp + fn)
    metrics['tnr'] = tn / (tn + fp)
    metrics['fpr_roc'] = fpr_roc
    metrics['tpr_roc'] = tpr_roc
    metrics['roc_auc'] = roc_auc

    return metrics


def metrics_by_model(model_pred):
    ''' Organizes the metrics for each kfold and saves it in a file

    Parameters
    ----------
    model_pred : dict
        A dict that follows prediction[<FOLD>][<METRICS>] : int, float, list
    write : bool
        Defines if the metrics should be saved in a file or not
    path : str
    file_name : str

    The path of the file is <PATH>/metrics/<FILE_NAME>.csv
    '''
    metrics_dict = {}
    kfolds_pred = model_pred['kfolds']

    for kfold in kfolds_pred:
        metrics = metrics_by_prediction(kfolds_pred[kfold]['y_true'],
                                        kfolds_pred[kfold]['y_pred'])
        for metric in metrics:
            value = metrics[metric]
            if isinstance(value, int) or isinstance(value, float):
                if metric not in metrics_dict.keys():
                    metrics_dict[metric] = []
                metrics_dict[metric].append(value)

    metrics_dataframe = pd.DataFrame.from_dict(metrics_dict)
    return metrics_dataframe


def plot_roc_auc(model_pred, label, path='', file_name=''):
    ''' Plots the area under roc curve for each kfold and its mean then saves it in a file

    Parameters
    ----------
    model_pred : dict
        A dict that follows prediction[<FOLD>][<METRICS>] : int, float, list
    label : str
        Defines a identification label for the plot
    path : str
    file_name : str
    features_to_show : int
        Max number of features on the plot

    The path of the file is <PATH>/importances_bar/<FILE_NAME>.png
    '''
    file_path = '%s/graphs' % path
    mkdir_p(file_path)
    file_path = '%s/%s.png' % (file_path, file_name)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    kfolds_pred = model_pred['kfolds']

    for i, kfold in enumerate(kfolds_pred):
        metrics = metrics_by_prediction(kfolds_pred[kfold]['y_true'],
                                        kfolds_pred[kfold]['y_pred'])
        fpr, tpr = metrics['fpr_roc'], metrics['tpr_roc']
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics['roc_auc']
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s - ROC Curve' % label)
    plt.legend(loc="lower right")
    plt.savefig(file_path, dpi=250)
    ## Destroy plot so that it wont be overlaid with the next plot
    plt.close()
    ## If you want to show the figure in a window:
    # plt.show()


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise