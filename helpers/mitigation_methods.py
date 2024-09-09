import pandas as pd
from aequitas.flow.methods.preprocessing import massaging, label_flipping, data_repairer
from aequitas.flow.methods.postprocessing import group_threshold, balanced_group_threshold
from fairlearn.metrics import MetricFrame
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import (
    MetricFrame,
    true_positive_rate,
    false_positive_rate,
    false_negative_rate,
    selection_rate,
    count,

)
from sklearn.metrics import balanced_accuracy_score

from helpers.training import *

'''
Messaging
'''


def pre_process_massaging(dataset, sensitive_attr, target_class):
    y_m = dataset.loc[:, target_class]
    x_m = dataset.drop(columns=[target_class], axis=1)
    s_m = x_m[sensitive_attr]
    s_m = s_m.astype("category")
    x_m = x_m.drop(columns=[sensitive_attr], axis=1)

    ps = massaging.Massaging()
    ps.fit(x_m, y_m, s_m)
    x_tr_ps, y_tr_ps, s_tr_ps = ps.transform(x_m, y_m, s_m)
    data_transformed = x_tr_ps.copy()
    data_transformed[sensitive_attr] = s_tr_ps.copy()
    data_transformed[target_class] = y_tr_ps.copy()
    return data_transformed


'''
Label flipping
'''


def pre_process_label_flip(dataset, sensitive_attr, target_class):
    y_m = dataset.loc[:, target_class]
    x_m = dataset.drop(columns=[target_class], axis=1)
    s_m = x_m[sensitive_attr]
    s_m = s_m.astype("category")
    x_m = x_m.drop(columns=[sensitive_attr], axis=1)

    ps = label_flipping.LabelFlipping(
        max_flip_rate=0.1, max_depth=3)
    ps.fit(x_m, y_m, s_m)
    x_tr_ps, y_tr_ps, s_tr_ps = ps.transform(x_m, y_m, s_m)
    data_transformed = x_tr_ps.copy()
    data_transformed[sensitive_attr] = s_tr_ps.copy()
    data_transformed[target_class] = y_tr_ps.copy()
    return data_transformed


from aequitas.flow.methods.preprocessing import prevalence_sample

'''
Prevelance sampling 
'''


def pre_process_prev_sampling(dataset, sensitive_attr, target_class):
    y_ps = dataset.loc[:, target_class]
    x_ps = dataset.drop(columns=[target_class], axis=1)
    s_ps = x_ps[sensitive_attr]
    s_ps = s_ps.astype("category")
    x_ps = x_ps.drop(columns=[sensitive_attr], axis=1)

    ps = prevalence_sample.PrevalenceSampling()
    ps.fit(x_ps, y_ps, s_ps)
    x_tr_ps, y_tr_ps, s_tr_ps = ps.transform(x_ps, y_ps, s_ps)
    data_transformed = x_tr_ps.copy()
    data_transformed[sensitive_attr] = s_tr_ps.copy()
    data_transformed[target_class] = y_tr_ps.copy()
    return data_transformed


'''
Data repairer
'''


def pre_process_data_repairer(dataset, sensitive_attr, target_class, columns_to_change):
    y_r = dataset.loc[:, target_class]
    x_r = dataset.drop(columns=[target_class], axis=1)
    s_r = x_r[sensitive_attr]
    s_r = s_r.astype("category")
    x_r = x_r.drop(columns=[sensitive_attr], axis=1)

    dr = data_repairer.DataRepairer(columns=columns_to_change)
    dr.fit(x_r, y_r, s_r)
    x_tr_r, y_tr_r, s_tr_r = dr.transform(x_r, y_r, s_r)
    data_transformed = x_tr_r.copy()
    data_transformed[sensitive_attr] = s_tr_r.copy()
    data_transformed[sensitive_attr] = data_transformed[sensitive_attr].astype(int)
    data_transformed[target_class] = y_tr_r.copy()
    return data_transformed


'''
Group threshold
'''


def post_process_group_threshold_aequitas(data, attribute, target):
    y = data.loc[:, target]
    x = data.drop(target, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    model = choose_model("DecisionTree", x_train, y_train)
    scores_test = model.predict_proba(x_test)[:, 1]
    #print("prob scores", scores_test)
    scores_test = pd.Series(scores_test, index=x_test.index)
    threshold = group_threshold.GroupThreshold(threshold_type="tpr", threshold_value=0.3)
    s_test = x_test[attribute]
    s_train = x_train[attribute]
    threshold.fit(X=x_train, y=y_train,y_hat=scores_test, s=s_test)
    corrected_scores = threshold.transform(x_test, scores_test, s_test)
    x_test[attribute] = s_test.copy()
    calc_metrics(x_test, y_test, corrected_scores, [attribute], target)
    print(f"Accuracy score test corrected:\n{accuracy_score(y_test, corrected_scores):.4f}")

    return corrected_scores


'''
Group threshold: fairlearn
'''


def post_process_group_threshold_fairlearn(data, attribute, target):
    y = data.loc[:, target]
    x = data.drop(target, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    model = choose_model("DecisionTree", x_train, y_train)
    scores_test = model.predict_proba(x_test)[:, 1]
    s_test = x_test[attribute]
    s_train = x_train[attribute]
    threshold = ThresholdOptimizer(
        estimator=model,
        constraints="false_negative_rate_parity",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method='predict_proba'
    )
    threshold.fit(X=x_train, y=y_train, sensitive_features=s_train)
    metrics_dict = {
        "selection_rate": selection_rate,
        "false_negative_rate": false_negative_rate,
        "balanced_accuracy": balanced_accuracy_score,
    }
    y_pred_postprocess = threshold.predict(x_test, sensitive_features=s_test)
    metricframe_postprocess = MetricFrame(
        metrics=metrics_dict,
        y_true=y_test,
        y_pred=y_pred_postprocess,
        sensitive_features=s_test
    )
    print(f"Accuracy score test corrected:\n{accuracy_score(y_test, y_pred_postprocess):.4f}")

    return y_pred_postprocess, metricframe_postprocess
