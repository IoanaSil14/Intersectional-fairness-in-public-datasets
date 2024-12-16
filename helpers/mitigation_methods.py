import pandas as pd
from aequitas.flow.methods.inprocessing import FairlearnClassifier
from aequitas.flow.methods.preprocessing import massaging, label_flipping, data_repairer, prevalence_sample
from aequitas.flow.methods.postprocessing import group_threshold, balanced_group_threshold
from fairlearn.metrics import MetricFrame
from fairlearn.postprocessing import ThresholdOptimizer
from aif360.datasets import StandardDataset
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.inprocessing import GerryFairClassifier
from fairlearn.metrics import (
    MetricFrame,
    true_positive_rate,
    false_positive_rate,
    false_negative_rate,
    selection_rate,
    count,
)
from sklearn import tree, ensemble

from sklearn.metrics import balanced_accuracy_score, classification_report
from helpers.training_methods import *
from helpers.aequitas_methods import calc_fairness_report


'''
Messaging
'''
def prep_massaging(df, protected_attributes, target):
    data_to_transform = df.copy()
    for attr in protected_attributes:
        data_transformed_mess = pre_process_massaging(data_to_transform, attr, target)
        data_to_transform = data_transformed_mess.copy()
    return data_to_transform


def pre_process_massaging(dataset, sensitive_attr, target_class):
    y_m = dataset.loc[:, target_class]
    x_m = dataset.drop(columns=[target_class], axis=1)
    s_m = x_m[sensitive_attr]
    s_m = s_m.astype("category")
    x_m = x_m.drop(columns=[sensitive_attr], axis=1)

    ms = massaging.Massaging()
    ms.fit(x_m, y_m, s_m)
    x_tr_ms, y_tr_ms, s_tr_ms = ms.transform(x_m, y_m, s_m)
    data_transformed = x_tr_ms.copy()
    data_transformed[sensitive_attr] = s_tr_ms.copy()
    data_transformed[target_class] = y_tr_ms.copy()
    return data_transformed


'''
Label flipping
'''
def prep_label_flipping(df, proetcted_attributes, target):
    # transform data
    data_to_transform = df.copy()
    columns_to_change = df.columns.difference(proetcted_attributes).tolist()
    columns_to_change.remove(target)

    for attr in proetcted_attributes:
        data_transformed_lf = pre_process_label_flip(data_to_transform, attr, target)
        data_to_transform = data_transformed_lf.copy()
    return data_to_transform

def pre_process_label_flip(dataset, sensitive_attr, target_class):
    y_lf = dataset.loc[:, target_class]
    x_lf = dataset.drop(columns=[target_class], axis=1)
    s_lf = x_lf[sensitive_attr]
    s_lf = s_lf.astype("category")
    x_lf = x_lf.drop(columns=[sensitive_attr], axis=1)

    lf = label_flipping.LabelFlipping(
        max_flip_rate=0.2)
    lf.fit(x_lf, y_lf, s_lf)
    x_tr_lf, y_tr_lf, s_tr_lf = lf.transform(x_lf, y_lf, s_lf)
    data_transformed = x_tr_lf.copy()
    data_transformed[sensitive_attr] = s_tr_lf.copy()
    data_transformed[target_class] = y_tr_lf.copy()
    return data_transformed



'''
Prevelance sampling 
'''

def prep_prev_sampling(df, protected_attributes, target):
    data_to_transform = df.copy()
    for attr in protected_attributes:
        data_transformed_ps = pre_process_prev_sampling(data_to_transform, attr, target)
        data_to_transform = data_transformed_ps.copy()
    return data_to_transform


def pre_process_prev_sampling(dataset, sensitive_attr, target_class, strategy="oversample"):
    y_ps = dataset.loc[:, target_class]
    x_ps = dataset.drop(columns=[target_class], axis=1)
    s_ps = x_ps[sensitive_attr]
    s_ps = s_ps.astype("category")
    x_ps = x_ps.drop(columns=[sensitive_attr], axis=1)

    ps = prevalence_sample.PrevalenceSampling(strategy=strategy)
    ps.fit(x_ps, y_ps, s_ps)
    x_tr_ps, y_tr_ps, s_tr_ps = ps.transform(x_ps, y_ps, s_ps)
    data_transformed = x_tr_ps.copy()
    data_transformed[sensitive_attr] = s_tr_ps.copy()
    data_transformed[target_class] = y_tr_ps.copy()
    return data_transformed


'''
Data repairer
'''

def prep_data_repairer(df, protected_attributes, target):
    data_to_transform = df.copy()
    columns_to_change = df.columns.difference(protected_attributes).tolist()
    columns_to_change.remove(target)

    for attr in protected_attributes:
        data_transformed_dr = pre_process_data_repairer(data_to_transform, attr, target, columns_to_change)
        data_transformed_dr[attr] = data_transformed_dr[attr].astype(int)
        data_to_transform = data_transformed_dr.copy()
    return data_to_transform

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
Fairlearn classifier 
'''
def train_with_fairlearn(data, attribute, model, metrics_dict, target):
    y = data.loc[:, target]
    x = data.drop(target, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    s_train = x_train[attribute]
    x_train = x_train.drop(columns=[attribute], axis=1)
    s_test = x_test[attribute]
    x_test = x_test.drop(columns=[attribute], axis=1)

    fairlearn_clf = FairlearnClassifier(estimator=model, constraint="fairlearn.reductions.EqualizedOdds",
                                        reduction='fairlearn.reductions.ExponentiatedGradient')

    fairlearn_clf.fit(x_train, y_train, s_train)
    y_train_pred = fairlearn_clf.predict_proba(x_train, s_train).astype(int)
    y_test_pred = fairlearn_clf.predict_proba(x_test, s_test).astype(int)
    x_test.insert(len(x_test.columns) - 1, attribute, s_test, True)  # insert back the attribute
    print("Model:\n", model)
    print(f"Accuracy score training:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Accuracy score test:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Classification report for model: {model} : \n {classification_report(y_test, y_test_pred)}")

    return x_test, y_test, y_test_pred, metrics_dict

"""
Gerry fair
"""
def in_process_gerryfair(dataframe, target, protected_attributes, favourable_classes, train_size):
    standard_df = StandardDataset(df=dataframe, label_name=target, favorable_classes=[1],
                                                   protected_attribute_names=protected_attributes,
                                                   privileged_classes=favourable_classes)

    gerryfair_class = GerryFairClassifier(C=10, printflag=False, heatmapflag=False, heatmap_iter=10, heatmap_path='.', max_iters=10, gamma=0.01, fairness_def='FN')

    train, test = standard_df.split([train_size])
    test_df, _ = test.convert_to_dataframe()
    train_df, _ = train.convert_to_dataframe()

    gerryfair_class.fit(train, early_termination=True)
    y_gf_predicted = gerryfair_class.predict(test)
    test_pred, _ =  y_gf_predicted.convert_to_dataframe() #return the test data but as labels are the corrected predicitons

    y_train = train_df.loc[:, target]
    x_train = train_df.drop(target, axis=1)

    y_test = test_df.loc[:, target]
    x_test = test_df.drop(target, axis=1)

    y_pred = test_pred.loc[:, target]
    x_test_pred = test_pred.drop(target, axis=1)

    for attr in protected_attributes:
        x_test_pred[attr] = test_pred[attr].astype(int)

    print(f"Accuracy score test:\n{accuracy_score(y_test, y_pred):.4f}")
    return x_test, y_test, y_pred

'''
Group threshold
'''
def post_process_group_threshold_aequitas(data, attribute, target,list_of_disparities,priv,model,columns_to_encode=[]):
    print(priv)
    y = data.loc[:, target]
    x = data.drop(target, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    model = choose_model(model, x_train, y_train,columns_to_encode)
    scores_train = model.predict_proba(x_train)[:, 1]
    scores_test = model.predict_proba(x_test)[:, 1]
    #print("prob scores", scores_test)
    scores_train = pd.Series(scores_train, index=x_train.index)
    scores_test = pd.Series(scores_test, index=x_test.index)
    threshold = balanced_group_threshold.BalancedGroupThreshold(threshold_type="fpr", threshold_value=0.6,
                                                                fairness_metric="fpr")
    s_test = x_test[attribute]
    s_train = x_train[attribute]
    threshold.fit(X=x_train, y=y_train, y_hat=scores_train, s=s_train)
    corrected_scores = threshold.transform(x_test, scores_test, s_test)
    x_test[attribute] = s_test.copy()

    print(f"Accuracy score test corrected:\n{accuracy_score(y_test, corrected_scores):.4f}")
    df_test_tr= calc_fairness_report(x_test, y_test, corrected_scores, target, [attribute], list_of_disparities,priv, display_disp=True)
    return df_test_tr


'''
Group threshold: fairlearn
'''
def post_process_group_threshold_fairlearn(data, attributes, target,list_of_disparities,priv,model,to_encode):
    y = data.loc[:, target]
    x = data.drop(target, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    model = choose_model(model, x_train, y_train,to_encode)
    scores_test = model.predict_proba(x_test)[:, 1]
    s_test = x_test[attributes]
    s_train = x_train[attributes]
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

    df_test_tr = calc_fairness_report(x_test, y_test, y_pred_postprocess, target, attributes, list_of_disparities,priv, display_disp=True)
    return df_test_tr


def post_process_eq_ods(data, attributes,target,list_of_disparities,priv, model,to_encode):
    y = data.loc[:, target]
    x = data.drop(target, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    model = choose_model(model, x_train, y_train,[])
    y_pred_test = model.predict(x_test)

    bld_test = x_test.copy()
    bld_test[target] = y_test
    test_data_bld = BinaryLabelDataset(df=bld_test, label_names=[target],
                                       protected_attribute_names=attributes,
                                       favorable_label=1,
                                       unfavorable_label=0)
    bld_test_pred = x_test.copy()
    bld_test_pred[target] = y_pred_test


    test_data_bld_pred = BinaryLabelDataset(df=bld_test_pred, label_names=[target],
                                       protected_attribute_names=attributes,
                                       favorable_label=1,
                                       unfavorable_label=0)

    for attribute in attributes:
        min_classes = get_minority_classes(x_test[attribute])
        unprivileged_groups = [{attribute: v} for v in min_classes]

        maj_classes = get_majority_classes(x_test[attribute])
        if np.size(maj_classes) > 1:
            privileged_groups = [{attribute: maj_classes[0]}]
            maj_classes.remove(maj_classes[0])  #keep only one maj class
            unprivileged_groups = unprivileged_groups + [{attribute: v} for v in maj_classes]
        else:
            privileged_groups = [{attribute: v} for v in maj_classes]

        print("Privileged groups: ", privileged_groups)
        print("Unprivileged groups: ", unprivileged_groups)


        eq_ods_post_process = EqOddsPostprocessing(unprivileged_groups, privileged_groups)
        scores_modified = eq_ods_post_process.fit_predict(test_data_bld, test_data_bld_pred)
        print(f"Accuracy score test corrected:\n{accuracy_score(y_test, scores_modified.labels):.4f}")
        # metrics = calc_metrics(x_test=x_test, y_test=y_test, y_predicted= scores_modified.labels, attributes=attributes,
        #              target=target)
        # calc_avrg(metrics)
        df_test_eq = calc_fairness_report(x_test, y_test, scores_modified.labels, target, attributes,list_of_disparities,priv, display_disp=True)
        return df_test_eq


