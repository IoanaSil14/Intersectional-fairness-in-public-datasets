import numpy as np
import scipy
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn import tree, ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, \
    auc, roc_curve, RocCurveDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV


def split_and_train(data, attributes, target):
    y = data.loc[:, target]
    x = data.drop(target, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    models = {
             # "Catboost",
            "LogisticRegression",
            # "RandomForest",
              # "DecisionTree"
              }
    y_predicted_dict = {}
    metrics_dict = {}
    for m in models:
        model = choose_model(m, x_train, y_train)
        y_predicted = evaluate_model(model, x_train, x_test, y_train, y_test)
        # attribute_metrics = calc_metrics(x_test=x_test, y_test=y_test, y_predicted=y_predicted, attributes=attributes,
        #                                  target=target)
        # metrics_dict[m] = attribute_metrics
        y_predicted_dict[m] = y_predicted
        # calc_avrg(metrics_dict[m])
        print(f"Classification report for model: {model} : \n {classification_report(y_test, y_predicted)}")
        # plot_roc_curve(y_true=y_test, y_pred=y_predicted, model_name=m)
    del x_train, y_train # not needed, free up memory
    return x_test, y_test, y_predicted_dict, metrics_dict


def calc_avrg(metrics_dict_model):
    fnr_values = [values[4] for values in metrics_dict_model.values()]
    average_fnr = sum(fnr_values) / len(fnr_values)

    fpr_values = [values[5] for values in metrics_dict_model.values()]
    average_fpr = sum(fpr_values) / len(fpr_values)

    for_values = [values[6] for values in metrics_dict_model.values()]
    average_for = sum(for_values) / len(for_values)

    fdr_values = [values[7] for values in metrics_dict_model.values()]
    average_fpr = sum(fdr_values) / len(fdr_values)

    print(f"Average FNR: %.3f" % average_fnr)
    print(f"Average FPR: %.3f" % average_fpr)
    print(f"Average FOR: %.3f" % average_for)
    print(f"Average FDR: %.3f" % average_fpr)


def create_train_val_test_data(df, target_column_name):
    y = df.loc[:, target_column_name]
    x = df.drop(target_column_name, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return x_train, y_train, x_test, y_test


def choose_model(model, x_train, y_train, categorical_features=None):
    print("---------- Model name: ", model, "----------\n")
    if model == 'Catboost':
        return catboost_classifier(x_train, y_train, categorical_features)
    elif model == 'LogisticRegression':
        return logistic_regression(x_train, y_train)
    elif model == 'RandomForest':
        return random_forest_classifier(x_train, y_train)
    elif model == 'DecisionTree':
        return decision_tree_classifier(x_train, y_train)
    print("Not known model")
    return 0


def catboost_classifier(x_train, y_train, categorical_features=None):
    model = CatBoostClassifier(custom_loss=[metrics.Accuracy()], random_seed=7, logging_level='Silent', l2_leaf_reg=2,
                               eval_metric='CrossEntropy', early_stopping_rounds=3, depth=2,
                               )
    model.fit(x_train, y_train, cat_features=categorical_features, plot=False)
    return model


def decision_tree_classifier(x_train, y_train):
    tree_classifier = tree.DecisionTreeClassifier(min_samples_leaf=50, min_samples_split=2, max_depth=14,
                                                  criterion='gini', random_state=7)

    tree_classifier = tree_classifier.fit(x_train, y_train)
    return tree_classifier

    # param_grid = {  'max_depth': [5, 10, 15, 2],
    #                 'min_samples_split': [2, 10, 20],
    #                 'min_samples_leaf': [1, 5, 10],
    #                 'max_features': ['sqrt', 'log2']
    #               }
    #
    # tree_classifier = tree.DecisionTreeClassifier(criterion="gini")
    # clf = GridSearchCV(tree_classifier, param_grid,cv=5)
    # grid_search = clf.fit(x_train, y_train)
    # print("Best parameters:", grid_search.best_params_)
    # print("Best cross-validation accuracy:", grid_search.best_score_)
    # return grid_search



def random_forest_classifier(x_train, y_train):
    forest_classifier = ensemble.RandomForestClassifier(min_samples_split=2, max_depth=14, min_samples_leaf=1,
                                                        criterion='gini', n_estimators=500, max_features='sqrt',
                                                        random_state=7)

    forest_classifier = forest_classifier.fit(x_train, y_train)

    return forest_classifier
    # param_grid = {'n_estimators': [50, 100, 150],
    #         'max_depth': [10, 20, 5],
    #         'min_samples_split': [2, 5, 10]
    #     }
    #
    # forest_classifier = ensemble.RandomForestClassifier(criterion="gini", max_features='sqrt')
    # clf = GridSearchCV(forest_classifier, param_grid, scoring="accuracy", cv=5, n_jobs=-1)
    # grid_search = clf.fit(x_train, y_train)
    # print("Best parameters:", grid_search.best_params_)
    # print("Best cross-validation accuracy:", grid_search.best_score_)
    # return grid_search


def logistic_regression(x_train, y_train):
    lr = LogisticRegression(max_iter=100, solver='liblinear', random_state=7, fit_intercept=True, C=2.1, penalty='l1',intercept_scaling=1)
    lr.fit(x_train, y_train)
    return lr
    #
    # lr = LogisticRegression(max_iter=100, random_state=42)
    # param_grid = {
    #     'C': [0.01, 0.1, 0.2, 1, 10, 100],
    #     'penalty': ['l1', 'l2', 'elasticnet'],
    #     'solver': ['saga','liblinear']
    # }
    # clf = GridSearchCV(lr, param_grid,cv=5)
    # grid_search = clf.fit(x_train, y_train)
    # print("Best parameters:", grid_search.best_params_)
    # print("Best cross-validation accuracy:", grid_search.best_score_)
    # return grid_search

def evaluate_model(model, x_train, x_test, y_train, y_test):
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)

    print(f"Accuracy score training:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Accuracy score test:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print("----------------------------------------\n")
    return y_test_pred


def plot_roc_curve(y_true, y_pred, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name)
    display.plot()
    plt.show()


def get_majority_classes(series):
    """
    Majority classes are considered to be those with higher empirical_distribution
    in comparisson to related attribute equiprobability
    """
    if series.dtype == object or series.dtype == 'category' or series.dtype == int:
        unique_classes, class_counts = np.unique(series, return_counts=True)
        empirical_distribution = class_counts / class_counts.sum()

        eqp = 1 / len(class_counts)  # equiprobability

        result = {unique_classes[i]: x for i, x in enumerate(empirical_distribution) if x > eqp}
        result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
        return list(
            result.keys())
    else:
        return np.nan


def get_minority_classes(series):
    """
    Minority classes are considered to be those with lower empirical_distribution
    in comparisson to related attribute equiprobability
    """
    if series.dtype == object or series.dtype == 'category' or series.dtype == int:
        unique_classes, class_counts = np.unique(series, return_counts=True)
        empirical_distribution = class_counts / class_counts.sum()

        eqp = 1 / len(class_counts)  # equiprobability

        result = {unique_classes[i]: x for i, x in enumerate(empirical_distribution) if x < eqp}
        result = dict(sorted(result.items(), key=lambda item: item[1]))
        return list(result.keys())
    else:
        return np.nan


def get_attribute_label_encoder_mapping(df, attribute):
    le = LabelEncoder()
    le.fit(df[attribute])
    return dict(zip(le.classes_, le.transform(le.classes_)))


def normalize_values(value):
    if value > 2:
        return 0
    if 0 <= value <= 1:
        return value
    if 1 < value <= 2:
        return 2 - value


def calc_metrics(x_test, y_test, y_predicted, attributes, target):
    attribute_metric = {}
    for attribute in attributes:
        print("attribute:",attribute)
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

        bld_test = x_test.copy()
        bld_test[target] = y_test
        test_data_bld = BinaryLabelDataset(df=bld_test, label_names=[target],
                                           protected_attribute_names=[attribute],
                                           favorable_label=1,
                                           unfavorable_label=0)

        classified_ds = x_test.copy()
        classified_ds[target] = y_predicted
        classified_ds_bld = BinaryLabelDataset(df=classified_ds, label_names=[target],
                                               protected_attribute_names=[attribute],
                                               favorable_label=1,
                                               unfavorable_label=0)

        c_metric = ClassificationMetric(test_data_bld, classified_ds_bld, privileged_groups=privileged_groups,
                                        unprivileged_groups=unprivileged_groups)
        normalized_dir = normalize_values(c_metric.disparate_impact())

        dir = c_metric.disparate_impact()
        spd = c_metric.statistical_parity_difference()

        eq_ods_diff = c_metric.equalized_odds_difference()
        eq_opportunity_diff = c_metric.equal_opportunity_difference()

        fnr = c_metric.false_negative_rate_ratio()

        fpr = c_metric.false_positive_rate_ratio()
        for_ =c_metric.false_omission_rate_ratio()
        fdr = c_metric.false_discovery_rate_ratio()

        print("---------- Metrics --------\n")

        print(f"Disparate Impact Ratio for {attribute}: %.3f" % dir)

        print(f"Statistical Parity Difference for {attribute}: %.3f" % spd)
        print(f"equalized opportunity difference for {attribute}: %.3f" % eq_opportunity_diff)

        print(f"equalized ods difference for {attribute}: %.3f" % eq_ods_diff)

        print(f" FNR ratio for {attribute}: %.3f" % fnr)
        print(f" FNR privileged for {attribute}: %.3f" % c_metric.false_negative_rate())
        print(f" FNR unpriv for {attribute}: %.3f" %  c_metric.false_negative_rate(privileged=False))

        print(f" FPR ratio for {attribute}: %.3f" % fpr)
        print(f" FOR ratio for {attribute}: %.3f" % for_)
        print(f" FDR ratio for {attribute}: %.3f" % fdr)


        attribute_metric[attribute] = [dir, spd,eq_opportunity_diff, eq_ods_diff, fnr, fpr, for_, fdr]

        print("\n")

    return attribute_metric
