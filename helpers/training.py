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
    models = {"Catboost",
              "LogisticRegression",
              "RandomForest",
              "DecisionTree"
              }
    y_predicted_dict = {}
    metrics_dict = {}
    for m in models:
        model = choose_model(m, x_train, y_train)
        y_predicted = evaluate_model(model, x_train, x_test, y_train, y_test)
        attribute_metrics = calc_metrics(x_test=x_test, y_test=y_test, y_predicted=y_predicted, attributes=attributes,
                                         target=target)
        metrics_dict[m] = attribute_metrics
        y_predicted_dict[m] = y_predicted
        print(f"Classification report for model: {model} : \n {classification_report(y_test, y_predicted)}")
        # plot_roc_curve(y_true=y_test, y_pred=y_predicted, model_name=m)
    return x_test, y_test, y_predicted_dict, metrics_dict


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
    # tree_classifier = tree.DecisionTreeClassifier(min_samples_leaf=50, min_samples_split=2, max_depth=12,
    #                                               criterion='gini', splitter='best', random_state=42)
    #
    # tree_classifier = tree_classifier.fit(x_train, y_train)
    # return tree_classifier

    param_grid = {'min_samples_split': [2,5,10],
                  'min_samples_leaf': [1,2,5],
                  'max_depth': [5,10,15,20],
                  }


    tree_classifier = tree.DecisionTreeClassifier(criterion="gini")
    clf = RandomizedSearchCV(lr, params)
    random_search = clf.fit(x_train, y_train)
    best_model = random_search.best_estimator_
    print(best_model)
    return random_search



def random_forest_classifier(x_train, y_train):
    # forest_classifier = ensemble.RandomForestClassifier(min_samples_split=4, max_depth=10, min_samples_leaf=5,
    #                                                     criterion='gini', n_estimators=500, max_features='sqrt',
    #                                                     random_state=42)
    #
    # forest_classifier = forest_classifier.fit(x_train, y_train)
    #
    # return forest_classifier
    param_grid = {'min_samples_split': [2,5,10],
                  'min_samples_leaf': [1,2,5],
                  'max_depth': [5,10,15,20],
                  }

    forest_classifier = ensemble.RandomForestClassifierc(criterion="Gini", max_features="sqrt", n_estimators=100)
    clf = RandomizedSearchCV(lr, params)
    random_search = clf.fit(x_train, y_train)
    best_model = random_search.best_estimator_
    print(best_model)
    return random_search


def logistic_regression(x_train, y_train):
    lr = LogisticRegression(max_iter=100, solver='liblinear', random_state=42, fit_intercept=True)
    params = {'C': scipy.stats.uniform(loc=0, scale=4), 'penalty': ['l1', 'l2']}
    clf = RandomizedSearchCV(lr, params)
    random_search = clf.fit(x_train, y_train)
    best_model = random_search.best_estimator_
    print(best_model)
    return random_search


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


def normalize_dir(value):
    if value > 2:
        return 0
    if 0 <= value <= 1:
        return value
    if 1 < value <= 2:
        return 2 - value


def calc_metrics(x_test, y_test, y_predicted, attributes, target):
    attribute_metric = {}
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
        normalized_dir = normalize_dir(c_metric.disparate_impact())

        dir = c_metric.disparate_impact()
        spd = c_metric.statistical_parity_difference()

        print(f"Disparate Impact Ratio for {attribute}: %.3f" % dir)
        if normalized_dir is not None:
            print(f"Normalized Disparate Impact Ratio for {attribute}: %.3f" % normalized_dir)
        print(f"Statistical Parity Difference for {attribute}: %.3f" % spd)
        attribute_metric[attribute] = [dir, normalized_dir, spd]

        print("\n")

    return attribute_metric
