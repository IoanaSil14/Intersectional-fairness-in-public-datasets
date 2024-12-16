import numpy as np
import scipy
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn import tree, ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, \
    auc, roc_curve, RocCurveDisplay, classification_report, make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from helpers.preprocessing_methods import *
from sklearn.preprocessing import LabelEncoder
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


def split_and_train(data, attributes, target, columns_to_encode=[]):
    y = data.loc[:, target]
    x = data.drop(target, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    x_train_encoded, x_test_encoded = encode_categorical_attributes(x_train,x_test)
    models = {
               "Catboost",
               # "LogisticRegression",
              # "RandomForest",
              #  "DecisionTree"
              }
    y_predicted_dict = {}
    metrics_dict = {}
    for m in models:
        if m== "LogisticRegression":
          x_train_num, x_train_num = scale(x_train_encoded, x_test_encoded)
        x_train_num, x_test_num = one_hot_encode(x_train_encoded, x_test_encoded, columns_to_encode)

        model = choose_model(m, x_train_num, y_train)
        y_predicted = evaluate_model(model, x_train_num, x_test_num, y_train, y_test)
        y_predicted_dict[m] = y_predicted

    del x_train, y_train,x_train_encoded,x_test_encoded,x_train_num # not needed, free up memory
    return x_test, y_test, y_predicted_dict, metrics_dict

def encode_categorical_attributes(x_train,x_test):
    # encode categorical columns to numerical
    x_train_enc=x_train.copy()
    x_test_enc=x_test.copy()
    categorical_attributes = get_categorical_attributes(x_train_enc)
    for col in categorical_attributes:
        le = LabelEncoder()
        x_train_enc[col] = le.fit_transform(x_train_enc[col])
        x_test_enc[col] = le.transform(x_test_enc[col])
        le_map = dict(zip(le.classes_, le.transform(le.classes_)))
        print('Attribute: ' + col)
        print(le_map)
    return x_train_enc, x_test_enc
def scale(x_train, x_test):
    x_train_num = x_train.copy()
    x_test_num = x_test.copy()
    scaler = StandardScaler()
    for col in x_train_num.columns:
        if  x_train_num[col].dtypes != "categorical" and x_train_num[col].dtypes != "object":
            x_train_num[col] = scaler.fit_transform(x_train_num[[col]])
            x_test_num[col] = scaler.transform(x_test_num[[col]])
    return x_train_num, x_test_num

def one_hot_encode(x_train, x_test,cat_attributes):
    if len(cat_attributes) == 0:
        return x_train, x_test
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    x_train_num = one_hot_encoder.fit_transform(x_train[cat_attributes])
    x_test_num = one_hot_encoder.transform(x_test[cat_attributes])
    return x_train_num, x_test_num

def catboost_classifier(x_train, y_train, categorical_features=None):
    model = CatBoostClassifier(custom_loss=[metrics.Accuracy()], random_seed=7, logging_level='Silent', l2_leaf_reg=2,
                               eval_metric='CrossEntropy', early_stopping_rounds=3, depth=2,
                               )
    model.fit(x_train, y_train, plot=False)
    return model

def decision_tree_classifier(x_train, y_train):
    # tree_classifier = tree.DecisionTreeClassifier(min_samples_leaf=50, min_samples_split=2, max_depth=14,
    #                                               criterion='gini', random_state=7)
    #
    # tree_classifier = tree_classifier.fit(x_train, y_train)
    # return tree_classifier

    param_grid = {'max_depth': list(np.arange(3, 15, 1)), 'min_samples_leaf': list(np.arange(1, 200, 15))}

    tree_classifier = tree.DecisionTreeClassifier()
    random_cv = RandomizedSearchCV(tree_classifier, param_grid, n_iter=15,
                                   scoring=make_scorer(f1_score, average='macro'),
                                   n_jobs=-1, cv=5, random_state=7)
    random_cv.fit(x_train, y_train)
    model = random_cv.best_estimator_
    print("Best estimator:",random_cv.best_estimator_)
    return model



def random_forest_classifier(x_train, y_train):
    # forest_classifier = ensemble.RandomForestClassifier(min_samples_split=2, max_depth=14, min_samples_leaf=1,
    #                                                     criterion='gini', n_estimators=500, max_features='sqrt',
    #                                                     random_state=7)
    #
    # forest_classifier = forest_classifier.fit(x_train, y_train)
    # #
    # return forest_classifier
    param_grid = {'max_depth': list(np.arange(3, 15, 1)), 'n_estimators': [100, 200, 300, 400, 500]}

    forest_classifier = ensemble.RandomForestClassifier()
    random_cv = RandomizedSearchCV(forest_classifier, param_grid, n_iter=15,
                                   scoring=make_scorer(f1_score, average='macro'),
                                   n_jobs=-1, cv=5, random_state=7)
    random_cv.fit(x_train, y_train)
    model = random_cv.best_estimator_
    print("Best estimator:",random_cv.best_estimator_)
    return model


def logistic_regression(x_train, y_train):
    param_grid = {'C': list(np.arange(0.01, 1.0, 0.1)), 'penalty': ['l1', 'l2']}
    lr = LogisticRegression(solver='liblinear', random_state=42)
    random_cv = RandomizedSearchCV(lr, param_grid, n_iter=15,
                       scoring=make_scorer(f1_score, average='macro'),
                       n_jobs=-1, cv=5, random_state=7)
    random_cv.fit(x_train, y_train)
    model = random_cv.best_estimator_
    print("Best estimator:",random_cv.best_estimator_)
    return model
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
