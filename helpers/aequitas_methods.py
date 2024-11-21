import pandas as pd
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
import aequitas.plot as ap
from helpers.training import get_majority_classes, get_minority_classes
from IPython.display import display

from aequitas.plotting import Plot
from matplotlib import pyplot as plt
import numpy as np
"""
Calculates the confusion matrix for each defined group in "attr_cols" and
the statistical metrics per group.
"""

def get_priv_and_unpriv_groups(protected_attribute,data):
    priv_groups = {}
    unpriv_groups = {}

    for attribute in protected_attribute:
        min_classes = get_minority_classes(data[attribute])
        unprivileged_groups = [ v for v in min_classes]

        maj_classes = get_majority_classes(data[attribute])
        if np.size(maj_classes) > 1:
            privileged_groups = [maj_classes[0]]
            maj_classes.remove(maj_classes[0])  # keep only one maj class
            unprivileged_groups = unprivileged_groups + [ v for v in maj_classes]
        else:
            privileged_groups = [v for v in maj_classes]

        priv_groups.update({attribute: privileged_groups})
        unpriv_groups.update({attribute: unprivileged_groups})

    return priv_groups, unpriv_groups

def init_group_and_get_metrics(df, attr_cols):
    g = Group()
    xtab, _ = g.get_crosstabs(df, attr_cols=attr_cols)
    #print(xtab)
    # get the metrics
    absolute_metrics = g.list_absolute_metrics(xtab)

    return xtab, absolute_metrics


"""
Calculates the disparities between a reference group {ref_gorup_dict} and every other group
from the method init_group_and_plot_metrics
"""


def init_bias_and_print_metrics(xtab, df_no_features, dict):
    b = Bias()
    ## calculate disparities
    bdf = b.get_disparity_predefined_groups(xtab, original_df=df_no_features,
                                            ref_groups_dict=dict,
                                            alpha=0.05, check_significance=True,
                                            mask_significance=True)
    print(b.list_significance(df_no_features))

    return b, bdf


"""
Calculates the fairness based on the disparities.
"""


def init_fairness_and_print_results(bdf):
    f = Fairness()
    fdf = f.get_group_value_fairness(bdf)
    parity_detrminations = f.list_parities(fdf)
    #print(fdf[['attribute_name', 'attribute_value'] + absolute_metrics + b.list_disparities(fdf) + parity_detrminations])
    gaf = f.get_group_attribute_fairness(fdf)
    overall_fairness = f.get_overall_fairness(fdf)

    return gaf, fdf, overall_fairness


def plot_disparities(bdf, metrics, attribute, disparity_tolerance):
    ap.absolute(bdf, metrics, attribute, fairness_threshold=disparity_tolerance)


def calc_fairness_report(xt, yt, y_pred, target_class, sensitive_attributes, list_of_disparities, priv,display_disp=False):
    df_test = xt.copy()
    y_test_df = yt.to_frame()
    df_test['label_value'] = y_test_df[target_class]
    df_test['score'] = y_pred

    df_test[sensitive_attributes] = df_test[sensitive_attributes].astype(str)

    df_test_no_features = df_test[["score", "label_value"] + sensitive_attributes]
    """
    1. Calculate metrics for each group within the sensitive attributes
    """
    xtab, absolute_metrics = init_group_and_get_metrics(df_test_no_features, attr_cols=sensitive_attributes)
    display(xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(2))

    reference_groups = {}

    for attr in sensitive_attributes:
        result = get_majority_classes(df_test[attr])
        reference_groups[attr] = result[0]
        print("Majority class for: ", attr, " is:", result[0])

    """
    2. Calculate disparity metrics for each minority group wrt the majority group.
    """
    b, bdf = init_bias_and_print_metrics(xtab=xtab, dict=reference_groups, df_no_features=df_test_no_features)
    display(bdf[['attribute_name', 'attribute_value'] +
                  b.list_disparities(bdf)].round(2))
    disparities_pd = pd.DataFrame(bdf[['attribute_name', 'attribute_value'] +
                  b.list_disparities(bdf) + b.list_significance(bdf)].round(2))
    avg_disparities = calculate_averages_for_disparities(disparity_df=disparities_pd,list_of_disparities=list_of_disparities, protected_attributes=sensitive_attributes,priv=priv)
    for disparity in list_of_disparities:
        print(f"Overall average for {disparity}:  %.3f" % (avg_disparities[disparity].mean()))
    """
    3. Check if parity is met. By default, an attribute satisfies the parity if the metric value for the minority group lies between [0.8, 1.2]
    """
    gaf, fdf, overall_fairness = init_fairness_and_print_results(bdf)
    if display_disp:
        aqp = Plot()
        fg = aqp.plot_fairness_group_all(fdf, ncols=5, metrics="all")
    display(gaf)


    return df_test_no_features


def calculate_averages_for_disparities(disparity_df,list_of_disparities,protected_attributes,priv):

    averages = {}
    protected_attributes = list(priv.keys())
    for attr in protected_attributes:
        # Filter data for the current attribute
        filtered_data = disparity_df[disparity_df['attribute_name'] == attr]

        # Exclude rows where the attribute_value is in the privileged group
        filtered_data = filtered_data[filtered_data["attribute_value"]!=str(priv[attr][0])]

        # Handle each metric
        for disparity in list_of_disparities:


            # Calculate average for the metric
            metric_avg = filtered_data[disparity].mean()

            # Store results in a dictionary
            if disparity not in averages:
                averages[disparity] = {}
            averages[disparity][attr] = metric_avg
    averages_df = pd.DataFrame(averages)

    print(averages_df)
    return averages_df