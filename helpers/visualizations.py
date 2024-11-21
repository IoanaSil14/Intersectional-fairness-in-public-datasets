import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import display
from sklearn.metrics import classification_report
from aequitas import Audit


def plot_reports(y_predicted_dict, y_true):
    classification_reports = {}

    for model in y_predicted_dict.keys():
        report = classification_report(y_true, y_predicted_dict[model], output_dict=True)
        classification_reports[model] = report

    metrics_to_plot = ['precision', 'recall', 'f1-score']

    # Extract the relevant data for plotting
    plot_data = {}
    for algo, report in classification_reports.items():
        plot_data[algo] = {metric: report['weighted avg'][metric] for metric in metrics_to_plot}

    # Convert the dictionary to a format suitable for plotting
    algorithms = list(plot_data.keys())
    metrics = metrics_to_plot
    values = np.array([[plot_data[algo][metric] for metric in metrics] for algo in algorithms])

    # Number of algorithms and metrics
    num_algorithms = len(algorithms)
    num_metrics = len(metrics)

    # Set up bar plot parameters
    bar_width = 0.2
    index = np.arange(num_metrics)

    # Create the plot
    plt.figure(figsize=(8, 6))

    for i in range(num_algorithms):
        plt.bar(index + i * bar_width, values[i], bar_width, label=algorithms[i])

    # Add labels and titles
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Comparison of Classification Metrics Across Algorithms')
    plt.xticks(index + bar_width, metrics)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_metrics(metrics_dict, attribute):
    # Initialize lists to store the data
    algorithms = list(metrics_dict.keys())  # Extract algorithm names
    metrics = list(metrics_dict.values())  # Extract the list of metrics for each algorithm

    # Separate the metrics into disparate impact and statistical parity
    disparate_impact = [metric[0] for metric in metrics]
    statistical_parity = [metric[1] for metric in metrics]

    # Set up bar plot parameters
    bar_width = 0.35  # Width of the bars
    index = np.arange(len(algorithms))  # X locations for the groups

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the Disparate Impact Ratio
    plt.bar(index, disparate_impact, bar_width, label='Disparate Impact Ratio')

    # Plot the Statistical Parity Difference
    plt.bar(index + bar_width, statistical_parity, bar_width, label='Statistical Parity Difference')

    # Add labels and title
    plt.xlabel('Algorithms')
    plt.ylabel('Values')
    plt.title(f'Disparate Impact Ratio and Statistical Parity Difference for {attribute}')
    plt.xticks(index + bar_width / 2, algorithms)

    # Add legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_attributes(dataset, attributes, target, num_rows, num_cols):
    fontsize = 10
    num_elements = len(attributes)

    fig_width = 7 * num_cols
    fig_height = 7 * num_rows

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), sharey=True,
                           gridspec_kw={'hspace': 0.5})
    ax = ax.flatten()
    plt.subplots_adjust(hspace=1.5)
    sns.set_style("whitegrid")
    for i, attribute in enumerate(attributes):
        g0 = sns.countplot(x=attribute, data=dataset, palette="hls",
                           ax=ax[i], hue=target)  # order=dataset[attribute].value_counts().index)
        g0.set_xlabel(attribute, fontsize=fontsize)
        g0.set_ylabel("Count", fontsize=fontsize)


def plot_audit(dataset, attributes):
    audit = Audit(dataset[['score','label_value']+attributes])
    summary = audit.summary_plot(["tpr","fpr","fnr","tnr","pprev"])
    summary.show()