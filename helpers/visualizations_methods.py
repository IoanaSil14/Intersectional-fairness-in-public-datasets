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


def plot_accuracies():
    methods = [
        "Original",
        "Massaging", "Prev.Sampler", "Data Repairer",
        "Label Flip", "Fairlearn", "GerryFair",
        "Treshold Optimizer", "Group Threshold", "Eq.odds"
    ]
    # accuracy = [0.75,0.82,0.71,0.75,0.78,0.7,0.78,0.65,0.51,0.72]  # german ds
    # accuracy = [0.88,0.97,0.91,0.88,0.98,0.81,0.88,0.72,0.77,0.88]  # law ds
    # accuracy = [0.79,0.95,0.76,0.77,0.83,0.74,0.67,0.52,0.71,0.72]  # acs ds
    accuracy = [0.63, 0.66, 0.63, 0.63, 0.69, 0.6, 0.61, 0.63, 0.55, 0.63]  # diab ds

    # Create a DataFrame
    df = pd.DataFrame({
        "Methods": methods,
        "Accuracy": accuracy
    })

    # Plot settings
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Methods", y="Accuracy", data=df, palette="viridis")

    # Add accuracy values to bars
    for i, value in enumerate(df["Accuracy"]):
        plt.text(i, value + 0.02, f"{value:.2f}", ha='center', fontsize=10, color='black')

    # Add labels and title
    plt.title("Accuracy Across Original Dataset and Bias Mitigation Methods (trained on Catboost)", fontsize=14)
    plt.xlabel("Bias Mitigation Methods", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)

    # Show plot
    plt.tight_layout()
    plt.show()


from matplotlib.colors import ListedColormap
def plot_avg_disparties():
    categories = ['FNR','FOR']
    original = [1.14,1]
    intersectional = [1.8,0.66]

    # Bar positions
    x = np.arange(len(categories))  # the label locations
    width = 0.35  # the width of the bars

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, original, width, label='Original', color='skyblue')
    bars2 = ax.bar(x + width/2, intersectional, width, label='Intersectional', color='indianred')

    # Add text for labels, title, and axes ticks
    ax.set_ylabel('Disparity')
    ax.set_title('Comparison of FNR and FOR Disparities')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)

    fairness_low = 0.8
    fairness_high = 1.25
    line1 = ax.axhline(fairness_low, color='red', linestyle='--', label='Fairness Limit (Low)')
    line2 = ax.axhline(fairness_high, color='red', linestyle='--', label='Fairness Limit (High)')

    # Annotate bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset text above bar
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Add fairness limits to legend
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([line1, line2])  # Add lines to legend
    ax.legend(handles, labels)

    plt.ylim(0, max(max(original), max(intersectional)) + 0.5)  # Adjust y-axis for better spacing
    plt.tight_layout()
    plt.show()