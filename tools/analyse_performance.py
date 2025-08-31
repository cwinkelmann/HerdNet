"""
Helper Script to find the best epoch in a wandb run



"""

import pandas as pd

import numpy as np
from wandb.apis.public import Run
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def get_history(run: Run) -> pd.DataFrame:
    print(f"Run: {run.name}")
    print(f"Config: {run.config}")
    history = run.scan_history()

    selected_keys = {"precision", "f5_score", "recall", "f1_score"}  # Convert to set
    df = pd.DataFrame([row for row in history if selected_keys & row.keys()])  # Set intersection



    # Assuming your data is in a DataFrame called 'df'
    def compact_epochs(df):
        # Group by epoch and aggregate, taking the first non-NaN value for each column
        return df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan).reset_index()

    compacted_df = compact_epochs(df)
    return compacted_df

def get_best_epoch(df: pd.DataFrame, metric: str = "f1_score") -> pd.Series:
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame columns.")

    best_epoch_row = df.loc[df[metric].idxmax()]
    return best_epoch_row


def plot_metric_over_epochs(df, main_metric="f1_score", aux_metrics=None, figsize=(12, 7),
                            style="whitegrid", palette="husl", show_best=True, smooth=False,
                            title_suffix=None):
    """
    Create a beautiful plot of metrics over epochs using seaborn.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'epoch' column and the specified metrics
    main_metric : str
        Name of the primary metric column to plot (highlighted)
    aux_metrics : list or None
        List of auxiliary metric names to plot with reduced alpha
    figsize : tuple
        Figure size (width, height)
    style : str
        Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
    palette : str
        Color palette name
    show_best : bool
        Whether to highlight the best value for main metric
    smooth : bool
        Whether to add a smoothed trend line for main metric
    title_suffix : str or None
        Custom title suffix. Use {main_metric} as placeholder for metric name.
        Default: "{main_metric} Over Training Epochs"
    """

    # Validate main metric
    if main_metric not in df.columns:
        raise ValueError(f"Main metric '{main_metric}' not found in DataFrame columns: {list(df.columns)}")

    # Validate auxiliary metrics
    if aux_metrics:
        missing_aux = [m for m in aux_metrics if m not in df.columns]
        if missing_aux:
            raise ValueError(f"Auxiliary metrics not found: {missing_aux}")

    # Set the aesthetic style
    sns.set_style(style)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors from palette - need more colors if we have aux metrics
    n_colors = 1 + len(aux_metrics or []) + 2  # main + aux + trend + best
    colors = sns.color_palette(palette, n_colors)

    # Plot auxiliary metrics first (so main metric is on top)
    if aux_metrics:
        for i, aux_metric in enumerate(aux_metrics):
            sns.lineplot(data=df, x='epoch', y=aux_metric,
                         marker='o', markersize=4, linewidth=1.5, alpha=0.4,
                         color=colors[i + 3], label=aux_metric.replace('_', ' ').title(),
                         ax=ax)

    # Main metric plot with enhanced styling
    main_line = sns.lineplot(data=df, x='epoch', y=main_metric,
                             marker='o', markersize=8, linewidth=3,
                             color=colors[0], alpha=0.9,
                             label=f"{main_metric.replace('_', ' ').title()} (Main)",
                             ax=ax)

    # Add smooth trend line if requested (only for main metric)
    if smooth and len(df) > 3:
        sns.regplot(data=df, x='epoch', y=main_metric,
                    scatter=False, ax=ax, color=colors[1],
                    line_kws={'alpha': 0.6, 'linestyle': '--', 'linewidth': 2},
                    label='Trend')

    # Highlight the best value (only for main metric)
    if show_best:
        best_idx = df[main_metric].idxmax()
        best_epoch = df.loc[best_idx, 'epoch']
        best_value = df.loc[best_idx, main_metric]

        ax.scatter(best_epoch, best_value, color=colors[2], s=150,
                   zorder=5, edgecolor='white', linewidth=2,
                   label='Best Value')

        # Smart annotation positioning to avoid overlaps
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]

        # Position annotation based on where best point is
        if best_epoch > (ax.get_xlim()[0] + x_range * 0.7):  # Right side
            xytext = (-15, 15)
        else:  # Left side
            xytext = (15, 15)

        ax.annotate(f'Best: {best_value:.4f}\nEpoch: {int(best_epoch)}',
                    xy=(best_epoch, best_value),
                    xytext=xytext, textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc=colors[2], alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'),
                    fontsize=10, fontweight='bold', color='white')

    # Enhanced title with template support
    main_metric_title = main_metric.replace('_', ' ').title()
    if title_suffix:
        title = title_suffix.format(main_metric=main_metric_title)
    else:
        title = f'{main_metric_title} Over Training Epochs'

    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')

    # Customize grid and spines
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Add subtle background
    ax.set_facecolor('#fafafa')

    # Improve tick styling
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add summary statistics for main metric
    stats_text = (f'{main_metric_title}:\n'
                  f'Max: {df[main_metric].max():.4f}\n'
                  f'Min: {df[main_metric].min():.4f}\n'
                  f'Final: {df[main_metric].iloc[-1]:.4f}')

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Add legend if we have multiple metrics
    if aux_metrics or smooth or show_best:
        legend = ax.legend(loc='lower center', frameon=True, fancybox=True,
                           shadow=True, fontsize=10)
        # Make legend background semi-transparent
        legend.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    best_run_metric_value = 0
    best_run_name = None
    
    api = wandb.Api()
    metric: str = "f1_score"
    run = api.run("karisu/herdnet_delplanque2022_full_eval/kg4sok4r")
    history_df = get_history(run)

    plot_metric_over_epochs(history_df,
                            main_metric=metric,
                            aux_metrics=["precision", "recall"],
                            title_suffix=f"{run.name} Scores Over Training Epochs",)

    # history_df.to_csv("metrics.csv")
    best_epoch = get_best_epoch(history_df, metric=metric)
    print(best_epoch[metric])

    # Get runs matching filters
    runs = wandb.Api().runs(
        path="karisu/herdnet_delplanque2022_full_eval", filters={"config.batch_size": 4}
    )
    
    for run in runs:
        logger.info(f"Processing run: {run.name}")
        history_df = get_history(run)

        plot_metric_over_epochs(history_df,
                                main_metric=metric,
                                aux_metrics=["precision", "recall"],
                                title_suffix=f"{run.name} Scores Over Training Epochs", )

        # history_df.to_csv("metrics.csv")
        best_epoch = get_best_epoch(history_df, metric=metric)
        print(f"best metrics: {best_epoch[metric]}")

        if best_epoch[metric] > best_run_metric_value:
            best_run_metric_value = best_epoch[metric]
            best_run_name = run.name


    print(f"DONE, best run: {best_run_name} with {metric}={best_run_metric_value}")