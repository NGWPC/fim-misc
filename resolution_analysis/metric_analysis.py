'''
Analyze covariate and metrics data.
'''

import os
from itertools import product

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# plot scatter median slope and CSI

def check_combinations(df, groupby_cols, combination_cols, return_dropped=False):
    """
    Check if each group of `groupby_cols` in the DataFrame `df` contains all unique combinations
    of the columns specified in `combination_cols`.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to check.
    groupby_cols (list or str): Column(s) to group by.
    combination_cols (list or str): Columns to check combinations for.
    
    Returns:
    dict: A dictionary where keys are group identifiers and values are sets of missing combinations.
    """
    
    # Ensure groupby_cols and combination_cols are lists
    if isinstance(groupby_cols, str):
        groupby_cols = [groupby_cols]
    if isinstance(combination_cols, str):
        combination_cols = [combination_cols]
    
    # Determine all unique combinations of the combination columns
    all_combinations = set(product(*(df[col].unique() for col in combination_cols)))
    
    def check_group(group):
        group_combinations = set(zip(*(group[col] for col in combination_cols)))
        missing_combinations = all_combinations - group_combinations
        return missing_combinations
    
    # Apply the function to each group
    result = df.groupby(groupby_cols).apply(check_group)
    
    # Convert result to a dictionary
    missing_combinations_dict = result[result.apply(len) > 0].to_dict()

    if return_dropped:
        # drop the test_case_id that have missing combinations
        df = df[~df[groupby_cols].isin(missing_combinations_dict.keys()).any(axis=1)]
        return missing_combinations_dict, df
    
    return missing_combinations_dict

def compute_logs(metrics):

    metrics['median_slope_log'] = np.log10(metrics['median_slope'])
    metrics['freq_high_dev_log'] = np.log10(metrics['freq_high_dev'])

    return metrics

def covariate_scatter_plots(metrics, plot_output_dir, benchmark_source=None):
    '''
    Plot a figure with square subplots. The first row is for median_slope_log on x-axis. The second row is for freq_high_dev_log on x-axis. The columns are for different metrics including MCC, CSI, TPR, and FAR. Only add axis and tick labels to the leftmost and bottommost subplots. Add sub-titles for each columns. Remove extra white space between subplots.
    '''

    if (benchmark_source is not None) & (benchmark_source != 'all'):
        metrics = metrics[metrics['benchmark_source'] == benchmark_source]

    resolutions = metrics['resolution']
    magnitudes = metrics['magnitude']

    # drop resolution and magnitude columns
    metrics = metrics.drop(columns=['resolution', 'magnitude'])

    if benchmark_source in ['nws', 'usgs']:
        mag_vals = ['action', 'minor', 'moderate', 'major']
    elif benchmark_source == 'ras2fim':
        mag_vals = ['2yr', '5yr', '10yr', '25yr', '50yr', '100yr']
    else:
        mag_vals = ['100yr', '500yr']

    # reorder the categories
    metrics.loc[:, 'magnitude'] = pd.Categorical(magnitudes, categories=mag_vals, ordered=True)
    metrics.loc[:, 'resolution'] = pd.Categorical(resolutions, categories=[10, 5, 3], ordered=True)
    
    fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=False, sharey=False)
    for i, metric in enumerate(['MCC', 'CSI', 'TPR', 'FAR']):
        # set resolution as hue, set magnitude as style, change style for bigger size by 
        sns.scatterplot(x='median_slope_log', y=metric, data=metrics, ax=axs[0, i], hue='magnitude', style='resolution')
        sns.scatterplot(x='freq_high_dev_log', y=metric, data=metrics, ax=axs[1, i], hue='magnitude', style='resolution')

        axs[0, i].set_title(metric)
        #axs[1, i].set_xlabel('Log Covariate Value')

        # set the same limits and tick locations
        axs[0, i].set_xlim(-3.70, -.10)
        axs[0, i].set_ylim(-0.1, 1.1)
        axs[0, i].set_xticks([-3.5, -3, -2.5, -2, -1.5, -1, -0.5])
        axs[0, i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # compute correlation
        corr = metrics[['median_slope_log', metric]].corr().iloc[0, 1]

        # compute regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            metrics['median_slope_log'].values, metrics[metric].values
        )
        axs[0, i].plot(metrics['median_slope_log'], slope*metrics['median_slope_log'] + intercept, color='red')

        # compute ci for slope
        lower_ci, upper_ci = stats.t.interval(0.99, len(metrics['median_slope_log'])-2, loc=slope, scale=std_err)
        
        # list on the plot
        axs[0, i].text(0.01, 0.01, f'Corr. Coef.: {corr:.2f}\ny={slope:.2f}x+{intercept:.2f}\n99% CI: ({lower_ci:.2f}, {upper_ci:.2f})', fontsize=10, transform=axs[0, i].transAxes)
        
        corr = metrics[['freq_high_dev_log', metric]].corr().iloc[0, 1]

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            metrics['freq_high_dev_log'].values, metrics[metric].values
        )
        axs[1, i].plot(metrics['freq_high_dev_log'], slope*metrics['freq_high_dev_log'] + intercept, color='red')

        lower_ci, upper_ci = stats.t.interval(0.99, len(metrics['freq_high_dev_log'])-2, loc=slope, scale=std_err)

        axs[1, i].text(0.01, 0.01, f'Corr. Coef.: {corr:.2f}\ny={slope:.2f}x+{intercept:.2f}\n99% CI: ({lower_ci:.2f}, {upper_ci:.2f})', fontsize=10, transform=axs[1, i].transAxes)


        axs[0, i].set_xlabel('Log of Median Slope, rise/run (m/m)')
        axs[1, i].set_xlabel('Log of High Developed LC Frequency')
        
        if i == 0:
            axs[0, i].set_ylabel('Metric Value')
            axs[1, i].set_ylabel('Metric Value')
        else:
            axs[0, i].set_ylabel('')
            axs[1, i].set_ylabel('')

        # save legend
        if i == 0:
            saved_legend = axs[0, i].get_legend()

        # remove legend
        axs[0, i].get_legend().remove()
        axs[1, i].get_legend().remove()


    # set title
    if benchmark_source:
        fig.suptitle('Scatter Plot of Metrics vs. Covariates: ' + benchmark_source.upper(), x=.75, y=.98)
    else:
        fig.suptitle('Scatter Plot of Metrics vs. Covariates', x=.75, y=.98)

    # make legend
    fig.legend(
        loc=(0.05, 0.96), ncol=9, handles=saved_legend.legend_handles
    )

    plt.tight_layout()

    plt.savefig(os.path.join(plot_output_dir, f'scatter_plots_{benchmark_source}.png'))

    plt.close()


def box_plots_by_resolution_and_magnitude(covariates, benchmark_source, plot_output_dir):

    #if benchmark_source in ['nws', 'usgs']:
    figsize = (9, 6)
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=False)

    # update magnitude categories
    magnitudes = covariates['magnitude']

    # drop the magnitude column
    covariates = covariates.drop(columns='magnitude')

    if benchmark_source in ['nws', 'usgs']:
        vals = ['action', 'minor', 'moderate', 'major']
    elif benchmark_source == 'ras2fim':
        vals = ['2yr', '5yr', '10yr', '25yr', '50yr', '100yr']
    else:
        vals = ['100yr', '500yr']

    # reorder the categories
    covariates.loc[:, 'magnitude'] = pd.Categorical(magnitudes, categories=vals, ordered=True)

    for i, metric in enumerate(['MCC', 'CSI', 'TPR', 'FAR']):

        # get multi-index
        mi = np.unravel_index(i, axs.shape)
        
        sns.boxplot(
            x='magnitude', y=metric, hue='resolution', data=covariates, ax=axs[mi], hue_order=[10, 5, 3], palette='bright'
        )

        axs[mi].set_title(metric)
        #axs[mi].tick_params(axis='x', rotation=45, labelright=True, labelleft=False)
        #axs[mi].set_ylim(0, 1)
        
        if (i == 0) | (i == 2):
            axs[mi].set_ylabel('Metric Value')
            if i == 0:
                saved_legend = axs[mi].get_legend()
        else:
            axs[mi].set_ylabel('')

        # remove legend
        axs[mi].get_legend().remove()

        # remove x-label
        axs[mi].set_xlabel('')

    # make legend below the plot, remove repeated labels, add title
    fig.legend(
        loc=(0.75, 0.9), ncol=3, title='Resolution (m)', labels=['10', '5', '3'], handles=saved_legend.legend_handles
    )

    # set figure title, make benchmark_source all caps
    fig.suptitle(f'Metrics by Magnitude & Resolution: {benchmark_source.upper()}')

    # seet figure x-label
    fig.text(0.5, 0.01, 'Magnitude', ha='center')

    plt.tight_layout(pad=1.8)

    plt.savefig(os.path.join(plot_output_dir, f'box_plots_{benchmark_source}.png'))

    plt.close()


def scatter_plot_matrix(metrics, plot_output_dir):
    '''
    Plot a scatter plot matrix with MCC, CSI, TPR, FAR, median_slope_log, freq_high_dev_log, magnitude, resolution, and benchmark_source. The diagonal should be histograms of the corresponding variables.
    '''

    variables = [
        'MCC', 'CSI', 'TPR', 'FAR',
        'median_slope_log', 'freq_high_dev_log', 'magnitude', 'resolution', 'benchmark_source', 'percent_covered_by_tiles', 'elapsed_wall_clock_mins', 'dir_size_gb'
    ]

    # subset the data
    #metrics = metrics[variables]

    sns.pairplot(metrics, kind='scatter', diag_kind='hist', vars=variables)
    
    plt.tight_layout()

    plt.savefig(os.path.join(plot_output_dir, 'scatter_plot_matrix.png'))

    plt.close()


def multiple_linear_regression(metrics, plot_output_dir):
    '''
    Multiple linear regression by metric using median_slope_log, freq_high_dev_log, magnitude, resolution, and benchmark_source as predictors. Plot the residuals vs. fitted values for each model.
    '''

    # change 100yr and 500yr to moderate and major
    metrics['magnitude'] = metrics['magnitude'].replace({100: 'moderate', 500: 'major'})

    # dummy encode the benchmark_source, magnitude, and resolution
    metrics = pd.get_dummies(metrics, columns=['benchmark_source', 'magnitude', 'resolution'])

    # predictors
    predictors = [
        'median_slope_log', 'freq_high_dev_log', 'benchmark_source_ble', 'benchmark_source_nws', 'benchmark_source_usgs', 'magnitude_action', 'magnitude_minor', 'magnitude_moderate', 'magnitude_major', 'resolution_3', 'resolution_5', 'resolution_10'
    ]

    # loop through each metric


def box_plots_by_tile_availability(metrics, plot_output_dir):
    '''
    Make boxplots for three groups of percent_covered_by_tiles: none, partial, full. Make 2x2 plots for each metric: MCC, CSI, TPR, FAR.
    '''

    # make a copy of the data
    metrics = metrics.copy()

    # create a new column for tile availability
    metrics.loc[metrics['percent_covered_by_tiles'] == 0, 'tile_availability'] = 'none'
    metrics.loc[metrics['percent_covered_by_tiles'] > 0, 'tile_availability'] = 'partial'
    metrics.loc[metrics['percent_covered_by_tiles'] == 100, 'tile_availability'] = 'full'

    # print the sample size by tile availability
    print(f"Sample size by tile availability: {metrics.groupby('tile_availability').size()}")

    # make a list of metrics
    metrics_list = ['MCC', 'CSI', 'TPR', 'FAR']
            
    # create a figure
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    
            # loop through each tile availability
    for i, metric in enumerate(metrics_list):

        # get the axes
        ax = axs.flatten()[i]

        # make a boxplot
        sns.boxplot(
            x='tile_availability', y=metric, data=metrics, ax=ax, palette='bright', order=['none', 'partial', 'full'], hue='resolution', hue_order=[10, 5, 3]
        )

        # set the title
        ax.set_title(f'{metric}')

        # set the y-label
        if (i == 0) | (i == 2):
            ax.set_ylabel('Metric Value')
            if i == 0:
                saved_legend = ax.get_legend()
        else:
            ax.set_ylabel('')

        ax.set_xlabel('')

        # remove the legend
        ax.get_legend().remove()

    # set the x-label
    plt.text(-.2, -.2, 'Tile Availability (none, partial, full)', ha='center', va='center', transform=plt.gca().transAxes)

    # make the legend
    fig.legend(
        loc=(0.75, 0.9), ncol=3, title='Resolution (m)', labels=['10', '5', '3'], handles=saved_legend.legend_handles
    )

    # set the title
    fig.suptitle(f'Metric by Tile Availability & Resolution')

    # save the figure
    plt.savefig(os.path.join(plot_output_dir, f'box_plots_by_tile_availability.png'))

    # close the figure
    plt.close()


def histogram_of_compute_and_storage_costs_by_algorithm(metrics, plot_output_dir, usd=False):

    # make histograms of compute and storage costs by algorithm, stack the histograms horizontally
    fig, axs = plt.subplots(1, 2, figsize=(9, 6), sharey=True)

    if usd:
        compute_column = 'elapsed_wall_clock_cost_usd'
        storage_column = 'storage_cost_month_usd'
    else:
        compute_column = 'elapsed_wall_clock_mins'
        storage_column = 'dir_size_gb'

    # make histograms
    sns.histplot(metrics, x=compute_column, hue='algorithm', ax=axs[0])
    sns.histplot(metrics, x=storage_column, hue='algorithm', ax=axs[1])

    
    # set the x-label
    if usd:
        axs[0].set_xlabel('HAND Compute Cost by Algorithm (USD)')
        axs[1].set_xlabel('Storage Cost by Algorithm (USD)')
    else:
        axs[0].set_xlabel('HAND Compute Time by Algorithm (mins)')
        axs[1].set_xlabel('Storage Space by Algorithm (GB)')
    
    # set the y-label
    axs[0].set_ylabel('Frequency')
    axs[1].set_ylabel('Frequency')

    # save the legend
    saved_legend = axs[0].get_legend()

    # remove the legend
    axs[0].get_legend().remove()
    axs[1].get_legend().remove()

    # make the legend
    handles, labels = saved_legend.legend_handles, [t.get_text() for t in saved_legend.texts]
    fig.legend(
        loc=(.45, 0.90), ncol=3, title='Algorithm', handles=handles, labels=labels
    )

    # compute mean, std, median, 95% CI, min, max for compute and storage costs by algorithm
    compute_costs = metrics.groupby('algorithm', observed=False)[compute_column].agg(['mean', 'std', 'median', 'min', 'max'])
    storage_costs = metrics.groupby('algorithm', observed=False)[storage_column].agg(['mean', 'std', 'median', 'min', 'max'])

    # add the statistics to the plot
    for i, (costs, ax) in enumerate(zip([compute_costs, storage_costs], axs)):
        for j, (algorithm, row) in enumerate(costs.iterrows()):
            ax.text(0.75, 0.85 - j*0.1, f'\n{algorithm}\nMean: {row["mean"]:.2f}\nStd: {row["std"]:.2f}\nMedian: {row["median"]:.2f}', fontsize=8, transform=ax.transAxes)


    # save the figure
    if usd:
        plt.savefig(os.path.join(plot_output_dir, 'histogram_of_compute_and_storage_costs_by_algorithm.png'))
    else:
        plt.savefig(os.path.join(plot_output_dir, 'histogram_of_compute_and_storage_amounts_by_algorithm.png'))

    # close the figure
    plt.close()

def boxplot_of_algorithm_performance(metrics, benchmark_source, plot_output_dir):
    '''
    Make boxplots of MCC, CSI, TPR, FAR by algorithm. Make 2x2 plots.
    '''

    # make a copy of the data
    metrics = metrics.copy()

    # make a list of metrics
    metrics_list = ['MCC', 'CSI', 'TPR', 'FAR']

    # update categories
    metrics['algorithm'] = metrics['algorithm'].astype('category')

    # only keep test_case_id that have both algorithms: 'richdem' and 'wbt'
    metrics = metrics.groupby('test_case_id').filter(lambda x: set(x['algorithm']) == {'richdem', 'wbt'})

    # create a figure
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=False)

    # set x order
    if benchmark_source in ['nws', 'usgs']:
        mag_vals = ['action', 'minor', 'moderate', 'major']
    elif benchmark_source == 'ras2fim':
        mag_vals = ['2yr', '5yr', '10yr', '25yr', '50yr', '100yr']
    else:
        mag_vals = ['100yr', '500yr']

    # loop through each metric
    for i, metric in enumerate(metrics_list):

        # get the axes
        ax = axs.flatten()[i]

        # make a boxplot
        sns.boxplot(
            x='magnitude', y=metric, data=metrics, ax=ax, palette='bright', hue='algorithm', hue_order=['richdem', 'wbt'], order=mag_vals
        )

        # set the title
        ax.set_title(f'{metric}')

        # set the y-label
        if (i == 0) | (i == 2):
            ax.set_ylabel('Metric Value')
            if i == 0:
                saved_legend = ax.get_legend()
        else:
            ax.set_ylabel('')

        ax.set_xlabel('')

        # remove the legend
        try:
            ax.get_legend().remove()
        except AttributeError:
            pass

    # set the x-label
    plt.text(-.2, -.2, 'Algorithm', ha='center', va='center', transform=plt.gca().transAxes)

    # make the legend
    fig.legend(
        loc=(0.75, 0.9), ncol=3, title='Algorithm', labels=['richdem', 'wbt'], handles=saved_legend.legend_handles
    )

    # set the title
    fig.suptitle(f'Metric by Algorithm & Magnitude: {benchmark_source.upper()}')

    # save the figure
    plt.savefig(os.path.join(plot_output_dir, f'box_plots_by_algorithm_{benchmark_source}.png'))

    # close the figure
    plt.close()

def box_plot_of_compute_performance_by_resolutions(compute_metrics, plot_output_dir, usd=False, total_hucs = 1451):
    '''
    Make boxplots of elapsed_wall_clock_mins by resolution.
    '''

    # make a copy of the data
    compute_metrics = compute_metrics.copy()

    # create a figure
    fig, axs = plt.subplots(1, 2, figsize=(9, 6))

    if usd:
        y_time = 'elapsed_wall_clock_cost_usd'
        title_time = 'Elapsed Wall Clock Cost by Resolution'
        y_label_time = 'Elapsed Wall Clock Cost (USD)'
        y_storage = 'storage_cost_month_usd'
        title_storage = 'Monthly Storage Cost (USD) by Resolution'
        y_label_storage = 'Monthly Storage Cost (USD)'
        median_df_columns = ['Median Compute Cost (USD)', 'Median Monthly Storage Cost (USD)']
        total_df_columns = ['Total Compute Cost (USD)', 'Total Monthly Storage Cost (USD)']
    else:
        y_time = 'elapsed_wall_clock_mins'
        title_time = 'Elapsed Wall Clock Time by Resolution'
        y_label_time = 'Elapsed Wall Clock Time (mins)'
        y_storage = 'dir_size_gb'
        title_storage = 'Storage Space by Resolution'
        y_label_storage = 'Storage Space (GB)'
        median_df_columns = ['Median Compute Time (mins)', 'Median Storage (GB)']
        total_df_columns = ['Total Compute Time (mins)', 'Total Storage (GB)']


    # process
    compute_metrics = compute_metrics[compute_metrics.algorithm == 'wbt']

    # total time/storage by resolution
    
    # first remove rows that don't have all resolutions for every huc
    #compute_metrics = compute_metrics.groupby(['huc', 'resolution']).filter(lambda x: len(x) == 3)

    print(f'Total HUCs: {total_hucs}')

    median_df = pd.concat(
        (
            compute_metrics.groupby("resolution")[y_time].median(),
            compute_metrics.groupby("resolution")[y_storage].median() 
        ),
        axis=1
    )
    median_df.columns = median_df_columns
    #print(median_df)

    # now totals
    total_df = median_df * total_hucs
    total_df.columns = total_df_columns
    #print(total_df)

    # combine the two dataframes
    all_df = pd.concat((median_df, total_df), axis=1)
    print(all_df)

    # make a boxplot
    sns.boxplot(
        x='resolution', y=y_time, data=compute_metrics, ax=axs[0], order=[10, 5, 3]
    )
    sns.boxplot(
        x='resolution', y=y_storage, data=compute_metrics, ax=axs[1], order=[10, 5, 3]
    )

    # set the title
    axs[0].set_title(title_time)
    axs[1].set_title(title_storage)

    # set the x-label
    axs[0].set_xlabel('Resolution (m)')
    axs[1].set_xlabel('Resolution (m)')

    # set the y-label
    axs[0].set_ylabel(y_label_time)
    axs[1].set_ylabel(y_label_storage)

    # set y-limits
    if usd:
        axs[0].set_ylim(0, 20)
        axs[1].set_ylim(0, 10)
    else:
        axs[0].set_ylim(0, 350)
        axs[1].set_ylim(0, 35)

    # make figure title
    fig.suptitle('HUC8 Level')

    # save the figure
    units_name = 'USD' if usd else 'mins'
    plt.savefig(os.path.join(plot_output_dir, f'box_plot_of_compute_performance_by_resolutions_{units_name}.png'))

    # close the figure
    plt.close()


def build_tile_regression_model(metrics, plot_output_dir, categorical=True):
    """
    Build regression model of tile availability and metric values by metric.
    """

    # make a copy of the data
    metrics = metrics.copy()

    # compute tile availability and encode as 0, 1, 2 for none, partial, full
    if categorical:
        metrics.loc[metrics['percent_covered_by_tiles'] == 0, 'tile_availability'] = 0
        metrics.loc[metrics['percent_covered_by_tiles'] > 0, 'tile_availability'] = 1
        metrics.loc[metrics['percent_covered_by_tiles'] == 100, 'tile_availability'] = 2
        metrics.loc[metrics['percent_covered_by_tiles'] == 0, 'tile_availability_for_plot'] = 'none'
        metrics.loc[metrics['percent_covered_by_tiles'] > 0, 'tile_availability_for_plot'] = 'partial'
        metrics.loc[metrics['percent_covered_by_tiles'] == 100, 'tile_availability_for_plot'] = 'full'

        x_col = 'tile_availability'
        x_col_plot = 'tile_availability_for_plot'
        x_label = 'Tile Availability (none, partial, full)'
        title = 'Metric by Tile Availability'
        xticks = [0, 1, 2]
        xtick_labels = ['none', 'partial', 'full']
    else:
        x_col = 'percent_covered_by_tiles'
        x_col_plot = x_col
        x_label = 'HUC8 Percent Covered by Tiles'
        title = 'Metric by HUC8 Percent Covered by Tiles'
        xticks = [0, 25, 50, 75, 100]
        xtick_labels = ['0', '25', '50', '75', '100']

    # make a list of metrics
    metrics_list = ['MCC', 'CSI', 'TPR', 'FAR']

    # create a figure
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=False)

    # loop through each metric
    for i, metric in enumerate(metrics_list):

        # get the axes
        ax = axs.flatten()[i]

        # make a regression plot
        sns.scatterplot(
            x=x_col, y=metric, data=metrics, ax=ax
        )

        # set the title
        ax.set_title(f'{metric}')

        # set the y-label
        if (i == 0) | (i == 2):
            ax.set_ylabel('Metric Value')
        else:
            ax.set_ylabel('')

        # only plot x-label for the bottom row
        if i >= 2:
            ax.set_xlabel(x_label)

        # set x-ticks
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)

        # fit regression line using lingress
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            metrics[x_col].values, metrics[metric].values
        )

        # plot the regression line
        ax.plot(metrics[x_col], slope*metrics[x_col] + intercept, color='red')

        # compute correlation
        corr = metrics[[x_col, metric]].corr().iloc[0, 1]

        # compute 99% CI for slope
        lower_ci, upper_ci = stats.t.interval(0.99, len(metrics[x_col])-2, loc=slope, scale=std_err)

        # add the correlation, regression line, and CI to the plot
        ax.text(0.01, 0.01, f'Corr. Coef.: {corr:.2f}\ny={slope:.2f}x+{intercept:.2f}\n99% CI: ({lower_ci:.2f}, {upper_ci:.2f})', fontsize=10, transform=ax.transAxes)

    # set the title
    fig.suptitle(f'Metric by Tile Availability')

    # save the figure
    name = 'categorical' if categorical else 'continuous'
    plt.savefig(os.path.join(plot_output_dir, f'regression_plot_by_tile_availability_{name}.png'))

    # close the figure
    plt.close()


def difference_histogram(metrics, plot_output_dir):
    '''
    For each metric, compute the difference between 3m - 10m, 5m - 10m, and 5m - 3m. Make a histogram for each metric, and set the hue as the resolution difference. This will only be for usgs and nws benchmarks at the action and minor levels. So set the titles to include the benchmark source and magnitude.
    '''

    # make a copy of the data
    metrics = metrics.copy()

    # only keep test_case_id that have all resolutions
    #metrics = metrics.groupby('test_case_id').filter(lambda x: set(x['resolution']) == {10, 5, 3})

    # make a list of metrics
    metrics_list = ['MCC', 'CSI', 'TPR', 'FAR']

    # subset benchmark source
    metrics = metrics[metrics['benchmark_source'].isin(['usgs', 'nws'])]

    # subset magnitude
    metrics = metrics[metrics['magnitude'].isin(['action', 'minor'])]

    # set x order
    mag_vals = ['action', 'minor']

    # set magnitude and resolution as categories
    magnitudes = metrics['magnitude']
    resolutions = metrics['resolution']

    # drop the magnitude and resolution columns
    metrics = metrics.drop(columns=['magnitude', 'resolution'])

    # set the categories
    metrics.loc[:, 'magnitude'] = pd.Categorical(magnitudes, categories=mag_vals, ordered=True)
    metrics.loc[:, 'resolution'] = pd.Categorical(resolutions, categories=[10, 5, 3], ordered=True)

    # for every resolution difference, compute the difference between the two resolutions
    metrics_diff_list = []
    for diff in ['3-10', '3-5', '5-10']:

        if diff == '3-10':
            reses = [3, 10]
        elif diff == '3-5':
            reses = [3, 5]
        else:
            reses = [5, 10]

        metrics_diff = (
            metrics[(metrics.resolution == reses[0])].set_index(['test_case_id', 'magnitude', 'benchmark_source'])[metrics_list] - metrics[(metrics.resolution == reses[1])].set_index(['test_case_id', 'magnitude'])[metrics_list]
        ).reset_index().dropna().merge(metrics.drop(columns=metrics_list), on=['test_case_id', 'magnitude', 'benchmark_source'], how='left').drop_duplicates(subset=['test_case_id', 'magnitude', 'benchmark_source'] + metrics_list)

        metrics_diff['resolution_diff'] = diff

        metrics_diff_list.append(metrics_diff)

    # drop resolution column
    metrics_diff = pd.concat(metrics_diff_list, ignore_index=True).drop(columns='resolution')

    # set resolution_diff as category
    resolution_diffs = metrics_diff['resolution_diff']
    metrics_diff.drop(columns='resolution_diff', inplace=True)
    metrics_diff['resolution_diff'] = pd.Categorical(resolution_diffs, categories=['3-10', '3-5', '5-10'], ordered=True)

    for diff in ['3-10', '3-5', '5-10']:

        # create a figure
        fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=False)

        # loop through each metric
        for i, metric in enumerate(metrics_list):

            # get the axes
            ax = axs.flatten()[i]

            # make a boxplot
            sns.histplot(
                x=metric, data=metrics_diff[metrics_diff['resolution_diff'] == diff], ax=ax#, bins = 50
            )

            # set the title
            ax.set_title(f'{metric}')

            # set the y-label
            if (i == 0) | (i == 2):
                ax.set_ylabel('Frequency')
                #if i == 0:
                #    saved_legend = ax.get_legend()
            else:
                ax.set_ylabel('')

            ax.set_xlabel('')

            ax.set_xlim(-0.1, 0.1)

            # remove the legend
            #ax.get_legend().remove()

            # compute summary statistics: mean, std, median, 25th, 75th, min, max, sample size (n)
            summary = metrics_diff[metrics_diff['resolution_diff'] == diff][metric].describe()
            n = len(metrics_diff[metrics_diff['resolution_diff'] == diff][metric])

            # add the statistics to the plot upper right corner, also add 5 and 95 percentiles
            #ax.text(0.68, 0.55, f'Mean: {summary["mean"]:.2f}\nStd: {summary["std"]:.2f}\nMedian: {summary["50%"]:.2f}\n25th: {summary["25%"]:.2f}\n75th: {summary["75%"]:.2f}\nMin: {summary["min"]:.2f}\nMax: {summary["max"]:.2f}\nN: {n}\n5th: {metrics_diff[metrics_diff['resolution_diff'] == diff][metric].quantile(0.05):.2f}\n95th: {metrics_diff[metrics_diff['resolution_diff'] == diff][metric].quantile(0.95):.2f}', fontsize=8, transform=ax.transAxes)

            # above block gives: SyntaxError: f-string: unmatched '['
            ax.text(0.68, 0.45, f'Mean: {summary["mean"]:.2f}\nStd: {summary["std"]:.2f}\nMedian: {summary["50%"]:.2f}\n25th: {summary["25%"]:.2f}\n75th: {summary["75%"]:.2f}\nMin: {summary["min"]:.2f}\nMax: {summary["max"]:.2f}\nN: {n}\n10th: {metrics_diff[metrics_diff["resolution_diff"] == diff][metric].quantile(0.1):.2f}\n90th: {metrics_diff[metrics_diff["resolution_diff"] == diff][metric].quantile(0.9):.2f}', fontsize=8, transform=ax.transAxes)

            # compute mean absolute deviation
            mad = np.mean(np.abs(metrics_diff[metrics_diff['resolution_diff'] == diff][metric] - summary['mean']))

            # plot normal distribution curve on ax with mean and standard deviation
            x = np.linspace(-0.1, 0.1, 100)
            y = stats.norm.pdf(x, loc=summary['mean'], scale=summary['std'])

            # normalize y to match the histogram
            # get the bin width from the plot
            bin_width = ax.patches[1].get_x() - ax.patches[0].get_x()

            ax.plot(x, y * n * bin_width, color='red')

        # set the x-label
        plt.text(-.2, -.2, f'Metric Differences Across Resolutions: {diff.upper()}m', ha='center', va='center', transform=plt.gca().transAxes)

        # make the legend
        #fig.legend(
        #    loc=(0.75, 0.9), ncol=3, title='Resolution Difference (m)', labels=['3-10', '5-10', '5-3'], handles=saved_legend.legend_handles
        #)

        # make legend just for pdf line
        from matplotlib.lines import Line2D
        fig.legend(
            loc=(0.75, 0.9), ncol=1, labels=['Normal Distribution'], handles=[Line2D([0], [0], color='red', lw=2, label='Normal Distribution')]
        )

        # set the title
        fig.suptitle(f'Metric by Resolution Difference (USGS & NWS @ Action & Minor): {diff.upper()}m')

        # save the figure
        plt.savefig(os.path.join(plot_output_dir, f'difference_histogram_{diff}.png'))

        # close the figure
        plt.close()


def scatter_plot_covariates_and_metrics_by_resolution(metrics, plot_output_dir):
    """
    Make a figure with 2 rows (one for each spatial covariate) and 4 columns (one for each metric). Each subplot will be a scatter plot of the metric vs. the log covariates, with a regression line and correlation coefficient. The title of each subplot will be the metric name. Use both covariates: median_slope_log and freq_high_dev_log.
    """
    
    # make a copy of the data
    metrics = metrics.copy()

    # make a list of metrics
    metrics_list = ['MCC', 'CSI', 'TPR', 'FAR']

    # create a figure
    fig, axs = plt.subplots(2, 4, figsize=(12, 6), sharex=False, sharey=True)

    # resolutions recategorized
    resolutions = metrics['resolution']
    metrics.drop(columns='resolution', inplace=True)
    metrics['resolution'] = pd.Categorical(resolutions, categories=[10, 5, 3], ordered=True)

    from itertools import product

    # loop through each metric
    for i, (covariate, metric) in enumerate(product(['median_slope_log', 'freq_high_dev_log'], metrics_list)):

        # get the axes
        ax = axs.flatten()[i]

        # make a regression plot
        sns.scatterplot(
            x=covariate, y=metric, data=metrics, ax=ax, hue='resolution', palette='bright', hue_order=[10, 5, 3]
        )

        # set the title
        if covariate == 'median_slope_log':
            ax.set_title(f'{metric}')

        # set the y-label
        if (i == 0) | (i == 4):
            ax.set_ylabel('Metric Value')
        else:
            ax.set_ylabel('')

        # only plot x-label for the bottom row
        ax.set_xlabel(covariate)

        ax.set_ylim(0, 1)

        # save legend
        if i == 0:
            saved_legend = ax.get_legend()

        # remove the legend
        ax.get_legend().remove()

        for r, c in zip([10, 5, 3], ['blue', 'orange', 'green']):
            # fit regression line using lingress
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                metrics.loc[metrics.resolution == r, covariate].values, metrics.loc[metrics.resolution == r, metric].values
            )

            #print(f'{metric} vs. {covariate} at {r}m: slope={slope:.2f}, intercept={intercept:.2f}, r_value={r_value:.2f}, p_value={p_value:.2f}, std_err={std_err:.2f}')

            # plot the regression line
            x = np.linspace(metrics[covariate].min(), metrics[covariate].max(), 100)
            y = slope*x + intercept
            ax.plot(x, y, color=c)

    # set the title
    fig.suptitle(f'Metric by Resolution & Covariate')

    # make the legend
    fig.legend(
        loc=(0.7, 0.92), ncol=3, title='Resolution (m)', labels=['10', '5', '3'], handles=saved_legend.legend_handles
    )

    # save the figure
    plt.savefig(os.path.join(plot_output_dir, f'scatter_plot_covariates_and_metrics_by_resolution.png'))

    # close the figure
    plt.close()

def counts_plot(metrics, plot_output_dir):

    counts_df = metrics[(metrics['Exit status'] == 0)].dropna(subset=['MCC', 'CSI', 'TPR', 'FAR']).groupby(['benchmark_source', 'magnitude', 'resolution'])[['MCC']].count().reset_index(drop=False).rename(columns={'MCC' : 'counts'})

    # counts df columns: benchmark_source, magnitude, resolution, counts

    # with counts df make a figure with 4 subplots, one for each benchmark_source, with 2 rows and 2 columns. sources are ble, nws, usgs, ras2fim. Make bar plot with a bar for each resolution grouped by magnitude. The counts is what to plot. Order the legend according to a specific magnitude order by benchmark source

    # create a figure
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=False, sharey=False)

    # set magnitude order
    nws_usgs_mag_order = ['action', 'minor', 'moderate', 'major']
    ras2fim_mag_order = ['2yr', '5yr', '10yr', '25yr', '50yr', '100yr']
    ble_mag_order = ['100yr', '500yr']

    # resolution order
    res_order = [10, 5, 3]

    # loop through each benchmark source
    for i, (benchmark_source, mag_order) in enumerate(
        zip(['ble', 'nws', 'usgs', 'ras2fim'],
        [ble_mag_order, nws_usgs_mag_order, nws_usgs_mag_order, ras2fim_mag_order])
    ):
            
            # get the axes
            ax = axs.flatten()[i]
    
            # make a bar plot
            sns.barplot(
                x='magnitude', y='counts', data=counts_df[counts_df['benchmark_source'] == benchmark_source], ax=ax, hue='resolution', hue_order=res_order, palette='bright', order=mag_order
            )
    
            # set the title
            ax.set_title(f'{benchmark_source.upper()}')
    
            # set the y-label
            if (i == 0) | (i == 2):
                ax.set_ylabel('Counts')
                if i == 0:
                    saved_legend = ax.get_legend()
            else:
                ax.set_ylabel('')
    
            ax.set_xlabel('')
    
            # remove the legend
            ax.get_legend().remove()

            # make sure y tick labels are integers
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))


    # set the x-label
    plt.text(-.2, -.2, 'Magnitude', ha='center', va='center', transform=plt.gca().transAxes)

    # make the legend
    fig.legend(
        loc=(0.75, 0.9), ncol=3, title='Resolution (m)', labels=['10', '5', '3'], handles=saved_legend.legend_handles
    )

    # set the title
    fig.suptitle(f'Sample Sizes by Benchmark, Magnitude & Resolution')

    # save the figure
    plt.savefig(os.path.join(plot_output_dir, f'sample_size_plot.png'))

    # close the figure
    plt.close()


def metric_difference_by_covariate(metrics, plot_output_dir, hist_var='median_slope_log'):
    '''
    For each metric, make a box-plot of positive and negative differences
    '''

    # make a copy of the data
    metrics = metrics.copy()

    # only keep test_case_id that have all resolutions
    #metrics = metrics.groupby('test_case_id').filter(lambda x: set(x['resolution']) == {10, 5, 3})

    # make a list of metrics
    metrics_list = ['MCC', 'CSI', 'TPR', 'FAR']

    # subset benchmark source
    metrics = metrics[metrics['benchmark_source'].isin(['usgs', 'nws'])]

    # subset magnitude
    metrics = metrics[metrics['magnitude'].isin(['action', 'minor'])]

    # set x order
    mag_vals = ['action', 'minor']

    # set magnitude and resolution as categories
    magnitudes = metrics['magnitude']
    resolutions = metrics['resolution']

    # drop the magnitude and resolution columns
    metrics = metrics.drop(columns=['magnitude', 'resolution'])

    # set the categories
    metrics.loc[:, 'magnitude'] = pd.Categorical(magnitudes, categories=mag_vals, ordered=True)
    metrics.loc[:, 'resolution'] = pd.Categorical(resolutions, categories=[10, 5, 3], ordered=True)

    # for every resolution difference, compute the difference between the two resolutions
    metrics_diff_list = []
    #for diff in ['3-10', '3-5', '5-10']:
    diff = '3-10'

    if diff == '3-10':
        reses = [3, 10]
    elif diff == '3-5':
        reses = [3, 5]
    else:
        reses = [5, 10]

    metrics_diff = (
        metrics[(metrics.resolution == reses[0])].set_index(['test_case_id', 'magnitude', 'benchmark_source'])[metrics_list] - metrics[(metrics.resolution == reses[1])].set_index(['test_case_id', 'magnitude'])[metrics_list]
    ).reset_index().dropna().merge(metrics.drop(columns=metrics_list), on=['test_case_id', 'magnitude', 'benchmark_source'], how='left').drop_duplicates(subset=['test_case_id', 'magnitude', 'benchmark_source'] + metrics_list)

    metrics_diff['resolution_diff'] = diff

    metrics_diff_list.append(metrics_diff)

    # drop resolution column
    metrics_diff = pd.concat(metrics_diff_list, ignore_index=True).drop(columns='resolution')

    # set resolution_diff as category
    resolution_diffs = metrics_diff['resolution_diff']
    metrics_diff.drop(columns='resolution_diff', inplace=True)
    metrics_diff['resolution_diff'] = pd.Categorical(resolution_diffs, categories=['3-10', '3-5', '5-10'], ordered=True)

    # fit logistic regression model for CSI. Use median_slope_log and freq_high_dev_log as predictors. Use resolution_diff > 0 and resolution_diff < 0 as response categories
    metrics_diff['CSI_diff_positive'] = metrics_diff['CSI'].apply(lambda x: 'Positive' if x > 0 else 'Non-Positive')
    metrics_diff['MCC_diff_positive'] = metrics_diff['MCC'].apply(lambda x: 'Positive' if x > 0 else 'Non-Positive')
    metrics_diff['TPR_diff_positive'] = metrics_diff['TPR'].apply(lambda x: 'Positive' if x > 0 else 'Non-Positive')
    metrics_diff['FAR_diff_positive'] = metrics_diff['FAR'].apply(lambda x: 'Positive' if x > 0 else 'Non-Positive')
    #metrics_diff['resolution_diff_not_positive'] = metrics_diff['resolution_diff'].apply(lambda x: 1 if x == '10-3' else 0)

    # set resolution_diff_positive to category
    metrics_diff['CSI_diff_positive'] = metrics_diff['CSI_diff_positive'].astype('category')
    metrics_diff['MCC_diff_positive'] = metrics_diff['MCC_diff_positive'].astype('category')
    metrics_diff['TPR_diff_positive'] = metrics_diff['TPR_diff_positive'].astype('category')
    metrics_diff['FAR_diff_positive'] = metrics_diff['FAR_diff_positive'].astype('category')

    # make histogram of median_slope_log and freq_high_dev_log
    # in each histogram use resolution_diff_positive as hue
    if hist_var == 'median_slope_log':
        x_var = 'median_slope_log'
        x_axis_label = 'Log Median Slope'
        title = 'Log of Median Slope by Metric Difference'
    else:
        x_var = 'freq_high_dev_log'
        x_axis_label = 'Log Frequency High-Dev LC'
        title = 'Log of Frequency High-Dev LC by Metric Difference'

    # create a figure
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=False)

    # make histogram for CSI
    for i, metric in enumerate(metrics_list):
            
        # get the axes
        ax = axs.flatten()[i]

        # make a boxplot
        sns.histplot(
            x=x_var, data=metrics_diff, ax=ax, hue='CSI_diff_positive', palette='bright', bins=50, kde=True
        )

        # set the title
        ax.set_title(f'{metric}')

        # set the y-label
        if (i == 0) | (i == 2):
            ax.set_ylabel('Frequency')
            if i == 0:
                saved_legend = ax.get_legend()
        else:
            ax.set_ylabel('')

        ax.set_xlabel('')

        # remove the legend
        ax.get_legend().remove()

        # compute the means, std devs, medians, 25th, 75th, min, max, and sample size for each hue
        for hue, x_spot in zip(['Positive', 'Non-Positive'], [0.01, 0.3]):
            summary = metrics_diff[metrics_diff['CSI_diff_positive'] == hue][x_var].describe()
            n = len(metrics_diff[metrics_diff['CSI_diff_positive'] == hue][x_var])

            # add the statistics to the plot upper right corner, also add 5 and 95 percentiles
            #ax.text(x_spot, 0.47, f'{hue}\nMean: {summary["mean"]:.2f}\nStd: {summary["std"]:.2f}\nMedian: {summary["50%"]:.2f}\n25th: {summary["25%"]:.2f}\n75th: {summary["75%"]:.2f}\nMin: {summary["min"]:.2f}\nMax: {summary["max"]:.2f}\nN: {n}', fontsize=8, transform=ax.transAxes)

    # set the x-label
    plt.text(-.2, -.2, x_axis_label, ha='center', va='center', transform=plt.gca().transAxes)

    # make the legend
    fig.legend(
        loc=(0.75, 0.9), ncol=2, title='Positive or Non-Positive', labels=['Pos.', 'Non-Pos.'], handles=saved_legend.legend_handles
    )

    # set the title
    fig.suptitle(title)

    # add text as a footnote
    plt.text(-1, -0.25, 'USGS & NWS @ Action & Minor, (3-10m)', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)

    # save the figure
    plt.savefig(os.path.join(plot_output_dir, f'histogram_{hist_var}_by_metric_diff.png'))

    plt.close()


    # fit logistic regression model for CSI. Use median_slope_log and freq_high_dev_log as predictors. Use resolution_diff > 0 and resolution_diff < 0 as response categories
    #import statsmodels
    #from statsmodels.formula.api import logit
    #model = logit('resolution_diff_positive ~ median_slope_log + freq_high_dev_log', data=metrics_diff).fit()

    # make a copy of the data
    #print(model.summary())


if __name__ == '__main__':
        
        metrics_csv = os.path.join('data', 'merged_metrics.csv')
        metrics = pd.read_csv(
            metrics_csv,
            dtype={
                'huc': str,
                'nws_lid': str,
                'test_case_id': str,
                'resolution': int,
                'magnitude': str,
                'benchmark_source': str,
                'percent_covered_by_tiles': float
            }
        )

        compute_metrics_gpkg = os.path.join('data', 'compute_metrics.gpkg')
        compute_metrics = gpd.read_file(compute_metrics_gpkg)

        plot_output_dir = os.path.join('data', 'plots', 'metric_analysis')
        os.makedirs(plot_output_dir, exist_ok=True)


        # doing this here to avoid dropping ras2fm
        print('Computing counts plot...')
        counts_plot(metrics, plot_output_dir)


        # drop ras2fim
        metrics_ras2fim = metrics.loc[metrics.benchmark_source == 'ras2fim']

        # drop rows with missing values
        metrics.dropna(
            subset=['freq_high_dev', 'median_slope', 'MCC', 'CSI', 'TPR', 'FAR'],
            inplace=True
        )

        # ble, usgs, nws bool
        ble_bool = metrics.benchmark_source == 'ble'
        usgs_bool = metrics.benchmark_source == 'usgs'
        nws_bool = metrics.benchmark_source == 'nws'
        ras2fim_bool = metrics.benchmark_source == 'ras2fim'

        # update categories
        metrics['resolution'] = metrics['resolution'].astype('category')
        metrics['magnitude'] = metrics['magnitude'].astype('category')
        metrics['benchmark_source'] = metrics['benchmark_source'].astype('category')
        metrics['algorithm'] = metrics['algorithm'].astype('category')

        # compute logs
        metrics = compute_logs(metrics)

        print('Computing box plots by tile availability ...')
        box_plots_by_tile_availability(metrics, plot_output_dir)

        print('Computing box plots by resolution and magnitude ...')
        box_plots_by_resolution_and_magnitude(metrics[ble_bool], 'ble', plot_output_dir)
        box_plots_by_resolution_and_magnitude(metrics[nws_bool], 'nws', plot_output_dir)
        box_plots_by_resolution_and_magnitude(metrics[usgs_bool], 'usgs', plot_output_dir)
        box_plots_by_resolution_and_magnitude(metrics_ras2fim, 'ras2fim', plot_output_dir)
        
        print('Computing scatter plots...')
        for benchmark_source in ['ble', 'nws', 'usgs']:
            covariate_scatter_plots(metrics, plot_output_dir, benchmark_source)
            
        print('Computing scatter plot matrix...')
        scatter_plot_matrix(metrics, plot_output_dir)

        #print('Computing histogram of compute and storage costs by algorithm...')
        # dropping duplicate huc, algorithm pairs
        # onlt relevant if there are multiple algorithms
        """
        histogram_of_compute_and_storage_costs_by_algorithm(
            pd.concat((metrics, metrics_ras2fim)).drop_duplicates(subset=['huc', 'algorithm']),
            plot_output_dir,
            usd=False
        )
        histogram_of_compute_and_storage_costs_by_algorithm(
            pd.concat((metrics, metrics_ras2fim)).drop_duplicates(subset=['huc', 'algorithm']),
            plot_output_dir,
            usd=True
        )
        """

        print('Computing boxplot of algorithm performance...')
        # there is only one algorithm so not currently relevant
        boxplot_of_algorithm_performance(metrics, 'all', plot_output_dir)
        boxplot_of_algorithm_performance(metrics[ble_bool], 'ble', plot_output_dir)
        boxplot_of_algorithm_performance(metrics[nws_bool], 'nws', plot_output_dir)
        boxplot_of_algorithm_performance(metrics[usgs_bool], 'usgs', plot_output_dir)
        boxplot_of_algorithm_performance(metrics[ras2fim_bool], 'ras2fim', plot_output_dir)

        print('Computing box plot of compute performance by resolutions...')
        # HUCs completed by resolution for wbt: 1441 (10m), 1429 (5m), 207 (3m), total 1451
        box_plot_of_compute_performance_by_resolutions(compute_metrics, plot_output_dir, usd=False, total_hucs=1451)
        box_plot_of_compute_performance_by_resolutions(compute_metrics, plot_output_dir, usd=True, total_hucs=1451)

        print('Building tile regression model...')
        build_tile_regression_model(metrics, plot_output_dir, categorical=True)
        build_tile_regression_model(metrics, plot_output_dir, categorical=False)

        print('Computing difference histogram...')
        difference_histogram(metrics, plot_output_dir)

        print('Computing scatter plot of covariates and metrics by resolution...')
        scatter_plot_covariates_and_metrics_by_resolution(metrics, plot_output_dir)

        print('Metric difference by covariate...')
        metric_difference_by_covariate(metrics, plot_output_dir, hist_var='median_slope_log')
        metric_difference_by_covariate(metrics, plot_output_dir, hist_var='freq_high_dev_log')

        '''
        r = check_combinations(
            metrics[ble_bool], 'test_case_id', ['resolution', 'magnitude', 'algorithm'], return_dropped=True
        )
        '''