
import os
import argparse

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm


def median_slope_histogram(covariates, area, output_dir):

    # covariates is a dataframe in memory
    n = len(covariates)

    if area == 'huc':
        title = f'Median Slope Histogram (FIM100 HUC12s)'
        xlim = (0, 0.5)
        bins = 100
    elif area == 'benchmarks':
        title = f'Median Slope Histogram (FIM100 Benchmarks)'
        xlim = (0, 0.2)
        bins = 50

    plt.hist(covariates['median_slope'], bins=bins)

    # x and y labels
    plt.xlabel('Median Slope, rise/run (m/m)')
    plt.ylabel('Frequency')

    plt.title(title)
    plt.xlim(xlim)

    # plot the statistics
    plt.text(
        0.7, 0.8,
        f'Mean: {covariates["median_slope"].mean():.4f}\n'
        f'Median: {covariates["median_slope"].median():.4f}\n'
        f'Std: {covariates["median_slope"].std():.4f}\n'
        f'Min: {covariates["median_slope"].min():.4f}\n'
        f'Max: {covariates["median_slope"].max():.4f}\n'
        f'n: {n}',
        fontsize=14, ha='center', va='center', transform=plt.gca().transAxes
    )

    # save the plot
    plt.savefig(os.path.join(output_dir, f'median_slope_histogram_{area}.png'))

    # close the plot
    plt.close()

def freq_high_dev_histogram(covariates, area, output_dir):

    n = len(covariates)

    if area == 'huc':
        title = f'High Developed LC Frequency Histogram (FIM100 HUC12s)'
        bins = 100
        xlim = (0, 0.01)
    elif area == 'benchmarks':
        title = f'High Developed LC Frequency Histogram (FIM100 Benchmarks)'
        bins = 50
        xlim = (0, 0.025)

    # bool mask between 0 and 0.05
    bool_mask = (covariates['freq_high_dev'] >= 0) & (covariates['freq_high_dev'] <= 0.025)

    plt.hist(covariates.loc[bool_mask, 'freq_high_dev'], bins=bins)

    # x and y labels
    plt.xlabel('High Developed LC Frequency')
    plt.ylabel('Frequency')

    plt.title(title)

    # set x limit
    plt.xlim(xlim)

    # plot the statistics
    plt.text(
        0.7, 0.8, f'Mean: {covariates["freq_high_dev"].mean():.4f}\n'
        f'Median: {covariates["freq_high_dev"].median():.4f}\n'
        f'Std: {covariates["freq_high_dev"].std():.4f}\n'
        f'Min: {covariates["freq_high_dev"].min():.4f}\n'
        f'Max: {covariates["freq_high_dev"].max():.4f}\n'
        f'n: {n}',
        fontsize=14, ha='center', va='center', transform=plt.gca().transAxes
    )

    # save the plot
    plt.savefig(os.path.join(output_dir, f'freq_high_dev_histogram_{area}.png'))

    # close the plot
    plt.close()

def slope_and_freq_scatter(covariates, area, output_dir):

    n = len(covariates)

    covariates.plot.scatter(x='median_slope', y='freq_high_dev')

    # x and y labels
    plt.xlabel('Median Slope, rise/run (m/m)')
    plt.ylabel('High Developed LC Frequency')

    if area == 'huc':
        title = f'High Developed LC vs Median Slope (FIM100 HUC12s)'
    elif area == 'benchmarks':
        title = f'High Developed LC vs Median Slope (FIM100 Benchmarks)'
    
    plt.title(title)

    # save
    plt.savefig(os.path.join(output_dir, f'slope_and_freq_scatter_{area}.png'))

    # close
    plt.close()


def slope_and_freq_scatter_log(covariates, area, output_dir):

    covariates['median_slope_log'] = np.log10(covariates.loc[~np.isclose(covariates['median_slope'], 0),'median_slope'])
    covariates['freq_high_dev_log'] = np.log10(covariates.loc[~np.isclose(covariates['freq_high_dev'], 0),'freq_high_dev'])

    # drop rows with missing values
    covariates = covariates.dropna(subset=['median_slope_log', 'freq_high_dev_log'])

    n = len(covariates)

    # scatter plot
    covariates.plot.scatter(x='median_slope_log', y='freq_high_dev_log')

    # x and y labels
    plt.xlabel('Log of Median Slope, rise/run (m/m)')
    plt.ylabel('Log of High Developed LC Frequency')
    
    if area == 'huc':
        title = f'Log-log: High Developed LC vs Median Slope (FIM100 HUC12s)'
    elif area == 'benchmarks':
        title = f'Log-log: High Developed LC vs Median Slope (FIM100 Benchmarks)'
    plt.title(title)

    # compute and plot correlation
    corr = covariates['median_slope_log'].corr(covariates['freq_high_dev_log'])
    plt.text(0.25, 0.1, f'Correlation: {corr:.2f}', fontsize=14, ha='center', va='center', transform=plt.gca().transAxes)

    # compute regression line with scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        covariates['median_slope_log'].values, covariates['freq_high_dev_log'].values
    )

    # plot the regression line
    x = np.linspace(covariates['median_slope_log'].min(), covariates['median_slope_log'].max(), 100)
    y = slope * x + intercept
    plt.plot(x, y, 'r', label='y={:.2f}x{:.2f}'.format(slope, intercept))

    # plot parameters and formula
    plt.text(0.25, 0.2, f'y={slope:.2f}x+{intercept:.2f}', fontsize=14, ha='center', va='center', transform=plt.gca().transAxes)

    # save the plot
    plt.savefig(os.path.join(output_dir, f'slope_and_freq_scatter_log_{area}.png'))

    # close the plot
    plt.close()

def main(huc_covariates_fn, benchmarks_covariates_fn, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    covariates_fn_list = [
        (huc_covariates_fn, 'huc'),
        (benchmarks_covariates_fn, 'benchmarks')
    ]
    
    for cov, area in tqdm(covariates_fn_list, desc='Spatial covariates analysis'):
        # read the spatial covariates
        covariates = gpd.read_file(cov)

        '''
        if area == 'benchmarks':
            # take median by test_case_id
            # This maybe no longer necessary since covariates are non-unique now.
            covariates = (
                covariates
                .groupby('test_case_id')[['median_slope','freq_high_dev']]
                .median()
                .reset_index()
                .merge(
                    covariates
                    .drop(
                        columns=['median_slope','freq_high_dev','magnitude','extent_file']),
                        how='left',
                        on='test_case_id'
                    )
                    .drop_duplicates(
                        subset=['test_case_id', 'median_slope', 'freq_high_dev']
                    )
            )
        '''

        # plot the median slope histogram
        median_slope_histogram(covariates, area, output_dir)

        # plot the freq_high_dev histogram
        freq_high_dev_histogram(covariates, area, output_dir)

        # plot the scatter plot
        slope_and_freq_scatter(covariates, area, output_dir)

        # plot the scatter plot in log
        slope_and_freq_scatter_log(covariates, area, output_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Spatial covariates analysis')
    parser.add_argument(
        '-u', '--huc_covariates_fn',
        type=str,
        required=True,
        help='Path to the HUC covariates file'
    )
    parser.add_argument(
        '-b', '--benchmarks_covariates_fn',
        type=str,
        required=True,
        help='Path to the benchmarks covariates file'
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Directory to save the output plots'
    )

    # example usage: python3 covariate_analysis.py -u ~/data/foss_fim/misc/resolution_analysis/spatial_covariates/hucs/hucs_spatial_covariates.gpkg -b ~/data/foss_fim/misc/resolution_analysis/spatial_covariates/test_sites/test_sites_spatial_covariates.gpkg -o ~/data/foss_fim/misc/resolution_analysis/plots/covariates_analysis

    main(**vars(parser.parse_args()))

