"""
merge and concat skill and compute metrics with spatial covariate data for fim100 PI7

Usage:
python3 merge_metrics_for_fim100_pi7.py -d /path_to_foss_fim/foss_fim/misc/resolution_analysis -w
"""
from typing import Literal

import os
import argparse

import pandas as pd
import geopandas as gpd
from tqdm import tqdm


data_dir = os.path.join(
    os.path.expanduser("~"), "data", "foss_fim", "misc", "resolution_analysis"
)

resolutions = [3, 5, 10]


def concat_skill_metrics(data_dir, resolutions):
    """
    Concatenate skill metrics for different resolutions.
    """

    metrics = pd.concat(
        [
            pd.read_csv(
                os.path.join(
                    data_dir, "metrics", "skill" ,f'fim100_huc12_{r}m_non_calibrated_master_metrics.csv'
                )
            )
            for r in resolutions
        ]
    )

    return metrics

def load_spatial_covariates(data_dir, type: Literal['test', 'hucs']):
    """
    Load spatial covariates from the specified directory.
    """
    if type == 'test':
        covariates = gpd.read_file(
            os.path.join(
                data_dir, "spatial_covariates", "test_sites","test_sites_spatial_covariates.gpkg"
            )
        )
    elif type == 'hucs':
        covariates = gpd.read_file(
            os.path.join(
                data_dir, "spatial_covariates", "hucs", "hucs_spatial_covariates.gpkg"
            )
        )
    return covariates

def merge_skill_metrics_with_covariates(skill_metrics, covariates):
    """
    Merge skill metrics with spatial covariates.
    """
    merged = skill_metrics.merge(
        covariates,
        left_on='stac_item_id',
        right_on='id',
        how='inner',
        validate='many_to_one'
    )

    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=covariates.crs)

    return merged

def concat_compute_metrics(data_dir, resolutions):
    """
    Concatenate compute metrics for different resolutions.
    """
    compute_metrics = pd.concat([
        gpd.read_file(
            os.path.join(
                data_dir, "metrics", "compute", f'fim100_huc12_{res}m_completion_map_8_21_2025.gpkg'
            )
        ).assign(resolution_m=res).drop(columns='geometry')
        for res in resolutions
    ])

    compute_metrics = compute_metrics.astype({'resolution_m': 'int32'})

    return compute_metrics

def merge_compute_metrics_with_covariates(covariates, compute_metrics):
    """
    Merge compute metrics with spatial covariates.
    """
    return compute_metrics.merge(
        covariates.drop(columns='geometry', errors='ignore'),
        left_on='HUC12', right_on='HUC12', how='left', validate='many_to_one'
    )


'''
def merge_compute_metrics_with_covariates(covariates, resolutions):
    """
    Merge compute metrics with spatial covariates.
    """
    
    compute_metrics, merged_compute_metrics = {}, {}
    for res in tqdm(resolutions, desc="Merging compute metrics with covariates"):
        compute_metrics[res] = gpd.read_file(
            os.path.join(
                data_dir, "metrics", "compute", f'fim100_huc12_{res}m_completion_map_8_21_2025.gpkg'
            )
        )

        merged_compute_metrics[res] = compute_metrics[res].merge(
            covariates.drop(columns='geometry'), left_on='HUC12', right_on='HUC12', how='left', validate='one_to_one'
        )

    return compute_metrics, merged_compute_metrics
'''

def main(
    data_dir, resolutions, write_merges
):
    # Concat and load skill metrics and spatial covariates
    print("Loading skill metrics ...")
    skill_metrics = concat_skill_metrics(data_dir, resolutions)

    print("Loading compute metrics ...")
    compute_metrics = concat_compute_metrics(data_dir, resolutions)

    print("Loading spatial covariates for test cases and hucs ...")
    covariates_test_cases = load_spatial_covariates(data_dir, type='test')
    covariates_hucs = load_spatial_covariates(data_dir, type='hucs')

    print(type(skill_metrics), type(covariates_test_cases))

    # merging
    print("Merging skill metrics with spatial covariates ...")
    merged_skill_metrics = merge_skill_metrics_with_covariates(skill_metrics, covariates_test_cases)
    print("Merging compute metrics with spatial covariates ...")
    #compute_metrics, merged_compute_metrics = merge_compute_metrics_with_covariates(covariates_hucs, resolutions)
    merged_compute_metrics = merge_compute_metrics_with_covariates(covariates_hucs, compute_metrics)

    print(type(skill_metrics), type(covariates_test_cases), type(merged_skill_metrics))

    # Save merged skill metrics
    if write_merges:
        # Save merged skill metrics to file
        #skill_metrics_output = os.path.join(data_dir, "metrics", "skill", "merged_skill_metrics_with_covariates.parquet")
        skill_metrics_output = os.path.join(data_dir, "metrics", "skill", "merged_skill_metrics_with_covariates.gpkg")
        print(f"Saving merged skill metrics to {skill_metrics_output} ...")
        if os.path.exists(skill_metrics_output): os.remove(skill_metrics_output)
        #merged_skill_metrics.to_parquet(skill_metrics_output, index=False)
        breakpoint()
        merged_skill_metrics.to_file(skill_metrics_output, index=False)

        """
        # Save merged compute metrics to file for legacy support
        compute_metrics_output = os.path.join(
            data_dir, "metrics", "compute", "merged_compute_metrics_with_covariates_{}m.gpkg"
        )
        print(f"Saving merged compute metrics to {compute_metrics_output} ...")
        for res in tqdm(resolutions, desc="Saving merged compute metrics by resolution"):
            merged_compute_metrics[res].to_file(compute_metrics_output.format(res), index=False)
        """
        # Save merged compute metrics to file
        compute_metrics_output = os.path.join(
            data_dir, "metrics", "compute", "merged_compute_metrics_with_covariates.parquet"
        )
        print(f"Saving merged compute metrics to {compute_metrics_output} ...")
        if os.path.exists(compute_metrics_output): os.remove(compute_metrics_output)
        merged_compute_metrics.to_parquet(compute_metrics_output, index=False)

    return {
        "merged_skill_metrics": merged_skill_metrics,
        "merged_compute_metrics": merged_compute_metrics,
        "skill_metrics": skill_metrics,
        "compute_metrics": compute_metrics,
        "covariates_test_cases": covariates_test_cases,
        "covariates_hucs": covariates_hucs,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge skill and compute metrics with spatial covariates for fim100 PI7"
    )
    parser.add_argument(
        "-d", "--data_dir",
        type=str,
        default=data_dir,
        help="Directory containing the data files",
    )
    parser.add_argument(
        "-r", "--resolutions",
        type=int,
        nargs='+',
        default=resolutions,
        help="List of resolutions to process",
    )
    parser.add_argument(
        "-w", "--write_merges",
        action='store_true',
        help="Write merged skill and compute metrics to files",
    )
    results = main(**vars(parser.parse_args()))
    breakpoint()

