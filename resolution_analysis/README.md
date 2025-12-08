


### Scripts

Should be ran in the following order:

- `get_spatial_covariates.py`
  - Retrieves spatial covariates by HUC8 for CONUS.
- `clip_pr_hi_ak.sh`
  - Prepares PR, HI, AK spatial covariates by HUC8. May need to move final outputs manually.
- `compute_spatial_covariates_huc8s.gpkg`
  - Initializes `huc8s_sp.gpkg` by computing aggregagte covariates by HUC8.
  - Data dictionary:
    - 'HUC8', 'geometry', 'median_slope', 'missing_median_slope', 'freq_high_dev', 'missing_freq_high_dev'
- `make_agreement_map_and_covariate_df.py`
  - Initializes benchmark vector file `benchmarks_sp.gpkg` by getting benchmark extent geometries.
  - Data dictionary:
    - 'benchmark_source', 'huc8', 'test_case_id', 'geometry'
- `compute_spatial_covariates_for_test_cases.py`
  - Computes spatial covariate aggregations for benchmarks updating `benchmarks_sp.gpkg`.
  - Data dictionary:
    - 'benchmark_source', 'test_case_id', 'median_slope', 'freq_high_dev', 'geometry'
- `make_resolution_availability_map.py`
  - Determines the percent covered by LiDAR tiles (1 or 3m) for each region in benchmarks and HUC8s. Also, computes the union of all 1 and 3m tiles each and writes to `tile_index_availability.gpkg`.
  - Three files:
    - Data dictionary `benchmarks_sp.gpkg`
      - 'benchmark_source', 'test_case_id', 'median_slope', 'freq_high_dev', 'percent_covered_by_tiles', 'geometry'
    - Data dictionary `huc8s_sp.gpkg`
      - 'HUC8', 'median_slope', 'missing_median_slope', 'freq_high_dev', 'missing_freq_high_dev', 'percent_covered_by_tiles', 'geometry'
    - Data dictionary `tile_index_availability.gpkg`
      - 'resolution', 'geometry'
- `covariate_analysis.py`
  - Creates several plots inspecting the distribution and relationships of the two covariates.
- `merge_metrics_csv.py`
  - Adds columns from `benchmarks_sp.gpkg` to metrics CSV file.
  - Data dictionary for metrics csv
    - Same columns as before but added: 'resolution', 'test_case_id', 'median_slope', 'freq_high_dev', 'percent_covered_by_tiles'
- `metric_analysis.py`
  - Creates varies plots to analyze relationships among metrics, covariates, and resolution across benchmark regions.
- `recommend_algorithm.py`
  - Creates GPKG with recommended depression filling algorithm based on exit statuses by HUC8.
