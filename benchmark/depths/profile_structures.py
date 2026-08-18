import time
import rioxarray as rxr
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
from exactextract import exact_extract

print("Loading agreement map...")
t0 = time.time()
agreement_map = rxr.open_rasterio('/Users/bradfordbates/Desktop/depth_evals/sample_data/agreement_30m.tif', masked=True).squeeze().compute()
print(f"  Done: {time.time()-t0:.1f}s  shape={agreement_map.shape} crs={agreement_map.rio.crs}")

print("Reprojecting agreement map to EPSG:4326...")
t1 = time.time()
agreement_map_wgs84 = agreement_map.rio.reproject("EPSG:4326")
print(f"  Done: {time.time()-t1:.1f}s")

print("Polygonizing agreement domain...")
t2 = time.time()
valid_mask = agreement_map_wgs84.notnull().values.astype(np.uint8)
transform = agreement_map_wgs84.rio.transform()
domain_polys = [shape(geom) for geom, val in shapes(valid_mask, mask=valid_mask == 1, transform=transform) if val == 1]
domain_polygon = unary_union(domain_polys)
res = abs(transform.a)
domain_simple = domain_polygon.simplify(res * 2).buffer(res * 3)
print(f"  Done: {time.time()-t2:.1f}s  polygons={len(domain_polys)}")

print("Loading structures within buffered domain...")
t3 = time.time()
domain_mask = gpd.GeoDataFrame(geometry=[domain_simple], crs="EPSG:4326")
domain_structures = gpd.read_file(
    '/Users/bradfordbates/Desktop/depth_evals/building_footprints/Deliverable20250606TX/TX_Structures.gdb',
    mask=domain_mask,
)
print(f"  Done: {time.time()-t3:.1f}s  count={len(domain_structures)}")

print("Running zonal stats...")
t4 = time.time()
stats = exact_extract(agreement_map_wgs84, domain_structures, ["mean", "min", "max", "count"], include_cols=[], output="pandas")
print(f"  Done: {time.time()-t4:.1f}s")

print("Computing metrics...")
t5 = time.time()
domain_structures = domain_structures.copy()
domain_structures["mean_depth_diff"] = stats["mean"].fillna(0.0)
domain_structures["min_depth_diff"] = stats["min"].fillna(0.0)
domain_structures["max_depth_diff"] = stats["max"].fillna(0.0)
domain_structures["pixel_count"] = stats["count"]
domain_structures = domain_structures[domain_structures["pixel_count"] > 0].copy()

diff = domain_structures["mean_depth_diff"].values
abs_diff = np.abs(diff)
n = len(domain_structures)

FT_TO_M = 0.3048
buckets = pd.cut(
    abs_diff,
    bins=[0, 1 * FT_TO_M, 3 * FT_TO_M, 5 * FT_TO_M, np.inf],
    labels=["< 1ft", "1-3ft", "3-5ft", "> 5ft"],
    include_lowest=True,
)
domain_structures["agreement_bucket"] = buckets.astype(str)
domain_structures["bias_direction"] = np.where(
    diff > FT_TO_M * 0.1, "over",
    np.where(diff < -FT_TO_M * 0.1, "under", "match"),
)

n_within_1ft = int(np.sum(abs_diff < 1 * FT_TO_M))
n_within_3ft = int(np.sum(abs_diff < 3 * FT_TO_M))
n_within_5ft = int(np.sum(abs_diff < 5 * FT_TO_M))
n_gt_5ft = int(np.sum(abs_diff >= 5 * FT_TO_M))
n_over = int(np.sum(diff > FT_TO_M * 0.1))
n_under = int(np.sum(diff < -FT_TO_M * 0.1))
n_match = n - n_over - n_under

print(f"  Done: {time.time()-t5:.1f}s  final={n}")

print(f"\n--- Summary ---")
print(f"Structures in domain: {n}")
print(f"MAE:    {np.mean(abs_diff):.3f}m")
print(f"RMSE:   {np.sqrt(np.mean(diff**2)):.3f}m")
print(f"Median AE: {np.median(abs_diff):.3f}m")
print(f"P90 AE:    {np.percentile(abs_diff, 90):.3f}m")
print(f"Max AE:    {np.max(abs_diff):.3f}m")
print(f"Mean signed error: {np.mean(diff):.3f}m")
print(f"\nAgreement buckets:")
print(f"  < 1ft:  {n_within_1ft} ({n_within_1ft/n*100:.1f}%)")
print(f"  1-3ft:  {n_within_3ft - n_within_1ft} ({(n_within_3ft - n_within_1ft)/n*100:.1f}%)")
print(f"  3-5ft:  {n_within_5ft - n_within_3ft} ({(n_within_5ft - n_within_3ft)/n*100:.1f}%)")
print(f"  > 5ft:  {n_gt_5ft} ({n_gt_5ft/n*100:.1f}%)")
print(f"\nBias direction:")
print(f"  Over:  {n_over} ({n_over/n*100:.1f}%)")
print(f"  Under: {n_under} ({n_under/n*100:.1f}%)")
print(f"  Match: {n_match} ({n_match/n*100:.1f}%)")

print("\nExporting GPKG...")
t6 = time.time()
domain_structures.to_file('/Users/bradfordbates/Desktop/depth_evals/sample_data/structures_30m.gpkg', driver="GPKG", layer="structures")
print(f"  Done: {time.time()-t6:.1f}s  exported={len(domain_structures)}")

print(f"\nTOTAL: {time.time()-t0:.1f}s")
