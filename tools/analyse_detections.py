"""
take detection and ground truth and look for the false positives and false negatives

This produces a csv file with the false positives because they are what we need to inspect

# TODO look for false negatives
Similar to Kellenberger - maybe this one Benjamin, et al. "Detecting and classifying elephants in the wild." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

"""
from time import sleep

import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from pyproj import CRS
from shapely.geometry import Point
from scipy.spatial import cKDTree

crs = CRS.from_proj4("+proj=cart +ellps=WGS84 +units=m +no_defs")

df_detections = pd.read_csv('/home/christian/hnee/HerdNet/data_iguana/val/20240824_HerdNet_results/20240824_detections.csv')
gdf_detections = gpd.GeoDataFrame(df_detections, geometry=gpd.points_from_xy(df_detections.x, df_detections.y))
gdf_detections = gdf_detections.set_crs(crs)
gdf_detections['buffer'] = gdf_detections.geometry.buffer(150)
# df_detections_all = gdf_detections.set_geometry('buffer')
gdf_detections_all = gdf_detections.copy()

df_ground_truth = pd.read_csv('/home/christian/hnee/HerdNet/data_iguana/val/herdnet_format.csv')
gdf_ground_truth = gpd.GeoDataFrame(df_ground_truth, geometry=gpd.points_from_xy(df_ground_truth.x, df_ground_truth.y))
gdf_ground_truth_all = gdf_ground_truth.set_crs(crs)
# gdf_ground_truth_all = gdf_ground_truth.set_geometry('geometry')

image_list = df_ground_truth['images'].unique()
l_fp = []

for i in image_list:
    gdf_ground_truth = gdf_ground_truth_all[
        gdf_ground_truth_all['images'] == i]  # TODO proof of concept

    gdf_detections = gdf_detections_all[gdf_detections_all['images'] == i]  # TODO proof of concept

    gt = gdf_ground_truth.copy()
    gt_coords = gt.geometry.apply(lambda geom: (int(geom.x), int(geom.y))).tolist()

    pred = gdf_detections.copy()
    pred_coords = pred.geometry.apply(lambda geom: (int(geom.x), int(geom.y))).tolist()

    gt_tree = cKDTree(gt_coords)

    # Query the nearest distance from each point in pred to any point in gt
    distances, indices = gt_tree.query(pred_coords, k=1)

    # Filter the points in pred where the nearest distance is greater than 150
    pred_distant = pred[distances > 150]

    # build a dataframe I can use to plot the false posive detections
    df_false_positves = pred_distant[['images', 'x', 'y', 'species', 'labels']]
    l_fp.append(df_false_positves)

    # gdf_detections.drop(columns=['images', 'labels', 'species', 'count_1'], inplace=True)
    # gdf_both = gpd.sjoin(gdf_ground_truth, gdf_detections, how='inner', predicate='intersects')
    # gdf_both_outer = gpd.sjoin(gdf_ground_truth, gdf_detections, how='inner', predicate='intersects')
    #
    # if len(gdf_both) == 0:
    #     print('no detections')
    # else:
    #     gdf_both = gdf_both.set_geometry('geometry_right')
    #     intersecting_indices = gdf_both.index
    #
    #
    #     df_false_negative = gdf_ground_truth[~gdf_ground_truth.index.isin(intersecting_indices)]
    #
    #
    #
    #     gdf_both.plot()
    #     plt.show()
    #
    #     sleep(1)

pd.concat(l_fp).to_csv('/home/christian/hnee/HerdNet/data_iguana/val/20240824_HerdNet_results/false_positives.csv', index=False)