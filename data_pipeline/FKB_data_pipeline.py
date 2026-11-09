import laspy
import os
import numpy as np
import geopandas as gpd
from shapely import STRtree
from typing import Dict
from shapely.geometry import box

def load_fkb_data(gml_path: str) -> Dict[str, gpd.GeoDataFrame]:
    """
    Loads FKB data from a GML file and returns a dictionary containing Takkant, Mønlinje and Bygningslinje as GeodataFrame
    """
    bbox =  (np.float64(566223.87), np.float64(7024412.59), np.float64(574253.35), np.float64(7035250.13)) #Found in exploration section

    takkant = gpd.read_file(
    gml_path,
    layer="Takkant",
    bbox=bbox,
    engine="pyogrio"
    )

    monelinje = gpd.read_file(
        gml_path,
        layer="Mønelinje",
        bbox=bbox,
        engine="pyogrio"
    )

    bygningslinje = gpd.read_file(
        gml_path,
        layer="Bygningslinje",
        bbox=bbox,
        engine="pyogrio"
    )
    return {"Takkant": takkant, "Mønelinje": monelinje, "Bygningslinje": bygningslinje}

def filter_fkb_data(fkb_data: Dict[str, gpd.GeoDataFrame], lidar_path: str):
    """
    Filters FKB data based on a proximity to LiDAR data and returns a dictionary containing Takkant, Mønlinje and Bygningslinje as GeodataFrame
    """

    lidar_boxes = []

    for file in os.listdir(lidar_path):
        las = laspy.read(os.path.join(lidar_path + "/", file))
        lidar_bounds = box(las.header.min_x, las.header.min_y, las.header.max_x, las.header.max_y) 
        lidar_boxes.append(lidar_bounds)

    lidar_tree = STRtree(lidar_boxes)
