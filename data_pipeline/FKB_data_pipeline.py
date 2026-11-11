import os
from typing import Dict

import geopandas as gpd
import laspy
import numpy as np
from shapely import STRtree
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


def load_fkb_data(
    gml_path: str,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Loads FKB data from a GML file and returns a dictionary of GeoDataFrames.
    """

    bbox = (
        566223.87,
        7024412.59,
        574253.35,
        7035250.13,
    )

    takkant = gpd.read_file(
        gml_path,
        layer="Takkant",
        bbox=bbox,
        engine="pyogrio",
    )

    monelinje = gpd.read_file(
        gml_path,
        layer="Mønelinje",
        bbox=bbox,
        engine="pyogrio",
    )

    bygningslinje = gpd.read_file(
        gml_path,
        layer="Bygningslinje",
        bbox=bbox,
        engine="pyogrio",
    )

    return {
        "Takkant": takkant,
        "Mønelinje": monelinje,
        "Bygningslinje": bygningslinje,
    }
def monelinje_match(
    monelinje_lines: gpd.GeoDataFrame,
    takkant_lines: gpd.GeoDataFrame,
    tolerance: float = 0.05,
) -> gpd.GeoDataFrame:
    """
    returns the monelinje lines that are within a certain tolerance of the takkant lines.
    """

    if monelinje_lines.empty or takkant_lines.empty:
        return monelinje_lines.iloc[0:0].copy()
    
    takkant_polygons = takkant_lines.geometry.polygonize(
        node=True
    )

    if takkant_polygons.empty:
        return monelinje_lines.iloc[0:0].copy()

    roof_geometry = takkant_polygons.union_all()

    roof_search_area = roof_geometry.buffer(tolerance)

    mask = monelinje_lines.geometry.intersects(
        roof_search_area
    )

    return monelinje_lines.loc[mask].copy()

def takkant_match(
    takkant_lines: gpd.GeoDataFrame,
    lidar_bbox: BaseGeometry,
    tolerance: float = 0.05,
) -> gpd.GeoDataFrame:
    """
    Chooses the best matching takkant lines for a given lidar bounding box.
    """

    if takkant_lines.empty:
        return takkant_lines.copy()

    polygons = takkant_lines.geometry.polygonize(
        node=True
    )

    if polygons.empty:
        return takkant_lines.iloc[0:0].copy()

    def polygon_score(polygon):
        intersection_area = polygon.intersection(
            lidar_bbox
        ).area

        union_area = polygon.union(lidar_bbox).area

        overlap_score = (
            intersection_area / union_area
            if union_area > 0
            else 0.0
        )

        centroid_distance = polygon.centroid.distance(
            lidar_bbox.centroid
        )

        # max() velger høyest overlapp først.
        # Ved lik overlapp velges kortest avstand.
        return overlap_score, -centroid_distance

    best_polygon = max(
        polygons,
        key=polygon_score,
    )

    boundary_area = best_polygon.boundary.buffer(
        tolerance
    )

    def belongs_to_boundary(line):
        if (
            line is None
            or line.is_empty
            or line.length == 0
        ):
            return False

        matching_length = line.intersection(
            boundary_area
        ).length

        matching_fraction = (
            matching_length / line.length
        )

        return matching_fraction >= 0.8

    mask = takkant_lines.geometry.apply(
        belongs_to_boundary
    )

    return takkant_lines.loc[mask].copy()


def filter_fkb_data(
    fkb_data: dict[str, gpd.GeoDataFrame],
    lidar_path: str,
) -> dict[str, dict[str, object]]:
    """
    Filters FKB data based on the bounding boxes of LiDAR files.
    Returns a dictionary with LiDAR file names as keys and dictionaries containing
    the LiDAR data and filtered FKB data as values."""

    takkant = fkb_data["Takkant"]
    monelinje = fkb_data["Mønelinje"]
    bygningslinje = fkb_data["Bygningslinje"]

    takkant_tree = STRtree(
        takkant.geometry.to_numpy()
    )

    monelinje_tree = STRtree(
        monelinje.geometry.to_numpy()
    )

    bygningslinje_tree = STRtree(
        bygningslinje.geometry.to_numpy()
    )

    filtered_fkb = {}

    for file in os.listdir(lidar_path):
        if not file.lower().endswith((".las", ".laz")):
            continue

        file_path = os.path.join(lidar_path, file)
        las = laspy.read(file_path)

        lidar_bbox = box(
            np.min(las.x),
            np.min(las.y),
            np.max(las.x),
            np.max(las.y),
        )

        takkant_candidate_indices = takkant_tree.query(
            lidar_bbox,
            predicate="intersects",
        )


        bygningslinje_indices = (
            bygningslinje_tree.query(
                lidar_bbox,
                predicate="intersects",
            )
        )

        takkant_candidates = takkant.iloc[
            takkant_candidate_indices
        ].copy()

        matched_takkant = takkant_match(
            takkant_candidates,
            lidar_bbox,
        )

        monelinje_indices = monelinje_tree.query(
            lidar_bbox,
            predicate="intersects",
        )
        
        matched_monelinje = monelinje_match(
            monelinje.iloc[monelinje_indices],
            matched_takkant,
        )
        

        filtered_fkb[file] = {
            "lidar": las,
            "bbox": lidar_bbox,
            "takkant": matched_takkant,
            "monelinje": matched_monelinje,
            "bygningslinje": bygningslinje.iloc[
                bygningslinje_indices
            ].copy(),
        }

    return filtered_fkb