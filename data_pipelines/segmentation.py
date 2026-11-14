import os
import geopandas as gpd
import laspy
import numpy as np
from shapely import STRtree, snap
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point

import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points


def snap_endpoint_to_target(
    line: LineString,
    target,
    tolerance: float,
) -> LineString:

    if (
        line is None
        or line.is_empty
        or line.geom_type != "LineString"
        or target is None
        or target.is_empty
    ):
        return line

    coords = list(line.coords)

    for position in (0, -1):
        endpoint = Point(coords[position])
        nearest = nearest_points(endpoint, target)[1]

        if endpoint.distance(nearest) <= tolerance:
            coords[position] = nearest.coords[0]

    return LineString(coords)

def snap_geometries(
    monelinje: gpd.GeoDataFrame,
    bygningslinje: gpd.GeoDataFrame,
    takkantlinjer: gpd.GeoSeries,
    tolerance: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

    mone_resultat = monelinje.copy()
    bygning_resultat = bygningslinje.copy()

    if monelinje.crs != bygningslinje.crs:
        raise ValueError("monelinje og bygningslinje må ha samme CRS")

    if takkantlinjer.crs != monelinje.crs:
        raise ValueError("takkantlinjer må ha samme CRS som linjelagene")

    gyldige_takkanter = takkantlinjer[
        takkantlinjer.notna() & ~takkantlinjer.is_empty
    ]

    takkant_union = (
        gyldige_takkanter.union_all()
        if not gyldige_takkanter.empty
        else None
    )

    if takkant_union is not None:
        mone_resultat.geometry = mone_resultat.geometry.apply(
            lambda line: snap_endpoint_to_target(
                line,
                takkant_union,
                tolerance,
            )
        )

        bygning_resultat.geometry = bygning_resultat.geometry.apply(
            lambda line: snap_endpoint_to_target(
                line,
                takkant_union,
                tolerance,
            )
        )

    if not bygning_resultat.empty:
        bygning_union = bygning_resultat.geometry.union_all()

        mone_resultat.geometry = mone_resultat.geometry.apply(
            lambda line: snap_endpoint_to_target(
                line,
                bygning_union,
                tolerance,
            )
        )

    if not mone_resultat.empty:
        mono_union = mone_resultat.geometry.union_all()

        bygning_resultat.geometry = bygning_resultat.geometry.apply(
            lambda line: snap_endpoint_to_target(
                line,
                mono_union,
                tolerance,
            )
        )

    return mone_resultat, bygning_resultat
    



def segment_rooftiles(fkb_data: dict[str, object]) -> gpd.GeoSeries:
    takkant = fkb_data.get("takkant")

    if takkant is None or takkant.empty:
        return gpd.GeoSeries([], crs=getattr(takkant, "crs", None))

    takkant_lines = takkant.geometry.force_2d()
    takkant_polygon = takkant_lines.polygonize()

    if fkb_data.get("monelinje").empty or fkb_data.get("bygningslinje").empty:
        return takkant_polygon
    
    return takkant_polygon
