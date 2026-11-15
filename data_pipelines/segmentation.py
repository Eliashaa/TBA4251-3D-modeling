import geopandas as gpd
from shapely import force_2d
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points


def snap_endpoint_to_target(
    line: BaseGeometry,
    target: BaseGeometry,
    tolerance: float,
) -> BaseGeometry:
    """Snap the endpoints of line to the closest points on target."""
    if line is None or line.is_empty or target is None or target.is_empty:
        return line

    line = force_2d(line)
    target = force_2d(target)

    if line.geom_type == "MultiLineString":
        return MultiLineString(
            [
                snap_endpoint_to_target(part, target, tolerance)
                for part in line.geoms
            ]
        )

    if line.geom_type != "LineString":
        return line

    coords = list(line.coords)
    if len(coords) < 2:
        return line

    for position in (0, -1):
        endpoint = Point(coords[position])
        nearest = nearest_points(endpoint, target)[1]

        if endpoint.distance(nearest) <= tolerance:
            coords[position] = nearest.coords[0]

    return LineString(coords)


def _valid_union(geometries: gpd.GeoSeries) -> BaseGeometry | None:
    """Return the union of non-empty geometries, or None."""
    valid = geometries[geometries.notna() & ~geometries.is_empty]
    if valid.empty:
        return None
    return valid.union_all()


def _as_2d_lines(geometries: gpd.GeoSeries) -> gpd.GeoSeries:
    """Convert polygon boundaries and line geometries to 2D linework."""
    lines = geometries.force_2d().apply(
        lambda geometry: (
            geometry.boundary
            if geometry is not None
            and not geometry.is_empty
            and geometry.geom_type in {"Polygon", "MultiPolygon"}
            else geometry
        )
    )
    return gpd.GeoSeries(lines, index=geometries.index, crs=geometries.crs)


def snap_geometries(
    monelinje: gpd.GeoDataFrame,
    bygningslinje: gpd.GeoDataFrame,
    takkantlinjer: gpd.GeoSeries,
    tolerance: float = 0.3,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Snap ridge/building line endpoints to roof edges and to each other."""
    if tolerance < 0:
        raise ValueError("tolerance kan ikke være negativ")
    if monelinje.crs != bygningslinje.crs:
        raise ValueError("monelinje og bygningslinje må ha samme CRS")
    if takkantlinjer.crs != monelinje.crs:
        raise ValueError("takkantlinjer må ha samme CRS som linjelagene")

    mone_resultat = monelinje.copy()
    bygning_resultat = bygningslinje.copy()
    mone_resultat.geometry = mone_resultat.geometry.force_2d()
    bygning_resultat.geometry = bygning_resultat.geometry.force_2d()
    takkantlinjer = _as_2d_lines(takkantlinjer)

    # The roof boundary is the fixed anchor and is never moved.
    takkant_union = _valid_union(takkantlinjer)
    if takkant_union is not None:
        mone_resultat.geometry = mone_resultat.geometry.apply(
            lambda line: snap_endpoint_to_target(line, takkant_union, tolerance)
        )
        bygning_resultat.geometry = bygning_resultat.geometry.apply(
            lambda line: snap_endpoint_to_target(line, takkant_union, tolerance)
        )

    # Connect ridge lines to building lines where they are close enough.
    bygning_union = _valid_union(bygning_resultat.geometry)
    if bygning_union is not None:
        mone_resultat.geometry = mone_resultat.geometry.apply(
            lambda line: snap_endpoint_to_target(line, bygning_union, tolerance)
        )

    # A building-line endpoint may instead be the loose side of a connection.
    mono_union = _valid_union(mone_resultat.geometry)
    if mono_union is not None:
        bygning_resultat.geometry = bygning_resultat.geometry.apply(
            lambda line: snap_endpoint_to_target(line, mono_union, tolerance)
        )

    return mone_resultat, bygning_resultat


def segment_rooftiles(
    fkb_data: dict[str, object],
    tolerance: float = 0.3,
) -> gpd.GeoSeries:
    """Create roof polygons from roof edges, ridge lines and building lines."""
    takkant = fkb_data.get("takkant")
    if not isinstance(takkant, gpd.GeoDataFrame) or takkant.empty:
        return gpd.GeoSeries(
            [],
            dtype="geometry",
            crs=getattr(takkant, "crs", None),
        )

    takkant_lines = _as_2d_lines(takkant.geometry)
    roof_polygons = takkant_lines.polygonize(node=True)
    if roof_polygons.empty:
        return roof_polygons

    def empty_lines() -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([], dtype="geometry", crs=takkant.crs),
            crs=takkant.crs,
        )

    monelinje = fkb_data.get("monelinje")
    if not isinstance(monelinje, gpd.GeoDataFrame):
        monelinje = empty_lines()

    bygningslinje = fkb_data.get("bygningslinje")
    if not isinstance(bygningslinje, gpd.GeoDataFrame):
        bygningslinje = empty_lines()

    snappet_mone, snappet_bygning = snap_geometries(
        monelinje,
        bygningslinje,
        takkant_lines,
        tolerance=tolerance,
    )

    linework = gpd.GeoSeries(
        [
            *takkant_lines.tolist(),
            *snappet_mone.geometry.tolist(),
            *snappet_bygning.geometry.tolist(),
        ],
        crs=takkant.crs,
    )
    linework = linework[linework.notna() & ~linework.is_empty]
    segmented = linework.polygonize(node=True)

    # Discard polygons created by linework outside the original roof outline.
    roof_area = roof_polygons.union_all()
    inside_roof = segmented.representative_point().apply(roof_area.covers)
    return segmented.loc[inside_roof].reset_index(drop=True)
