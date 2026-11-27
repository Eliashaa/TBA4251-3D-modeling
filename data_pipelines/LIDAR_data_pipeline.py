import geopandas as gpd
import numpy as np
from shapely import points as xy_points


def plane_from_points(p1, p2, p3):
    """Returner normalisert plan (a, b, c, d), eller None for kollineære punkter."""
    normal = np.cross(p2 - p1, p3 - p1)
    length = np.linalg.norm(normal)
    if length < 1e-10:
        return None
    normal = normal / length
    d = -np.dot(normal, p1)
    return np.array([*normal, d])

def point_plane_distance(points, plane):
    """Vinkelrett avstand fra hvert 3D-punkt til planet."""
    normal = plane[:3]
    return np.abs(points @ normal + plane[3]) / np.linalg.norm(normal)

def ransac_plane(points, iterations=1000, threshold=0.10, rng=None,
                 min_abs_normal_z=0.1, *, min_sample_area=0.0, xy_bins=None,
                 min_inliers=3, min_coverage=0.0):
    """Returner (inlier-indekser, planparametere) for beste takplan."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return np.array([], dtype=int), None
    if rng is None:
        rng = np.random.default_rng()

    best_inliers = np.array([], dtype=int)
    best_plane = None
    best_score = None
    occupied_bins = np.unique(xy_bins).size if xy_bins is not None else 1

    for _ in range(iterations):
        indices = rng.choice(len(points), size=3, replace=False)
        if min_sample_area:
            u, v = points[indices[1:], :2] - points[indices[0], :2]
            if abs(u[0] * v[1] - u[1] * v[0]) < 2 * min_sample_area:
                continue
        plane = plane_from_points(*points[indices])
        if plane is None or abs(plane[2]) < min_abs_normal_z:
            continue

        distances = point_plane_distance(points, plane)
        inliers = np.flatnonzero(distances < threshold)
        if len(inliers) < min_inliers:
            continue
        coverage = np.unique(xy_bins[inliers]).size / occupied_bins if xy_bins is not None else 1.0
        if coverage < min_coverage:
            continue
        score = (len(inliers) * np.sqrt(coverage), -np.mean(distances[inliers] ** 2))
        if best_score is None or score > best_score:
            best_inliers = inliers
            best_plane = plane
            best_score = score

    return best_inliers, best_plane


def fit_plane_pca(points):
    """Tilpass plan til alle punktene med SVD/PCA."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError('PCA krever minst tre punkter.')
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    return np.array([*normal, -np.dot(normal, centroid)])


def segment_planes(points, min_points=30, iterations=1000, threshold=0.15,
                   min_abs_normal_z=0.2, seed=42):
    """Segmenter takplan; hvert resultat har points, indices, plane og rmse."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points må ha formen (N, 3).')
    if not np.isfinite(points).all():
        raise ValueError('points inneholder NaN eller uendelige verdier.')
    if min_points < 3 or iterations < 1 or threshold <= 0:
        raise ValueError('Krever min_points >= 3, iterations >= 1 og threshold > 0.')

    remaining_indices = np.arange(len(points))
    segments = []
    rng = np.random.default_rng(seed)

    while len(remaining_indices) >= min_points:
        remaining = points[remaining_indices]

        inliers, plane = ransac_plane(
            remaining, iterations=iterations, threshold=threshold,
            rng=rng, min_abs_normal_z=min_abs_normal_z,
        )
        if plane is None or len(inliers) < min_points:
            break

        refined_plane = fit_plane_pca(remaining[inliers])
        refined_inliers = np.flatnonzero(
            point_plane_distance(remaining, refined_plane) < threshold
        )
        if len(refined_inliers) >= min_points:
            inliers = refined_inliers
            refined_plane = fit_plane_pca(remaining[inliers])

        if abs(refined_plane[2]) >= min_abs_normal_z and len(inliers) >= min_points:
            segment_points = remaining[inliers]
            distances = point_plane_distance(segment_points, refined_plane)
            segments.append({
                'points': segment_points,
                'indices': remaining_indices[inliers].copy(),
                'plane': refined_plane,
                'rmse': float(np.sqrt(np.mean(distances ** 2))),
            })

        remaining_indices = np.delete(remaining_indices, inliers)

    return segments


def fit_lidar_planes_by_roof_segment(
    points, roof_segments: gpd.GeoSeries, *, threshold: float = 0.15,
    min_points: int = 10, min_support_fraction: float = 0.5,
    min_coverage: float = 0.5,
) -> dict[int, dict[str, object]]:
    """Fit a plane independently inside each roof polygon in the LiDAR XY CRS."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite LiDAR points with shape (N, 3).")
    if min_points < 3 or threshold <= 0 or not (
        0 < min_support_fraction <= 1 and 0 <= min_coverage <= 1
    ):
        raise ValueError("Invalid roof plane fitting parameters.")

    tree = gpd.GeoSeries(xy_points(points[:, :2]), crs=roof_segments.crs).sindex
    rng = np.random.default_rng(42)
    matches = {}
    for position, roof in enumerate(roof_segments):
        local = points[tree.query(roof, predicate="contains", sort=True)]
        plane = None
        support = 0
        fraction = coverage = 0.0
        status = "too few interior returns"
        if len(local) >= min_points:
            low = np.array(roof.bounds[:2])
            span = np.array(roof.bounds[2:]) - low
            grid = np.clip(((local[:, :2] - low) / span * 4).astype(int), 0, 3)
            bins = 4 * grid[:, 0] + grid[:, 1]
            occupied = np.unique(bins).size
            required = max(min_points, int(np.ceil(min_support_fraction * len(local))))
            inliers, candidate = ransac_plane(
                local, threshold=threshold, rng=rng, min_abs_normal_z=0.2,
                min_sample_area=0.005 * roof.area, xy_bins=bins,
                min_inliers=required, min_coverage=min_coverage,
            )
            status = "no candidate met slope, support and coverage requirements"
            if candidate is not None:
                plane = candidate
                status = "fitted"
                for _ in range(2):
                    refined = fit_plane_pca(local[inliers])
                    updated = np.flatnonzero(point_plane_distance(local, refined) < threshold)
                    if (abs(refined[2]) < 0.2 or len(updated) < required
                            or np.unique(bins[updated]).size / occupied < min_coverage):
                        break
                    plane = refined
                    if np.array_equal(updated, inliers):
                        break
                    inliers = updated
                support = len(inliers)
                fraction = support / len(local)
                coverage = np.unique(bins[inliers]).size / occupied

        matches[position] = {
            "roof_segment": roof,
            "plane": plane,
            "plane_index": position if plane is not None else None,
            "interior_points": len(local),
            "support_points": support,
            "support_fraction": fraction,
            "coverage": coverage,
            "status": status,
        }
    return matches


def match_lidar_planes_to_segments(
    planes,
    roof_segments: gpd.GeoSeries,
    *,
    min_points: int = 3,
    min_support_fraction: float = 0.5,
) -> dict[int, dict[str, object]]:
    """Match each roof polygon to the plane with the most LiDAR inliers inside it.

    Use planes from segment_planes() and faces from segment_rooftiles().
    LiDAR XY and polygon coordinates must use the same CRS. Points on a shared
    boundary count for each touching face. A plane can match multiple faces.
    Support fraction is the winning plane's share of classified points inside
    the face. Results are keyed by zero-based roof segment position. Ties and
    insufficient support leave both plane and plane_index set to None.
    """
    if min_points < 1 or not 0 <= min_support_fraction <= 1:
        raise ValueError("Require min_points >= 1 and min_support_fraction in [0, 1].")

    counts = np.zeros((len(roof_segments), len(planes)), dtype=int)
    spatial_index = roof_segments.sindex
    for plane_index, plane in enumerate(planes):
        points = np.asarray(plane["points"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"Plane {plane_index} must contain finite (N, 3) points.")

        _, face_positions = spatial_index.query(
            xy_points(points[:, :2]), predicate="intersects"
        )
        counts[:, plane_index] = np.bincount(face_positions, minlength=len(roof_segments))

    matches = {}
    for position, (segment, votes) in enumerate(zip(roof_segments, counts)):
        total = votes.sum()
        best_index = int(votes.argmax()) if total else None
        best_count = int(votes[best_index]) if best_index is not None else 0
        fraction = best_count / total if total else 0.0
        if (
            best_count < min_points
            or fraction < min_support_fraction
            or np.count_nonzero(votes == best_count) != 1
        ):
            best_index = None
        matches[position] = {
            "roof_segment": segment,
            "plane": planes[best_index]["plane"] if best_index is not None else None,
            "plane_index": best_index,
            "support_points": best_count,
            "support_fraction": float(fraction),
        }

    return matches
