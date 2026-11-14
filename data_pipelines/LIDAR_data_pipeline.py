import numpy as np


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
                 min_abs_normal_z=0.1):
    """Returner (inlier-indekser, planparametere) for beste takplan."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return np.array([], dtype=int), None
    if rng is None:
        rng = np.random.default_rng()

    best_inliers = np.array([], dtype=int)
    best_plane = None

    for _ in range(iterations):
        indices = rng.choice(len(points), size=3, replace=False)
        plane = plane_from_points(*points[indices])
        if plane is None or abs(plane[2]) < min_abs_normal_z:
            continue

        inliers = np.flatnonzero(point_plane_distance(points, plane) < threshold)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = plane

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
