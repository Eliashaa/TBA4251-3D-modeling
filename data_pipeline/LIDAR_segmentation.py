import numpy as np

def plane_from_points(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)

    norm = np.linalg.norm(normal)

    if norm < 1e-10:
        return None

    normal = normal / norm

    d = -np.dot(normal, p1)

    return np.array([*normal, d])

def point_plane_distance(points, plane):
    normal = plane[:3]
    d = plane[3]

    return np.abs(points @ normal + d)

def ransac_plane(points, iterations= 1000, threshold = 0.15):
    best_plane = None
    best_inliers = np.Array([],dtype=int)

    for _ in iterations: 
        random_indices = np.random.choice(len(points), 3, replace=False)

        p1, p2, p3 = points[random_indices]

        plane = plane_from_points(p1, p2, p3)

        if plane is None:
            continue

        distances = point_plane_distance(points, plane)

        inliers = np.where(distances < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = plane

    return best_inliers, best_plane


def fit_plane_pca(points):
    centroid = np.mean(points, axis=0)

    centered = points - centroid

    _, _, vh = np.linalg.svd(centered)

    normal = vh[-1]

    normal = normal / np.linalg.norm(normal)

    d = -np.dot(normal, centroid)

    return np.array([*normal, d])

def segment_planes(
    points,
    min_points=30,
    iterations=1000,
    threshold=0.15
):  
    remaining = points.copy()
    segments = []

    while len(remaining) >= min_points:

        plane, inliers = ransac_plane(
            remaining,
            iterations=iterations,
            threshold=threshold
        )

        if plane is None or len(inliers) < min_points:
            break

        segment_points = remaining[inliers]

        refined_plane = fit_plane_pca(segment_points)

        if abs(refined_plane[2]) < 0.2:
            remaining = np.delete(remaining, inliers, axis=0)
            continue

        segments.append({
            "points": segment_points,
            "plane": refined_plane
        })

        remaining = np.delete(remaining, inliers, axis=0)

    return segments

