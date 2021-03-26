import math
import torch
import unittest

EPSILON: float = 1e-6

def _l2_normalise_rows(t: torch.Tensor):
    """l2-normalise (in place) each row of t: float(n, row_size)"""
    t.div_(t.norm(p = 2, dim = 1, keepdim = True).clamp(min = EPSILON))

def spherical_kmeans(points: torch.Tensor, nb_clusters: int, max_iter: int) -> torch.Tensor:
    """
    Finds nb_clusters L2-normalised centroids for the L2-normalised points.
    Computes using the device where 'points' is stored.

    points: float(nb_points, point_size) >= 0, will be normalised internally
    output: float(nb_clusters, point_size)
    """
    assert nb_clusters > 0
    device: torch.device = points.device
    nb_points, point_size = points.size()
    assert nb_clusters <= nb_points
    _l2_normalise_rows(points)

    def kmeanspp_init_clusters():
        """
        Returns an initial clustering using kmeans++ algorithm.
        output: float(nb_clusters, point_size), L2-normalised
        """
        nb_candidates: int = 2 + int(math.log(nb_clusters))
        centroids = torch.empty(nb_clusters, point_size, dtype = points.dtype, device = device)

        def half_squared_distances_to_points(x: torch.Tensor) -> torch.Tensor:
            """
            x: float(n, point_size), L2-normalised
            output: float(n, nb_points) = [0.5 * dist(x[i], points[j])^2]_{i,j}
            """
            # d(x,y)^2 = sum (x_i - y_i)^2 = sum x_i^2 + sum y_i^2 -2 sum x_i y_i = 2 - 2 sum x_i y_i (due to normalisation)
            return (1 - x.mm(points.t())).clamp(min = 0.) # Clamp to fix -epsilon values due to float imprecision ?
        
        # First point is random. Next points are selected randomly to minimize distance points - clusters centroids.
        centroids[0] = points[torch.randint(nb_points, size = ())]
        centroids_to_points_distances: torch.Tensor = half_squared_distances_to_points(centroids[[0]]).view(nb_points)
        objective = centroids_to_points_distances.sum() # tensor float()

        for cluster_i in range(1, nb_clusters):
            # Randomly select nb_candidates points, weighted by their distance to current clusters
            candidates = points[
                torch.multinomial(
                    input = centroids_to_points_distances,
                    num_samples = nb_candidates,
                    replacement = True
                ) # int(nb_candidates) indexes of [0, nb_points[
            ] # float(nb_candidates, point_size)
            candidates_to_points_distances = half_squared_distances_to_points(candidates) # float(nb_candidates, nb_points)
            # Keep candidate that best improves the points - cluster distances
            best = (float("inf"), None, None)
            for candidate_id in range(nb_candidates):
                centroids_to_points_distance_with_candidate = torch.min(
                    centroids_to_points_distances,
                    candidates_to_points_distances[candidate_id]
                )
                objective_with_candidate = centroids_to_points_distance_with_candidate.sum()
                if objective_with_candidate < best[0]:
                    best = (
                        objective_with_candidate,
                        candidates[candidate_id],
                        centroids_to_points_distance_with_candidate
                    )
            assert best[1] is not None
            (objective, centroids[cluster_i], centroids_to_points_distances) = best
        return centroids

    centroids = kmeanspp_init_clusters() # float(nb_clusters, point_size), L2-normalised
    assert centroids.size() == (nb_clusters, point_size)

    # Spherical Kmeans, see https://doi.org/10.1023%2Fa%3A1007612920971
    objective = torch.zeros((), dtype = torch.float, device = device) # Should increase
    for iter_i in range(max_iter):
        # Use cosine similarity as distance: sim(x,y) = sum x_i y_i, higher is "closer".
        # Link to L2 euclidian norm with L2-normalised vectors:
        # d(x,y)^2 = sum (x_i - y_i)^2 = sum x_i^2 + sum y_i^2 -2 sum x_i y_i = 2 - 2 sum x_i y_i (due to normalisation)
        cosine_similarities = points.mm(centroids.t()) # float(nb_points, nb_clusters) = sim(points[i], centroids[j])_{i,j}
        (
            max_similarity_to_centroids, # float(nb_points) = max_j sim(points[i], centroids[j])
            closest_centroid # int(nb_points) = argmax_j sim(points[i], centroids[j])
        ) = torch.max(cosine_similarities, dim = -1)
        # Iteration stop
        new_objective = max_similarity_to_centroids.mean()
        if torch.allclose(objective, new_objective):
            break
        objective = new_objective
        # Select new centroids points
        for cluster_i in range(nb_clusters):
            is_point_i_closest_to_centroid = (closest_centroid == cluster_i) # bool(nb_points)
            if is_point_i_closest_to_centroid.any():
                centroids[cluster_i] = points[is_point_i_closest_to_centroid].mean(dim = 0)
            else:
                farthest_point_from_all_centroids = max_similarity_to_centroids.argmin()
                centroids[cluster_i] = points[farthest_point_from_all_centroids]
                # Update similarities to prevent this point from being selected again
                max_similarity_to_centroids[farthest_point_from_all_centroids] = 1.
        _l2_normalise_rows(centroids)
    return centroids

###############################################################################

class Tests(unittest.TestCase):
    def assert_torch_allclose(self, lhs, rhs, msg = None, **kwds):
        if not torch.allclose(lhs, rhs, **kwds):
            if msg is None:
                msg = "\n{}\n\t!=\n{}".format(lhs, rhs)
            raise self.failureException(msg)

    def test_kmeans(self):
        def row_sorted(t: torch.Tensor):
            # Assume values are near [0, 1].
            # Build a single key for each row: [a, b, c] -> [100*a + 10*b + c]
            # Use key to do a lexicographic sort (increasing order) of rows
            _, width = t.size()
            factors = torch.pow(
                10.,
                torch.arange(start = width - 1, end = -1, step = -1, dtype = torch.float)
            ) # [100, 10, 1]
            keys = t.mv(factors) # keys(nb_rows) ; each row = [100*a + 10*b + c]
            permutation = keys.argsort(dim = 0)
            return t[permutation]
        # Kmeans on normalised points with nb_points clusters should return the points
        points = torch.rand(8, 2) + EPSILON
        _l2_normalise_rows(points)
        centroids = spherical_kmeans(points = points, nb_clusters = points.size(0), max_iter = 10)
        self.assert_torch_allclose(row_sorted(points), row_sorted(centroids))
        # Kmeans on duplicates should group them together (on R+^2 unit sphere)
        points = torch.tensor([
            [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0],
        ])
        centroids = spherical_kmeans(points = points, nb_clusters = 2, max_iter = 10)
        self.assert_torch_allclose(
            row_sorted(centroids),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        )
        # Test with 3 point groups on corners of the unit sphere restricted to R+^3
        points = torch.tensor([
            [0.0, 0.0, 1.0], [0.0, 0.1, 0.9], [0.1, 0.0, 0.9],
            [0.0, 1.0, 0.0], [0.0, 0.9, 0.1], [0.1, 0.9, 0.0],
            [1.0, 0.0, 0.0], [0.9, 0.0, 0.1], [0.9, 0.1, 0.0]
        ])
        centroids = spherical_kmeans(points = points, nb_clusters = 3, max_iter = 10)
        self.assert_torch_allclose(
            row_sorted(centroids),
            torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            atol = 0.2 # Large tolerance as these are not the real centroids, just approximations
        )