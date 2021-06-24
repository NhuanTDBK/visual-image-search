from typing import Tuple, List

import annoy


class FlatIndex(object):
    _cls: annoy.AnnoyIndex

    def k_nearest_neighbors(self, X, n_neighbors=50, include_distances=True) -> Tuple[List[int], List[float]]:
        indices, dists = self._cls.get_nns_by_vector(X, n_neighbors, include_distances=include_distances)
        return indices, dists

