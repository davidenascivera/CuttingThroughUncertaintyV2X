import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
import logging

logger = logging.getLogger(__name__)

class DataAssociationManager:
    def __init__( self, id_sens_1: int, id_sens_2: int, measurements: dict[int, dict[str, any]], sme_dict: dict[int, any],
                 blacklist: set[frozenset], mature_tracks_id: list, dt: float) -> None:
        self.sme_dict = sme_dict
        self.measurements = measurements
        # Filter only mature tracks for each sensor
        mature = {k: v for k, v in sme_dict.items() if v.mature_track}
        self.track_sens_1 = {k: v for k, v in mature.items() if v.sensor_id == id_sens_1}
        self.track_sens_2 = {k: v for k, v in mature.items() if v.sensor_id == id_sens_2}
        self.n_1 = len(self.track_sens_1)
        self.n_2 = len(self.track_sens_2)
        # preserve incoming blacklist of frozensets
        self.blacklist = set(tuple(x) if isinstance(x, np.ndarray) else x for x in blacklist)
        # initialize cost matrix placeholder
        self.cost_matrix = np.zeros((self.n_1, self.n_2), dtype=float)
        self.valid_assignments = []

    def compute_cost_matrix(self, verbose: bool = False, verbose_prepad: bool = False) -> np.ndarray:
        """
        Builds and pads the cost matrix. Applies infinite cost for zero IoU and strong penalties for blacklisted pairs.
        """
        id_row = [v.id + v.sensor_id for v in self.track_sens_1.values()]
        id_col = [v.id + v.sensor_id for v in self.track_sens_2.values()]

        costs = np.zeros((self.n_1, self.n_2), dtype=float)
        max_real_cost = 1e6
        # compute pairwise costs
        for i, (k1, t1) in enumerate(self.track_sens_1.items()):
            poly1 = Polygon(t1.Xc)
            center1 = np.array(t1.centroid)
            for j, (k2, t2) in enumerate(self.track_sens_2.items()):
                pair = frozenset([k1, k2])
                poly2 = Polygon(t2.Xc)
                inter = poly1.intersection(poly2).area
                if inter <= 1e-8:
                    # no overlap: infinite cost, add to blacklist
                    costs[i, j] = np.inf
                    self.blacklist.add(pair)
                    continue
                union = poly1.union(poly2).area
                iou = inter / union if union > 0 else 0.0
                center2 = np.array(t2.centroid)
                dist = np.linalg.norm(center1 - center2)
                # cost: distance penalized by IoU
                cost = - iou
                # enforce blacklist penalty
                if pair in self.blacklist:
                    cost += max_real_cost * 5 + 1.0
                costs[i, j] = cost
                # print(f"Cost for {k1} ↔ {k2}: {cost:.3f} (IoU: {iou:.3f}, dist: {dist:.3f})")
                max_real_cost = max(max_real_cost, cost)

        # pad to square with dummy cost above any real cost
        dummy_cost = max_real_cost * 1.2 + 1.0
        padded = self.cost_matrix_padding(costs, dummy_cost=dummy_cost, verbose=verbose_prepad)
        self.cost_matrix = padded
        if verbose:
            rows = id_row + [f"dummy_row_{d}" for d in range(padded.shape[0] - len(id_row))]
            cols = id_col + [f"dummy_col_{d}" for d in range(padded.shape[1] - len(id_col))]
            mat = pd.DataFrame(padded, index=rows, columns=cols)
            logger.info("Cost matrix after padding:\n%s", mat)
        return self.blacklist

    def solve(self, verbose: bool = False) -> list[tuple[int, int]]:
        """
        Always recompute cost matrix and solve with Hungarian, fallback to greedy if infeasible.
        """
        self.compute_cost_matrix(verbose=False)
        try:
            row_ind, col_ind = linear_sum_assignment(self.cost_matrix)
        except ValueError:
            logger.warning("Hungarian failed, falling back to greedy")
            # clear blacklist and retry once
            self.blacklist.clear()
            self.compute_cost_matrix(verbose=False)
            return self.solve_greedy(verbose=verbose)

        valid = []
        keys1 = list(self.track_sens_1.keys())
        keys2 = list(self.track_sens_2.keys())
        for r, c in zip(row_ind, col_ind):
            if r < self.n_1 and c < self.n_2 and not np.isinf(self.cost_matrix[r, c]):
                valid.append((keys1[r], keys2[c]))
        self.valid_assignments = valid
        if verbose:
            logger.info("Valid assignments: %s", valid)
        return valid

    def solve_greedy(self, verbose: bool = False) -> list[tuple[int, int]]:
        """
        Greedy nearest-match fallback that respects finite costs only.
        """
        costs = self.cost_matrix.copy()
        keys1 = list(self.track_sens_1.keys())
        keys2 = list(self.track_sens_2.keys())
        triples = [(i, j, costs[i, j]) for i in range(costs.shape[0]) for j in range(costs.shape[1])]
        triples.sort(key=lambda x: x[2])
        assigned_r, assigned_c = set(), set()
        valid = []
        for i, j, cost in triples:
            if i < self.n_1 and j < self.n_2 and np.isfinite(cost):
                if i not in assigned_r and j not in assigned_c:
                    valid.append((keys1[i], keys2[j]))
                    assigned_r.add(i)
                    assigned_c.add(j)
                    if verbose:
                        logger.info("Greedy fallback: %s ↔ %s | cost %.3f", keys1[i], keys2[j], cost)
            if len(assigned_r) >= self.n_1 or len(assigned_c) >= self.n_2:
                break
        self.valid_assignments = valid
        return valid

    

    @staticmethod
    def cost_matrix_padding(cost_mat: np.ndarray, dummy_cost: float = 0.0, verbose: bool = False) -> np.ndarray:
        """
        Pad cost matrix to square and handle all-inf rows/columns.
        """
        n_r, n_c = cost_mat.shape
        mat = cost_mat.copy()
        if n_r > n_c:
            mat = np.hstack((mat, np.full((n_r, n_r - n_c), dummy_cost)))
        elif n_c > n_r:
            mat = np.vstack((mat, np.full((n_c - n_r, n_c), dummy_cost)))
        # if any full-inf row or column, add extra dummy row+col
        if np.any(np.all(np.isinf(mat), axis=1)) or np.any(np.all(np.isinf(mat), axis=0)):
            mat = np.vstack((mat, np.full((1, mat.shape[1]), dummy_cost)))
            mat = np.hstack((mat, np.full((mat.shape[0], 1), dummy_cost)))
        return mat
