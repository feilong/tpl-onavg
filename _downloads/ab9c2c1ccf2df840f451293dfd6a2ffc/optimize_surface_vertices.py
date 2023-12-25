import os
import sys
import pickle
import numpy as np
import nibabel as nib
import scipy.sparse as sparse
from scipy.spatial import cKDTree, ConvexHull
from datetime import datetime
from joblib import Parallel, delayed

from surface import dijkstra_algorithm

from config import prev_name, tmpl_name


def compute_vertex_loss(v_idx, nc, neighbors, neighbor_distances, max_dist, power):
    nbrs, dists = dijkstra_algorithm(v_idx, nc, neighbors, neighbor_distances, max_dist)
    loss = np.reciprocal(dists) ** power
    # loss[nbrs == v_idx] = self_value
    # loss = loss**power
    return nbrs, loss


class Optimizer(object):
    variables = [
            'rng', 'step_count', 'seed', 'power',
            'losses', 'v_indices', 'nv', 'nc',
            'loss', 'init_loss', 'prev_loss', 'has_changed', 'loss_history',
    ]

    def __init__(self, nv, neighbors, neighbor_distances, seed=0, power=4):
        self.nv = nv
        self.nc = len(neighbors)
        self.neighbors = neighbors
        self.neighbor_distances = neighbor_distances
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.power = power

        self.loss_history = []
        self.step_count = 0

    def init(self):
        self.losses = np.zeros((self.nc, ))
        self.v_indices = self.rng.choice(self.nc, (self.nv, ), replace=False)
        loss = []
        for v_idx in self.v_indices:
            loss.append(self.losses[v_idx])

            nbrs, nbr_loss = compute_vertex_loss(
                v_idx, self.nc, self.neighbors, self.neighbor_distances, max_dist=2.0, power=self.power)
            self.losses[nbrs] += nbr_loss

        self.loss = np.sum(loss)
        self.init_loss = self.loss
        self.has_changed = np.ones((self.nc, ), dtype=bool)
        self.loss_history.append(self.loss)
        print(datetime.now(), self.init_loss)

    def step(self):
        has_changed = np.zeros((self.nc, ), dtype=bool)
        loss_delta = []
        indices = np.arange(self.nv)
        self.rng.shuffle(indices)
        for idx in indices:
            v_idx = self.v_indices[idx]
            if not self.has_changed[v_idx]:
                continue

            nbrs, nbr_loss = compute_vertex_loss(
                v_idx, self.nc, self.neighbors, self.neighbor_distances, max_dist=2.0, power=self.power)
            self.losses[nbrs] -= nbr_loss

            new_v_idx = np.argmin(self.losses)
            if new_v_idx != v_idx:
                has_changed[nbrs] = True
                new_nbrs, new_nbr_loss = compute_vertex_loss(
                    new_v_idx, self.nc, self.neighbors, self.neighbor_distances, max_dist=2.0, power=self.power)
                has_changed[new_nbrs] = True
                loss_delta.append(self.losses[v_idx] - self.losses[new_v_idx])
                self.losses[new_nbrs] += new_nbr_loss
            else:
                self.losses[nbrs] += nbr_loss
        loss_delta = np.sum(loss_delta)
        self.prev_loss = self.loss
        self.loss -= loss_delta
        self.loss_history.append(self.loss)
        self.has_changed = has_changed
        self.step_count += 1
        print(datetime.now(), f'{self.step_count:4d}, {self.loss / self.init_loss}')

    def run(self, max_steps=20, tmp_fn=None):
        while self.step_count < max_steps:
            if not np.any(self.has_changed):
                break
            self.step()
            if tmp_fn is not None:
                self.to_pickle(tmp_fn)

    def to_pickle(self, fn):
        output = {attr: getattr(self, attr) for attr in self.variables}
        with open(fn, 'wb') as f:
            pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, nv, neighbors, neighbor_distances, fn):
        with open(fn, 'rb') as f:
            d = pickle.load(f)
        opt = cls(nv=nv, neighbors=neighbors, neighbor_distances=neighbor_distances)
        for attr in cls.variables:
            setattr(opt, attr, d[attr])
        return opt


if __name__ == '__main__':
    lr = sys.argv[1]
    seed = int(sys.argv[2])
    n_div = 8
    power = 4
    nv = 4**7 * 10 + 2

    npz = np.load(f'../lab/cache/{prev_name}_upsampled{n_div}x_sphere_neighbors.npz')
    neighbors = np.array_split(npz['concatenated'], npz['boundaries'])
    neighbor_distances = np.load(f'../lab/cache/{prev_name}_upsampled{n_div}x_neighbor-distances-gmean_{lr}h.npy')
    neighbor_distances = np.array_split(neighbor_distances, npz['boundaries'])

    tmp_fn = f'../lab/cache/{tmpl_name}_{lr}h_pickle/seed{seed}_tmp.pickle'
    os.makedirs(os.path.dirname(tmp_fn), exist_ok=True)
    if os.path.exists(tmp_fn):
        opt = Optimizer.from_pickle(nv, neighbors, neighbor_distances, tmp_fn)
    else:
        opt = Optimizer(nv, neighbors, neighbor_distances, seed=seed, power=power)
        opt.init()
    opt.run(max_steps=100, tmp_fn=tmp_fn)

    pkl_fn = f'../lab/cache/{tmpl_name}_{lr}h_pickle/seed{seed}.pickle'
    opt.to_pickle(pkl_fn)
