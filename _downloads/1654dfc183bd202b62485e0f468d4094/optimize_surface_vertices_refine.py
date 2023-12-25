import os
import sys
import pickle
import heapq
from collections import OrderedDict
import numpy as np
import nibabel as nib
import scipy.sparse as sparse
from scipy.spatial import cKDTree, ConvexHull
from datetime import datetime
from joblib import Parallel, delayed

from surface import dijkstra_algorithm
from surface import Surface, barycentric_resample
from surface.coordinates import compute_mapping_types, compute_spherical_from_cartesian, compute_cartesian_from_spherical

from config import prev_name, tmpl_name
from optimize_surface_vertices_parallel import dijkstra_algorithm_concatenated


def compute_loss(f_indices1, f_weights1, f_indices2, f_weights2, faces, cache, power):
    n1, n2 = f_indices1.shape[0], f_indices2.shape[0]
    mm = np.zeros((n1, n2))
    for j, f_idx2 in enumerate(f_indices2):
        for jj, v2 in enumerate(faces[f_idx2]):
            nbrs_, nbr_ds_ = cache[v2]
            for i, f_idx1 in enumerate(f_indices1):
                for ii, v1 in enumerate(faces[f_idx1]):
                    idx_ = np.where(nbrs_ == v1)[0]
                    if len(idx_):
                        mm[i, j] += f_weights1[i, ii] * f_weights2[j, jj] * nbr_ds_[idx_[0]]
                    else:
                        mm[i, j] += np.inf
    loss = np.reciprocal(mm) ** power
    return loss


def optimize_coordinates(idx, f_indices, f_weights, cache, mapping, spherical, maps, sphere, power, eps=2**-23, step_sizes=2**np.linspace(-21, -10, 12)):
    f, w = sphere.faces[f_indices[idx]], f_weights[idx]
    nbrs = np.unique(np.concatenate([cache[v][0] for v in f]))
    nbrs = np.unique([mapping[_] for _ in nbrs if _ in mapping])
    nbrs = nbrs[nbrs != idx]
    f_indices2, f_weights2 = f_indices[nbrs], f_weights[nbrs]

    sphr = np.tile(spherical[[idx]], (5, 1))
    sphr[0, 0] -= eps
    sphr[1, 0] += eps
    sphr[2, 1] -= eps
    sphr[3, 1] += eps
    coords = compute_cartesian_from_spherical(sphr, np.tile(maps[[idx]], (5, 1)))
    f_indices1, f_weights1 = sphere.compute_barycentric_weights(coords)
    loss = compute_loss(f_indices1, f_weights1, f_indices2, f_weights2, sphere.faces, cache, power)

    l = loss.sum(axis=1)
    loss0, coords0 = l[4], coords[4]
    gd = np.array([l[0] - l[1], l[2] - l[3]])
    gd /= np.linalg.norm(gd)

    sphr = np.tile(spherical[[idx]], (len(step_sizes), 1))
    sphr += gd[np.newaxis] * step_sizes[:, np.newaxis]
    coords = compute_cartesian_from_spherical(sphr, np.tile(maps[[idx]], (len(step_sizes), 1)))
    f_indices1, f_weights1 = sphere.compute_barycentric_weights(coords)
    loss = compute_loss(f_indices1, f_weights1, f_indices2, f_weights2, sphere.faces, cache, power)

    l = loss.sum(axis=1)
    min_idx = 1
    while min_idx < 11 and l[min_idx+1] < l[min_idx]:
        min_idx += 1

    if loss0 < l[min_idx - 1]:
        return coords0, loss0, loss0, False
    return coords[min_idx - 1], loss0, l[min_idx - 1], True


def optimize_coordinates_chunk(chunk, f_indices, f_weights, cache, mapping, spherical, maps, sphere, power, eps=2**-23, step_sizes=2**np.linspace(-21, -10, 12)):
    results = []
    for idx in chunk:
        res = optimize_coordinates(idx, f_indices, f_weights, cache, mapping, spherical, maps, sphere, power, eps=eps, step_sizes=step_sizes)
        results.append(res)
    coords = np.stack([_[0] for _ in results], axis=0)
    prev_loss = np.array([_[1] for _ in results])
    new_loss = np.array([_[2] for _ in results])
    has_changed = np.array([_[3] for _ in results])
    return coords, prev_loss, new_loss, has_changed


class RefinementOptimizer(object):
    variables = [
            'coords', 'nv', 'nc', 'power', 'max_dist', 'step_count',
            'prev_loss', 'new_loss', 'has_changed', 'pct_changed', 'history',
    ]

    def __init__(self, sphere, coords, neighbors, neighbor_distances, boundaries, power=4, max_dist=2.0):
        self.sphere = sphere
        self.coords = coords
        self.cache = OrderedDict()
        self.nv = self.coords.shape[0]
        self.nc = self.sphere.coords.shape[0]
        self.neighbors = neighbors
        self.neighbor_distances = neighbor_distances
        self.boundaries = boundaries
        self.power = power
        self.max_dist = max_dist
        self.step_count = 0
        self.history = []

    def update(self):
        self.f_indices, self.weights = self.sphere.compute_barycentric_weights(self.coords)

        self.mapping = {}
        todo = []
        faces = self.sphere.faces[self.f_indices]
        for i, f in enumerate(faces):
            for idx in f:
                if idx not in self.mapping:
                    self.mapping[idx] = []
                self.mapping[idx].append(i)
                if idx in self.cache:
                    self.cache.move_to_end(idx)
                elif idx not in todo:
                    todo.append(idx)
        print(len(todo))

        with Parallel(n_jobs=-1, verbose=1) as parallel:
            results = parallel(
                delayed(dijkstra_algorithm_concatenated)(
                    src, self.nc, self.neighbors, self.neighbor_distances, self.boundaries, max_dist=self.max_dist)
                for src in todo)
        print(len(self.cache))
        for src, res in zip(todo, results):
            self.cache[src] = res
        print(len(self.cache))

        self.maps = compute_mapping_types(self.coords)
        self.spherical = compute_spherical_from_cartesian(self.coords, self.maps)

    def step(self):
        self.update()

        chunks = np.array_split(np.arange(self.nv, dtype=int), 40)
        jobs = [
            delayed(optimize_coordinates_chunk)(
                chunk, self.f_indices, self.weights, self.cache, self.mapping, self.spherical, self.maps, self.sphere, self.power)
            for chunk in chunks]
        with Parallel(n_jobs=-1) as parallel:
            results = parallel(jobs)

        self.coords = np.concatenate([_[0] for _ in results], axis=0)
        self.prev_loss = np.concatenate([_[1] for _ in results], axis=0).sum()
        self.new_loss = np.concatenate([_[2] for _ in results], axis=0).sum()
        self.has_changed = np.concatenate([_[3] for _ in results], axis=0)
        self.pct_changed = np.mean(self.has_changed)
        self.has_changed = np.any(self.has_changed)
        self.step_count += 1

        self.history.append(np.array([self.prev_loss, self.new_loss, self.pct_changed]))

        print(datetime.now(), f'{self.step_count:4d}, {self.prev_loss}, {self.new_loss}, {self.pct_changed}')

    def run(self, max_steps=100, tmp_fn=None):
        # if tmp_fn is not None:
        #     self.to_pickle(tmp_fn)
        while self.step_count < max_steps:
            self.step()
            if tmp_fn is not None:
                self.to_pickle(tmp_fn)
            if not self.has_changed:
                break

    def to_pickle(self, fn):
        output = {attr: getattr(self, attr) for attr in self.variables}
        with open(fn, 'wb') as f:
            pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, sphere, coords, neighbors, neighbor_distances, boundaries, fn):
        with open(fn, 'rb') as f:
            d = pickle.load(f)
        opt = cls(sphere, coords, neighbors, neighbor_distances, boundaries)
        for attr in cls.variables:
            if attr in d:
                setattr(opt, attr, d[attr])
            else:
                print(f'{attr} not in pickle file')
        return opt



if __name__ == '__main__':
    # lr = 'r'
    # seed = 0
    lr = sys.argv[1]
    seed = int(sys.argv[2])

    # n_div = 8
    # n_div, ico, max_dist = 2, 32, 8.0
    n_div, ico, max_dist = 4, 64, 4.0

    npz = np.load(f'../lab/cache/fsaverage_upsampled{n_div}x_sphere.npz')
    sphere = Surface(npz['coords'], npz['faces'], is_sphere=True)
    sphere.compute_vecs_for_barycentric()

    with open(f'../lab/cache/on-avg-1031-init_{lr}h_pickle_parallel_ico{ico}/seed{seed:04d}.pickle', 'rb') as f:
        d = pickle.load(f)
    coords = npz['coords'][d['v_indices']]

    npz = np.load(f'../lab/cache/{prev_name}_upsampled{n_div}x_sphere_neighbors.npz')
    neighbors, boundaries = npz['concatenated'], npz['boundaries']
    boundaries = np.concatenate([[0], boundaries, [neighbors.shape[0]]], axis=0)
    neighbor_distances = np.load(f'../lab/cache/{prev_name}_upsampled{n_div}x_neighbor-distances-gmean_{lr}h.npy')

    tmp_fn = f'../lab/cache/{tmpl_name}_{lr}h_pickle_parallel_refine_ico{ico}/seed{seed:04d}_tmp.pickle'
    os.makedirs(os.path.dirname(tmp_fn), exist_ok=True)
    if os.path.exists(tmp_fn):
        opt = RefinementOptimizer.from_pickle(sphere, coords, neighbors, neighbor_distances, boundaries, tmp_fn)
    else:
        opt = RefinementOptimizer(sphere, coords, neighbors, neighbor_distances, boundaries, max_dist=max_dist)
    opt.run(max_steps=100, tmp_fn=tmp_fn)

    pkl_fn = f'../lab/cache/{tmpl_name}_{lr}h_pickle_parallel_refine_ico{ico}/seed{seed:04d}.pickle'
    opt.to_pickle(pkl_fn)
