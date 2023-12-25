import os
import sys
import pickle
import numpy as np
import nibabel as nib
import scipy.sparse as sparse
from scipy.spatial import cKDTree, ConvexHull
from datetime import datetime
from joblib import Parallel, delayed

from surface import Surface, barycentric_resample
from surface.subdivision import surface_subdivision
from surface.optimization import compute_opposite_triangles, compute_opposite_triangles_parallel

from config import prev_name, tmpl_name


def compute_face_neighbors_chunk(mat, faces16, chunk, max_dist=4):
    neighbors = []
    inv_max = 1 / max_dist
    for f in chunk:
        inv = mat[faces16[f]].toarray().max(axis=0)
        vertices = np.where(inv > inv_max)[0]
        nbrs = np.where(np.any(np.isin(faces16, vertices), axis=1))[0]
        neighbors.append(nbrs[nbrs != f])
    return neighbors


def compute_face_neighbors(mat, faces16, max_dist=4, n_jobs=40, cache_fn=None):
    if cache_fn is not None and os.path.exists(cache_fn):
        npz = np.load(cache_fn)
        neighbors = np.array_split(npz['concatenated'], npz['boundaries'])

    chunks = np.array_split(np.arange(faces16.shape[0]), n_jobs)
    jobs = [delayed(compute_face_neighbors_chunk)(mat, faces16, chunk, max_dist=max_dist) for chunk in chunks]
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(jobs)
    neighbors = []
    for res in results:
        neighbors += res

    if cache_fn is not None:
        np.savez(
            cache_fn,
            concatenated=np.concatenate(neighbors),
            boundaries=np.cumsum([len(_) for _ in neighbors][:-1]))

    return neighbors


class Optimizer(object):
    variables = [
            'rng', 'step_count', 'seed', 'opt_n_div', 'power',
            'min_losses', 'losses', 't_indices', 't_w_indices',
            'loss', 'init_loss', 'prev_loss', 'has_changed', 'loss_history',
    ]

    def __init__(self, nv, mat, faces16, neighbors, seed=0, opt_n_div=8, power=4):
        self.nv = nv
        self.mat = mat
        self.faces16 = faces16
        self.nf = faces16.shape[0]
        self.neighbors = neighbors
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.opt_n_div = opt_n_div
        self.weight_combinations = np.array([
            (i, j, self.opt_n_div - i - j) for i in range(self.opt_n_div + 1) for j in range(self.opt_n_div + 1 - i)])
        self.power = power
        self.loss_history = []
        self.step_count = 0

    def init(self):
        self.min_losses = np.zeros((self.nf, ))
        self.losses = np.zeros((self.nf, self.weight_combinations.shape[0]))
        self.t_indices = np.zeros((self.nv, ), dtype=int)
        self.t_w_indices = np.zeros((self.nv, ), dtype=int)

        count = 0
        loss = []
        while count < self.nv:
            min_ = self.min_losses.min()
            f = self.rng.choice(np.where(self.min_losses == min_)[0])
            if min_ == 0:
                w_idx = self.rng.choice(self.weight_combinations.shape[0])
            else:
                w_idx = np.argmin(self.losses[f])

            loss.append(self.losses[f, w_idx])
            self.add_vertex(count, f, w_idx, is_add=True)
            count += 1
            if count % 1000 == 0:
                print(datetime.now(), count)
        self.loss = np.sum(loss)
        self.init_loss = self.loss
        self.has_changed = np.ones((self.nf, ), dtype=bool)
        self.loss_history.append(self.loss)

    def step(self):
        has_changed = np.zeros((self.nf, ), dtype=bool)
        loss_delta = []
        indices = np.arange(self.nv)
        self.rng.shuffle(indices)
        for idx in indices:
            if not self.has_changed[self.t_indices[idx]]:
                continue
            self.add_vertex(idx, self.t_indices[idx], self.t_w_indices[idx], is_add=False)
            f = np.argmin(self.min_losses)
            w_idx = np.argmin(self.losses[f])
            if self.t_indices[idx] != f or self.t_w_indices[idx] != w_idx:
                has_changed[self.neighbors[f]] = True
                has_changed[self.neighbors[self.t_indices[idx]]] = True
                loss_delta.append(self.losses[self.t_indices[idx], self.t_w_indices[idx]] - self.losses[f, w_idx])
            self.add_vertex(idx, f, w_idx, is_add=True)
        loss_delta = np.sum(loss_delta)
        self.prev_loss = self.loss
        self.loss -= loss_delta
        self.loss_history.append(self.loss)
        self.has_changed = has_changed
        self.step_count += 1
        print(datetime.now(), f'{self.step_count:4d}, {self.loss / self.init_loss}')

    def add_vertex(self, idx, f, w_idx, is_add=True):
        if is_add:
            self.t_indices[idx] = f
            self.t_w_indices[idx] = w_idx
        weights = self.weight_combinations[w_idx]

        nbrs = self.neighbors[f]
        indices1 = self.faces16[f]
        indices2 = self.faces16[nbrs].ravel()
        mm = self.mat[np.ix_(indices1, indices2)].toarray()
        mm = (np.reciprocal(mm) * (weights / self.opt_n_div)[:, np.newaxis]).sum(axis=0)
        mm = mm.reshape(-1, 3)

        for nbr, dd in zip(nbrs, mm):
            d = np.sum(dd[np.newaxis] * self.weight_combinations, axis=1) / self.opt_n_div
            loss = np.reciprocal(np.maximum(d, 0.01))**power
            if is_add:
                self.losses[nbr] += loss
            else:
                self.losses[nbr] -= loss
            self.min_losses[nbr] = self.losses[nbr].min()

        X = np.array([[-1., -1.], [1., 0.], [0., 1.]])
        x2, y2, z2 = np.reciprocal(mat[[indices1[0], indices1[0], indices1[1]], [indices1[1], indices1[2], indices1[2]]].A.ravel())**2
        diff = weights[np.newaxis] - self.weight_combinations
        b1, b2 = np.linalg.lstsq(X, diff.T, rcond=None)[0]
        sq = b1**2 * x2 + b2**2 * y2 + b1 * b2 * (x2 + y2 - z2)
        sq[sq < 0] = 0
        d = np.sqrt(sq) / self.opt_n_div
        loss = np.reciprocal(np.maximum(d, 0.01))**power
        if is_add:
            self.losses[f] += loss
        else:
            self.losses[f] -= loss
        self.min_losses[f] = self.losses[f].min()

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
    def from_pickle(cls, nv, mat, faces16, neighbors, fn):
        with open(fn, 'rb') as f:
            d = pickle.load(f)
        opt = cls(nv=nv, mat=mat, faces16=faces16, neighbors=neighbors)
        for attr in cls.variables:
            setattr(opt, attr, d[attr])
        return opt


if __name__ == '__main__':
    lr = sys.argv[1]
    seed = int(sys.argv[2])
    opt_n_div = 8
    power = 4

    mat = sparse.load_npz(f'../lab/templates/{tmpl_name}_midthickness_{lr}h_inv-dijkstra-gmean.npz')
    n = mat.shape[0]
    mat[np.arange(n), np.arange(n)] = 100

    if prev_name == 'fsaverage':
        darrays = nib.load('surface_utils/fsaverage_sphere.surf.gii').darrays
        sphere = {'vertices': darrays[0].data.astype(np.float64), 'faces': darrays[1].data.copy()}
        sphere['vertices'] /= np.linalg.norm(sphere['vertices'], axis=1, keepdims=True)
    else:
        sphere = np.load(f'../lab/templates/{prev_name}_midthickness_{lr}h_parallel-nbrs-opt-sphere.npz')

    coords16, faces16 = sphere['vertices'].copy(), sphere['faces'].copy()

    cache_fn = f'../lab/cache/{prev_name}_{lr}h_face-neighbors.npz'
    neighbors = compute_face_neighbors(mat, faces16, max_dist=4, n_jobs=40, cache_fn=cache_fn)

    tmp_fn = f'../lab/cache/{tmpl_name}_{lr}h_seed{seed}_tmp.pickle'
    if os.path.exists(tmp_fn):
        opt = Optimizer.from_pickle(coords16.shape[0], mat, faces16, neighbors, tmp_fn)
    else:
        opt = Optimizer(coords16.shape[0], mat, faces16, neighbors, seed=seed, opt_n_div=opt_n_div, power=power)
        opt.init()
    opt.run(max_steps=40, tmp_fn=tmp_fn)

    pkl_fn = f'../lab/cache/{tmpl_name}_{lr}h_seed{seed}.pickle'
    opt.to_pickle(pkl_fn)
