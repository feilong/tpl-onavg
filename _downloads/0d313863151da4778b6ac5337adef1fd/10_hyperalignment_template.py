import os
import numpy as np
from joblib import Parallel, delayed, parallel_backend, cpu_count

import neuroboros as nb
from hyperalignment.searchlight import searchlight_template


def compute_functional_template(dset_name, space, lr, dur='1st-half', radius=20):
    out_fn = f'ha_tpl/{dset_name}/{space}_{lr}h_{dur}_{radius}mm.npy'
    dset = nb.dataset(dset_name)
    sids = dset.subjects
    runs = {'budapest': [1, 2, 3], 'raiders': [1, 2]}[dset_name]

    dms = np.stack(
        [dset.get_data(sid, dset_name, runs, lr, space=space) for sid in sids],
        axis=0)
    center_space = space.split('-')[0] + '-ico32'
    sls, dists = nb.sls(lr, radius, space=space, center_space=center_space, return_dists=True)

    nb.record(out_fn, searchlight_template)(
        dms, sls, dists, radius, n_jobs=1)


if __name__ == '__main__':
    dsets = ['budapest', 'raiders']
    spaces = [f'{a}-ico{b}' for a in ['fsavg', 'fslr', 'onavg'] for b in [32]]

    jobs = []
    for dset_name in dsets:
        for space in spaces:
            for lr in 'lr':
                job = delayed(compute_functional_template)(dset_name, space, lr)
                jobs.append(job)
    print(len(jobs), cpu_count())

    with parallel_backend('loky', inner_max_num_threads=1):
        with Parallel(n_jobs=-2, verbose=1) as parallel:
            parallel(jobs)
