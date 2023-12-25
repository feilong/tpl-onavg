import os
import numpy as np
import neuroboros as nb
from joblib import Parallel, delayed, parallel_backend, cpu_count

from hyperalignment.sparse import initialize_sparse_matrix
from hyperalignment.searchlight import searchlight_procrustes, searchlight_ridge


if __name__ == '__main__':
    dsets = ['budapest', 'raiders']
    spaces = [f'{a}-ico{b}' for a in ['onavg', 'fsavg', 'fslr'] for b in [32]]
    dur = '1st-half'
    radius = 20

    for space in spaces:
        for lr in 'lr':
            center_space = space.split('-')[0] + '-ico32'
            sls, dists = nb.sls(lr, radius, space=space, center_space=center_space, return_dists=True)
            mat0_fn = f'mat0/{space}_{center_space}_{lr}h_{radius}mm.npz'
            nb.record(mat0_fn, initialize_sparse_matrix)(sls)
            mat0 = nb.load(mat0_fn)

            for dset_name in dsets:
                dset = nb.dataset(dset_name)
                sids = dset.subjects
                runs = {'budapest': [1, 2, 3], 'raiders': [1, 2]}[dset_name]
                tpl = np.load(f'ha_tpl/{dset_name}/{space}_{lr}h_{dur}_{radius}mm.npy')

                jobs = []
                for sid in sids:
                    dm = dset.get_data(sid, dset_name, runs, lr, space=space)
                    for label, ha_func in [('procr', searchlight_procrustes), ('ridge', searchlight_ridge)]:
                        xfm_fn = f'ha_xfm/{dset_name}/{space}_{dur}/{sid}_{lr}h_{label}_to-tmpl_{radius}mm.npz'
                        job = delayed(nb.record(xfm_fn, ha_func))(
                            dm, tpl, sls, dists, radius, mat0)
                        jobs.append(job)
                print(len(jobs), cpu_count())

                with parallel_backend('loky', inner_max_num_threads=1):
                    with Parallel(n_jobs=-2, verbose=1) as parallel:
                        parallel(jobs)
