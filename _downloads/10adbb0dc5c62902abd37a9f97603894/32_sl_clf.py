import os
import numpy as np
import neuroboros as nb
from joblib import Parallel, delayed, cpu_count, parallel_backend


def searchlight_classification(dm, pred, sls, size):
    accs = []
    for sl in sls:
        a = nb.classification(dm[:, sl], pred[:, sl], size)
        accs.append(a)
    accs = np.array(accs)
    return accs


if __name__ == '__main__':
    dsets = ['budapest', 'raiders']
    spaces = [f'{a}-ico{b}' for a in ['fsavg', 'fslr', 'onavg'] for b in [32]]
    train_dur, test_dur = '1st-half', '2nd-half'
    radius = 20
    align_radius = 20
    size = 5

    for dset_name in dsets:
        dset = nb.dataset(dset_name)
        sids = dset.subjects
        runs = {'budapest': [4, 5], 'raiders': [3, 4]}[dset_name]
        for space in spaces:
            center_space = space.split('-')[0] + '-ico32'
            for align in ['procr', 'ridge', 'surf']:
                for lr in 'lr':
                    sls = nb.sls(lr, radius, space=space, center_space=center_space)
                    dms = []
                    for sid in sids:
                        dm = dset.get_data(sid, dset_name, runs, lr, space=space)
                        if align != 'surf':
                            xfm = nb.load(f'ha_xfm/{dset_name}/{space}_1st-half/{sid}_{lr}h_{align}_to-tmpl_{align_radius}mm.npz')
                            dm = dm @ xfm
                        dms.append(dm)
                    dms = np.stack(dms, axis=0)

                    dm_sum = np.sum(dms, axis=0)
                    ns = dms.shape[0]
                    jobs = []
                    for sid, dm in zip(sids, dms):
                        out_dir = f'sl_clf/{dset_name}/{space}_{align}_{align_radius}mm_{train_dur}_{test_dur}_{size}TRs'
                        out_fn = f'{out_dir}/{sid}_{lr}h_{radius}mm.npy'
                        pred = (dm_sum - dm) / (ns - 1)  # average of others
                        job = delayed(nb.record(out_fn, searchlight_classification))(dm, pred, sls, size)
                        jobs.append(job)

                    with parallel_backend('loky', inner_max_num_threads=1):
                        with Parallel(n_jobs=-2, verbose=1) as parallel:
                            parallel(jobs)
