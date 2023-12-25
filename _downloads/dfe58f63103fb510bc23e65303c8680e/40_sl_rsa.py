import os
import sys
import numpy as np
import neuroboros as nb
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from joblib import Parallel, delayed, cpu_count, parallel_backend
from datetime import datetime
from functools import partial

# from nia.searchlights import get_searchlights
# from nia.stats import cronbach_alpha, noise_ceilings
# from nia.utils import load, monitor
# from nidatasets import Forrest

# from utils import load_data


def compute_rdms(dms, sl, metric='correlation'):
    sub_dms = dms[:, :, sl]
    rdms = []
    for ds in sub_dms:
        rdm = pdist(ds, metric)
        rdms.append(rdm)
    return rdms


def compute_searchlight_rsa(dms, sls, sl_chunk, func, out_pattern):
    for sl_idx in sl_chunk:
        out_fn = out_pattern.format(sl_idx=sl_idx)
        if os.path.exists(out_fn):
            continue
        timing_fn = out_fn + '.timing'

        sl = sls[sl_idx]
        rdms = nb.utils.monitor(func, timing_fn)(dms, sl)
        rdms = np.stack(rdms, axis=0)
        alpha = nb.stats.cronbach_alpha(rdms, 0, 1)
        ceilings = nb.stats.noise_ceilings(rdms, 0, 1, return_alpha=True)
        rdm_sum = np.sum(rdms, axis=0)
        r = []
        for rdm in rdms:
            pred = rdm_sum - rdm
            r.append(np.mean(zscore(pred) * zscore(rdm)))
        r = np.array(r)
        # z = np.arctanh(r)
        # r = 1 - pdist(rdms, 'correlation')
        # z = np.arctanh(r)
        np.savez(out_fn, alpha=alpha, ceilings=ceilings, r=r)


if __name__ == '__main__':
    # dsets = ['budapest', 'raiders']
    dsets = sys.argv[1:]
    spaces = [f'{a}-ico{b}' for a in ['fsavg', 'fslr', 'onavg'] for b in [64]]
    train_dur, test_dur = '1st-half', '2nd-half'
    radius = 20
    align_radius = 20
    metric = 'correlation'
    func = partial(compute_rdms, metric=metric)

    for align in ['surf', 'procr', 'ridge']:
        for dset_name in dsets:
            dset = nb.dataset(dset_name)
            sids = dset.subjects
            runs = {'budapest': [4, 5], 'raiders': [3, 4]}[dset_name]
            for space in spaces:
                center_space = space.split('-')[0] + '-ico32'
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

                    out_pattern = f'sl_rsa_isc/{dset_name}/{space}_{align}_{align_radius}_{test_dur}/{lr}h_{radius}mm_{metric}/' + '{sl_idx:05d}.npz'
                    os.makedirs(os.path.dirname(out_pattern), exist_ok=True)

                    chunks = np.array_split(np.arange(len(sls)), cpu_count() - 1)

                    jobs = [delayed(compute_searchlight_rsa)(dms, sls, chunk, func, out_pattern) for chunk in chunks]

                    with parallel_backend('loky', inner_max_num_threads=1):
                        with Parallel(n_jobs=-2) as parallel:
                            parallel(jobs)
                    print(datetime.now(), out_pattern)
