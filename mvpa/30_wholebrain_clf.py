import os
import sys
import numpy as np
import neuroboros as nb
from joblib import Parallel, delayed, cpu_count, parallel_backend


def wholebrain_classification_less_data(
        dms1, dms2, test_idx, seed=0, npcs=np.arange(10, 1010, 10)):
    rng = np.random.default_rng(seed)
    Y = dms2[test_idx]
    ns = len(dms2)
    choices = np.delete(np.arange(ns), test_idx)
    U, s, Vt = np.linalg.svd(dms1[test_idx], full_matrices=False)
    U = (Y @ Vt.T) / s[np.newaxis, :]
    arng = np.arange(U.shape[0])
    aaa = []
    for n in range(1, ns):
        train_idx = rng.choice(choices, n, replace=False)
        Yhat = np.mean([dms2[_] for _ in train_idx], axis=0)
        Uhat = (Yhat @ Vt.T) / s[np.newaxis, :]
        aa = [nb.classification(U, Uhat, npc=npc) for npc in npcs]
        # aa = []
        # for npc in npcs:
        #     d = cdist(U[:, :npc], Uhat[:, :npc], metric='correlation')
        #     a = np.mean(np.argmin(d, axis=1) == arng)
        #     aa.append(a)
        aaa.append(np.array(aa))
    aaa = np.array(aaa)
    return aaa


def load_data(dset, runs, space, align, align_radius):
    dms_lr = []
    for lr in 'lr':
        dms = []
        for sid in dset.subjects:
            dm = dset.get_data(sid, dset.name, runs, lr, space=space)
            if align != 'surf':
                xfm = nb.load(f'ha_xfm/{dset_name}/{space}_1st-half/{sid}_{lr}h_{align}_to-tmpl_{align_radius}mm.npz')
                dm = dm @ xfm
            dms.append(dm)
        dms = np.stack(dms, axis=0)
        dms_lr.append(dms)
    dms_lr = np.concatenate(dms_lr, axis=2)
    return dms_lr


if __name__ == '__main__':
    dsets = ['budapest', 'raiders']
    spaces = [f'{a}-ico{b}' for a in ['fsavg', 'fslr', 'onavg'] for b in [64]]
    train_dur, test_dur = '1st-half', '2nd-half'
    radius = 20
    align_radius = 20
    arg = int(sys.argv[1])
    slc = slice(None, None, -1)

    for align in ['procr', 'ridge', 'surf'][slc]:
        for dset_name in dsets[slc]:
            dset = nb.dataset(dset_name)
            sids = dset.subjects
            train_runs = {'budapest': [1, 2, 3], 'raiders': [1, 2]}[dset_name]
            test_runs = {'budapest': [4, 5], 'raiders': [3, 4]}[dset_name]
            for space in spaces[slc]:
                center_space = space.split('-')[0] + '-ico32'
                dms_train = load_data(dset, train_runs, space, align, align_radius)
                dms_test = load_data(dset, test_runs, space, align, align_radius)
                print(dms_train.shape, dms_test.shape)

                jobs = []
                for seed in range(arg * 10, (arg + 1) * 10):
                    out_dir = f'wholebrain_clf/{dset_name}/{space}_{align}_{align_radius}mm_{train_dur}_{test_dur}_seed{seed:04d}'
                    for test_idx, sid in enumerate(sids):
                        out_fn = f'{out_dir}/{sid}.npy'
                        job = delayed(nb.record(out_fn, wholebrain_classification_less_data))(
                            dms_train, dms_test, test_idx, seed=seed, npcs=np.arange(10, 1010, 10))
                        jobs.append(job)

                with parallel_backend('loky', inner_max_num_threads=1):
                    with Parallel(n_jobs=cpu_count() // 2, verbose=1) as parallel:
                        parallel(jobs)
