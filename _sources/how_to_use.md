# Use the `onavg` template with FreeSurfer

We have prepared the `onavg` template so that it can be easily used with [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/).
These files can be put under FreeSurfer's subjects directory, and they can be used as an reference subject during resampling.
Here is a step-by-step guide on how to use it with FreeSurfer.


## Part 1: Download and organize the template files

### Step 1: Enter FreeSurfer's subjects directory

```{margin}
If FreeSurfer has been properly configured in your system, you can find the directory with `echo $SUBJECTS_DIR` or `echo $FREESURFER_HOME/subjects`.
```

First, let's enter FreeSurfer's subjects directory using command line.
This directory is where the files of the template "subjects" (e.g., `fsaverage`) are located.
The path of the directory can be different in your system.

```bash
cd ~/lab/freesurfer/subjects
```


### Step 2: Download the template files

`````{tab-set}

````{tab-item} Using wget
```bash
wget "https://www.dropbox.com/s/rvg6ui3dqjx7v8i/tpl-onavg_freesurfer.zip"
```
````

````{tab-item} Using curl
```bash
curl -O -L "https://www.dropbox.com/s/rvg6ui3dqjx7v8i/tpl-onavg_freesurfer.zip"
```
````

````{tab-item} Using a web browser
Alternatively, you can open [the URL](https://www.dropbox.com/s/rvg6ui3dqjx7v8i/tpl-onavg_freesurfer.zip) in a web browser, and click the "Download" button to download the zip file and save it to FreeSurfer's subjects directory.
````
`````

### Step 3: Checksum the zip file (optional, recommended)

Let's ensure the downloaded file is intact using the md5 checksum.
The hash 
`````{tab-set}
````{tab-item} Linux
```bash
md5sum tpl-onavg_freesurfer.zip
```
```text
87a17ac12f0634aca1f3ab615edc5a99  tpl-onavg_freesurfer.zip
```
````

````{tab-item} MacOS
```bash
md5 tpl-onavg_freesurfer.zip
```
```text
MD5 (tpl-onavg_freesurfer.zip) = 87a17ac12f0634aca1f3ab615edc5a99
```
````
`````

Unzip the zip file to get the content.
```bash
unzip tpl-onavg_freesurfer.zip
```

(Optional) Remove the zip file to save some disk space.
```bash
rm tpl-onavg_freesurfer.zip
```

After uncompressing the zip file, FreeSurfer's subjects directory should look like this:

```bash
ls
```
```text
bert                fsaverage3  fsaverage_sym  onavg-ico32  rh.EC_average
cvs_avg35           fsaverage4  lh.EC_average  onavg-ico64  sample-001.mgz
cvs_avg35_inMNI152  fsaverage5  onavg-ico128   onavg-ico8   sample-002.mgz
fsaverage           fsaverage6  onavg-ico16    README       V1_average
```

## Resample data using the `onavg` template

After uncompressing the `onavg` template, it can be used directly by FreeSurfer's `mri_vol2surf` command using the `--trgsubject` parameter.

For example, the typical syntax that fMRIPrep uses to perform resampling is:
```
mri_vol2surf --hemi ${LR} --interp trilinear -o ${OUT_FN} \
  --srcsubject ${SUBJ} \
  --reg ${WF_DIR}/bold_surf_wf/itk2lta/out.lta \
  --projfrac-avg 0.000 1.000 0.200 --mov ${NII_FN} \
  --trgsubject fsaverage5
```

To use the `onavg` template, simply replace "fsaverage5" with "onavg-ico32", which has the same resolution (10,242 vertices per hemisphere).

```
mri_vol2surf --hemi ${LR} --interp trilinear -o ${OUT_FN} \
  --srcsubject ${SUBJ} \
  --reg ${WF_DIR}/bold_surf_wf/itk2lta/out.lta \
  --projfrac-avg 0.000 1.000 0.200 --mov ${NII_FN} \
  --trgsubject onavg-ico32
```

### Example Python script

Here is a Python script that calls `mri_vol2surf` and resamples data to the `onavg` template space using the files from fMRIPrep's work directory.

Resampling using FreeSurfer 7.2 is performed directly.
Resampling using FreeSurfer 6 is performed using the singularity image of fMRIPrep 20.2.7, which was generated from the [Docker image](https://hub.docker.com/r/nipreps/fmriprep/tags).

````{admonition} Python script
:class: dropdown

```python
import os
from glob import glob
import subprocess
from joblib import Parallel, delayed


if __name__ == '__main__':
    dset, fmriprep_version = 'forrest', '20.2.7'
    bids_dir = os.path.realpath(os.path.expanduser(f'~/lab/BIDS/ds000113'))
    sids = ['01', '02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20']

    jobs = []
    for sid in sids:
        sd = os.path.expanduser(f'~/lab/fmriprep_out_root/forrest_20.2.7/output_{sid}/freesurfer')
        onavg = os.path.expanduser('~/lab/freesurfer/subjects/onavg-ico32')
        link = os.path.join(sd, 'onavg-ico32')
        if not os.path.exists(link):
            os.symlink(onavg, link)

        wf_root = os.path.join(
            os.path.expanduser(f'~/lab/fmriprep_work_root/{dset}_{fmriprep_version}/work_{sid}'),
            f'fmriprep_wf', f'single_subject_{sid}_wf')

        raw_bolds = sorted(glob(f'{bids_dir}/sub-{sid}/ses-movie/func/*_bold.nii.gz')) + \
            sorted(glob(f'{bids_dir}/sub-{sid}/ses-localizer/func/*_bold.nii.gz'))

        for raw_bold in raw_bolds:
            label = os.path.basename(raw_bold).split(f'sub-{sid}_', 1)[1].rsplit('_bold.nii.gz', 1)[0]
            label2 = label.replace('-', '_')
            wf_dir = (f'{wf_root}/func_preproc_{label2}_wf')

            nii_fn = f'{wf_dir}/bold_t1_trans_wf/merge/vol0000_xform-00000_merged.nii'

            for space in ['onavg-ico32', 'fsavg-ico32']:
                target = 'fsaverage5' if space == 'fsavg-ico32' else space
                for lr in 'lr':
                    out_dir = os.path.join(
                        os.path.expanduser('~/lab/nb-data/forrest/20.2.7/resampled'),
                        space, f'{lr}-cerebrum', '2step_freesurfer7.2')
                    out_fn = os.path.join(out_dir, f'sub-{sid}_{label}.gii')
                    if os.path.exists(out_fn):
                        continue
                    os.makedirs(out_dir, exist_ok=True)

                    lta_fn = f'{wf_dir}/bold_surf_wf/itk2lta/out.lta'
                    if not os.path.exists(lta_fn):
                        os.makedirs(os.path.dirname(lta_fn), exist_ok=True)
                        import nitransforms as nt
                        lta = f'{wf_dir}/../anat_preproc_wf/surface_recon_wf/t1w2fsnative_xfm/out.lta'
                        nt.linear.load(lta, fmt='fs', reference=nii_fn).to_filename(
                            lta_fn, moving=f'{sd}/sub-{sid}/mri/T1.mgz', fmt='fs')

                    cmd = [
                        'mri_vol2surf',
                        # '--cortex',
                        '--hemi', f'{lr}h',
                        '--interp', 'trilinear',
                        '--o', out_fn,
                        '--srcsubject', f'sub-{sid}',
                        '--reg', lta_fn,
                        '--projfrac-avg', '0.000', '1.000', '0.200',
                        '--mov', nii_fn,
                        '--trgsubject', target,
                        '--sd', sd,
                    ]

                    jobs.append(delayed(subprocess.run)(cmd))

                for lr in 'lr':
                    out_dir = os.path.join(
                        os.path.expanduser('~/lab/nb-data/forrest/20.2.7/resampled'),
                        space, f'{lr}-cerebrum', '2step_freesurfer6')
                    out_fn = os.path.join(out_dir, f'sub-{sid}_{label}.gii')
                    if os.path.exists(out_fn):
                        continue
                    os.makedirs(out_dir, exist_ok=True)

                    lta_fn = f'{wf_dir}/bold_surf_wf/itk2lta/out.lta'

                    cmd = [
                        'FS_LICENSE=$HOME/FS_license.txt',
                        'mri_vol2surf',
                        # '--cortex',
                        '--hemi', f'{lr}h',
                        '--interp', 'trilinear',
                        '--o', out_fn,
                        '--srcsubject', f'sub-{sid}',
                        '--reg', lta_fn,
                        '--projfrac-avg', '0.000', '1.000', '0.200',
                        '--mov', nii_fn,
                        '--trgsubject', target,
                        '--sd', sd,
                    ]
                    cmd = ' '.join(cmd)
                    cmd = [
                        'singularity', 'exec', '-e',
                        '-B', '/dartfs:/dartfs',
                        '-B', '/scratch:/scratch',
                        '-B', '/dartfs-hpc:/dartfs-hpc',
                        '-H', os.path.realpath(os.path.expanduser(f'~/lab/singularity_home/fmriprep')),
                        os.path.expanduser('~/lab/fmriprep_20.2.7.sif'),
                        '/bin/bash', '-c', cmd,
                    ]
                    jobs.append(delayed(subprocess.run)(cmd))

    with Parallel(n_jobs=-1) as parallel:
        parallel(jobs)
```
````

