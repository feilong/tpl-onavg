# Use with Python

We wrote the [`process` package](https://github.com/feilong/process) to resample fMRI data onto cortical surface.
It reads the transforms from the intermediate files in [fMRIPrep](https://fmriprep.org/en/stable/)'s work directory and performs resampling based on these transforms.

This package implements two kinds of methods to resample data onto cortical surface.
1.  The "1step" methods.
    These methods first combine all transforms together, and resample directly from the original BOLD files onto the cortical surface.
    Therefore, the resampling only involves one interpolation, and it avoids accumulating noises related to interpolation.
2.  The "2step" methods, which mimic the behavior of fMRIPrep.
    They read the data that have been resampled to the subject's T1w space, and resample the data again onto the cortical surface.
    The results are expected to be similar to those based on [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)'s `mri_vol2surf`.
    Specifically, results of "2step_normals-sine_nnfr" should resemble those of FreeSurfer 6, and results of "2step_normals-equal_nnfr" should resemble those of FreeSurfer 7.2.

Note that when resampling onto the cortical surface, both kinds of methods resample to the subject's native space (i.e., the native surface mesh).
The data is then aggregated and downsampled to the template space.
Therefore, "1step" and "2step" only refers to the number of interpolations from original BOLD to surface, and this additional downsampling step on cortical surface does not count.

See [here](https://github.com/feilong/process/blob/main/scripts/forrest.py) for an example script to use the package to preprocess the [StudyForrest](https://www.studyforrest.org/) 3 T data. The package also affords packaging and compressing the output directories for archival purposes.

Moreover, the [`neuroboros` package](https://neuroboros.github.io/) affords access to various data based on the `onavg` template.
See the "Replication of key analyses" section on example Python code to use the `onavg` template and the `neuroboros` package to replicate the key analyses of [our manuscript](https://doi.org/10.1101/2023.03.21.533686).
