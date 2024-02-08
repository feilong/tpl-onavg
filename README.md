The onavg template website
--------------------------

This repository contains the source code for the onavg template website, including the code to create the template and code to replicate the analyses.
The website is hosted on GitHub Pages: https://feilong.github.io/tpl-onavg/

The analyses in the repository correspond to the replication analyses in the manuscript based on the _Raiders_ dataset and the _Budapest_ dataset.
To replicate the results based on the _Forrest_ dataset, simply replacing the dataset name to _Forrest_ (`nb.Forrest` or `nb.dataset("Forrest")`) and re-run the scripts.
While running the script, data will be automatically downloaded from the G-node GIN repository (https://gin.g-node.org/neuroboros/forrest) if a local copy does not exist.

The original analysis was performed in a Python 3.8.12 environment with standard scientific computing packages and libraries, including numpy 1.16.4, scipy 1.3.0, pandas 0.24.2, joblib 0.13.2, matplotlib 3.1.0, and seaborn 0.9.0. The data analysis was performed using the Discovery HPC cluster of Dartmouth College.

The code is purely Python. It does not require any specific hardware, and it should be able to on any operating systems with recent Python and dependency packages.
See `pip_freeze.txt` for the Python environment we used for the recent replication analyses, including software packages and versions.
Not all the packages are needed for the analysis.
All dependencies can be installed with `pip` or `conda`, usually within minutes.

To view example output of the code, please see the corresponding sections (Replication of key analyses, Vertex properties, MVPA performance, Computational time) of the [website](https://feilong.github.io/tpl-onavg/).

To replicate the analyses with other datasets, simply change the input data in the script. For example, you can use `nb.Life` to run the analyses using the Life dataset.
