# Outline of benchmarking analyses

## Vertex properties

Here is a list of Juptyer Notebooks and pages that characterize how vertex properties vary across the cortex for `onavg` and other templates.

- [](vertex_properties/inter-vertex_distance)
- [](vertex_properties/variation_across_vertices)
- [](vertex_properties/vertex_distribution)

## MVPA performance

Here is a list of Python scripts and Jupyter Notebooks to replicate the MVPA analyses in the paper. The scripts are approximately numbered in the order they should be run.

- **Create hyperalignment template**
    - [](mvpa/10_hyperalignment_template.py)
- **Compute hyperalignment transformations**
    - [](mvpa/20_hyperalignment_transformation.py)
- **Whole-brain movie time point classification**
    - [](mvpa/30_wholebrain_clf.py)
    - [](mvpa/31_summarize_wholebrain_clf)
- **Searchlight movie segment classification**
    - [](mvpa/32_sl_clf.py)
    - [](mvpa/33_summarize_sl_clf)
- **Searchlight RSA-ISC**
    - [](mvpa/40_sl_rsa.py)
    - [](mvpa/41_summarize_sl_rsa)

## Computational time

This Jupyter Notebook summarizes the computational time of the analyses above.

- [](mvpa/50_summarize_computational_time)
