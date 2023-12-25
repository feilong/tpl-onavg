# Optimizing the surface template

This section contains the code that was used to optimize the surface template, which is described in the Methods section of our [manuscript](https://doi.org/10.1101/2023.03.21.533686) under "Optimize the template using anatomy-based sampling."

:::{warning}
The code in this section is provided for reproducibility purposes only. **Do not create a new template for your own research** unless you have strong reasons to do so. This will make it unnecessarily difficult to compare your results with those of other studies and/or perform meta-analyses with your results.

For most of the users, we recommend using the standard `onavg` template as provided (see [Use the onavg template](index) of this documentation).
:::

:::{note}
Part of the code in these scripts is based on a legacy `surface` module, which is now deprecated.

These functions have been integreated into the [`neuroboros`](https://neuroboros.github.io/) package as the [surface](https://github.com/neuroboros/neuroboros/tree/main/src/neuroboros/surface) module.
:::

The three scripts correspond to the three steps of the optimization process:

- [Coarse optimization of vertex locations](optimization/optimize_surface_vertices.py)
- [Fine optimization of vertex locations](optimization/optimize_surface_vertices_refine.py)
- [Optimization of triangular faces](optimization/optimize_surface_triangles.py)
