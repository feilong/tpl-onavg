# Small numerical differences
The numbers based on the ``ouroboros`` package slightly differ from those reported in the manuscript.
- In the manuscript, the inter-vertex distance was computed by first averaging across all neighbors, and then averaging across all participants; In this Jupyter Book, the distance was first averaged across all participants, and then averaged across neighbors. Note that we used 95% trimmed mean when averaging across participants, and therefore these two steps are not commutative, and small numerical differences can occur based on the order of the operations.
- Resampling vertex properties from the native space of each participant.
- fMRIPrep version.
