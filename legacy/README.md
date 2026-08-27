# Legacy notebook

`Homework.ipynb` is the original 2022 assignment preserved for historical context. It explored image resizing, PCA visualization, k-nearest-neighbour classification, and regularized NMF.

It is not the supported entry point because:

- its image dataset was stored in a personal Google Drive path and is unavailable from a clean clone;
- execution depends on Colab-specific setup;
- the notebook contains state-dependent NMF cells;
- classification accuracy is not a retrieval evaluation.

The exact pre-migration repository is preserved by the `legacy-pca-v1` Git tag. The modern implementation retains PCA as a tested retrieval baseline under `src/`.
