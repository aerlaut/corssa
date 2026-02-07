# COaRse-grained Secondary Structure Annotation (CORSSA)

Repository for the COaRse-grained Secondary Structure Annotation (CORSSA)

## Requirements

- Python 3.13

## Installation

Install package from Github

```
pip install https://github.com/aerlaut/corssa.git@main
```

## Training new models

Training new models on a new coarse-graining scheme involves the steps below. See `training_new_model.ipynb` in the `examples` folder to get started.

### 1. Prepare Dataset

The model requires features derived from the coarse-graining schema. A `corssa.coarse_graining.featurizer.Featurizer` class is included in the library, that creates features from a coarse-grained representation of a peptide. The `Featurizer` expectes a Pandas DataFrame with the columns `x_rep`, `y_rep`, `z_rep` that denotes the coordinates of the coarse-grained bead.

If you do not have your own dataset, you can use the dataset at https://doi.org/10.5281/zenodo.18506092. This data contains CIF files derived from the [non-redundant CATH S40 dataset](https://www.cathdb.info/wiki?id=data:index) and corresponding DSSP annotations (ground-truth) processed using DSSP 4.1.

### 2. Define new coarse graining scheme

Define a new coarse graining scheme by subclassing `corssa.coarse_graining.CoarseGrainModel`. Override the `scheme` function.

### 3. Prepare feature CSVs

Prepare features based on the new coarse-graining schema.

### 4. Model training

Train a new model by subsclassing `corssa.model.CORSSA`. The model is a subclass of the [CatboostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier) library with some default settings added. You can modify these settings however you see fit.

## Using pre-trained weights

Three models have been trained on Cα, Cβ and centre-of-mass (CoM) representation. To use these pre-trained models, you need to be using the same coarse-graining.

See `use_existing_model.ipynb` in the `examples` folder to get started.