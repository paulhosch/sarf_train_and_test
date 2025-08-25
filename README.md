# saRFlood-3 Training and Testing

This repository contains the workflow for the third part of the **saRFlood pipeline**: Training and testing the Random Forest Classifier for SAR flood mapping.

## Features

- Multiple experiment iterations (resampling and retraining)
- LeaveOneGroup(site)Out Cross-Validation (LOGO-CV)
- Hyperparameter tuning (BayesSearchCV)
- Feature importance assessment (MDI, MDA, SHAP)
- Per-iteration sampling (use different training/testing samples for each iteration)

---

## Installation & Usage

### Option 1: Complete Environment Recreation (Recommended)

1. Clone the repository

```bash
git clone https://github.com/paulhosch/sarf_train_and_test.git
cd sarf_train_and_test
```

2. Create environment from environment.yml (cross-platform)

```bash
conda env create -f environment.yml
conda activate sarf_train_py38
```

### Option 2: Exact Environment Replication

For exact replication of the development environment (same OS):

```bash
conda create --name sarf_train_py38 --file conda-explicit.txt
conda activate sarf_train_py38
```

### Option 3: Manual Installation

1. Clone the repository

```bash
git clone https://github.com/paulhosch/sarf_train_and_test.git
cd sarf_train_and_test
```

2. Create and activate a conda environment with Python 3.8

```bash
conda create -n sarf_train_py38 python=3.8 -y
conda activate sarf_train_py38
```

3. Install core geospatial packages via conda

```bash
conda install -c conda-forge geopandas rasterio shapely fiona pyproj -y
```

4. Install remaining requirements

```bash
pip install -r requirements.txt
```

### Environment Files

- `environment.yml`: Cross-platform conda environment (recommended)
- `conda-explicit.txt`: Exact environment specification with URLs and checksums
- `requirements.txt`: Core packages for manual installation

---

## Experiment Configuration

All experiment variables and parameters are defined in `experiment_config.json`. Example:

```json
"experiment_variables": {
  "models": [
    "Random Forest",
    "Balanced Random Forest",
    "Weighted Random Forest"
  ],
  "training_sample_strategies": [
    "simple_random",
    "simple_systematic",
    "simple_grts",
    "balanced_random",
    "balanced_systematic",
    "balanced_grts",
    "proportional_random",
    "proportional_systematic",
    "proportional_grts"
  ],
  "training_sample_sizes": [100, 500, 1000],
  "tuning_methods": ["no_tuning", "BayesSearchCV"]
},
```

Other experiment parameters, such as the number of iterations or the hyperparameter search space, are also defined in this file.

---

## Running Experiments

The main experiment runner script is `experiment_runner.py`. It supports a flag for per-iteration sampling:

```python
# Set to True to use different sample sets for each iteration
new_samples_each_iteration = True
```

When this flag is enabled, the pipeline looks for samples in iteration-specific directories:

```
<input_data>/<case_study>/samples/iteration_<n>/<dataset_type>/<size>/samples/<strategy>.csv
```

When disabled, the same samples are used for all iterations:

```
<input_data>/<case_study>/samples/<dataset_type>/<size>/samples/<strategy>.csv
```

---

## Performance Evaluation

The pipeline evaluates every possible variable combination across each iteration and LOGO-CV fold, resulting in an extensive evaluation space of 162 differently trained models.

**Performance metrics** are computed from independent testing datasets using `proportional_systematic` or `balanced_systematic` sampling. Testing is performed both on the training sites of each LOGO fold (**same_site**) and on the left-out site (**cross_site**):

- `proportional_OA_same_site`
- `proportional_F1_same_site`
- `balanced_OA_same_site`
- `balanced_F1_same_site`
- `proportional_OA_cross_site`
- `proportional_F1_cross_site`
- `balanced_OA_cross_site`
- `balanced_F1_cross_site`

Results are stored in `results.csv` and can be visualized using the provided plotting scripts.
