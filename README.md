# saRFlood-3 Training and Testing

This repository contains the workflow for the third part of the **saRFlood pipeline**: Training and testing the Random Forest Classifier for SAR flood mapping.

## Features

- Multiple experiment iterations
- Leave-One-Group (site)-Out Cross-Validation (LOGO-CV)
- BayesSearchCV hyperparameter tuning
- Systematic square grid sampling
- Feature importance assessment (MDI, MDA, SHAP)
- Flexible sample selection (per-iteration or fixed)

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

## Performance Evaluation

The pipeline evaluates every possible variable combination across each iteration and LOGO-CV fold, resulting in an extensive evaluation space.

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

---

## Per-Iteration Sampling (Optional)

You can optionally use **different training and testing samples for each iteration**. To enable this, set the flag in `experiment_runner.py`:

```python
new_samples_each_iteration = True  # Use per-iteration samples
```

When enabled, the pipeline will load sample sets for each iteration using the following path convention:

```
/case_studies/{site_id}/samples/{iteration_id}/{train_or_test}/{sample_size}/samples/{strategy}.csv
```

- `iteration_id` is e.g. `iteration_1`, `iteration_2`, ...
- `train_or_test` is `train` or `test`
- `sample_size` is the integer sample size

If disabled, the pipeline uses a fixed sample set for all iterations.

---

## Visualization

Results can be plotted in various ways to compare different class and spatial distributions in the sample selection. See the visualization scripts for details.

---

## Getting Started

1. Edit `experiment_config.json` to define your experiment variables and parameters.
2. (Optional) Prepare per-iteration sample sets if you want to use new samples each iteration.
3. Run the experiment using `experiment_runner.py`.
4. Visualize and analyze the results.

---

For more details, see the code and comments in each script.
