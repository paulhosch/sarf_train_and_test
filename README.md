# saRFlood-3 Training and Testing

This repository contains the workflow for the third part of the **saRFlood pipeline**: Training and testing the Random Forest Classifier for SAR flood mapping.

## Features

- Multiple experiment iterations (resampling and retraining)
- LeaveOneGroup(site)Out Cross-Validation (LOGO-CV)
- Hyperparameter tuning (BayesSearchCV)
- Feature importance assessment (MDI, MDA, SHAP)

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
