# sarf_train_and_test

## A python pipeline to evaluate ml for flood mapping based on sar

## A python workflow to train different models on differemt sample sets and sizes, tune the models and evaluate the performance and feature importance of the models

## I want to use a LOGO-CV accross three different case studies

## I have all the different training sample sets and validation sample sets stored like this:

## data/case_studies/valencia/sample_sets/training/100/samples/simple_random.csv

## data/case_studies/valencia/sample_sets/testing/100/samples/balanced_systematic.csv

## I want to compare differnt configurations these should be set up in a .json file this file should contain the following information:

    - the name of the experiment

### the experiment variables (the combination of the variables will be the different configurations compared in the experiment)

    - the models to compare ( Random Forest, Balanced Random Forest, Weighted Random Forest)
    - the sample_strategies to compare for training eg (simple_random, simple_systematic, simple_grts, balanced_random, balanced_systematic, balanced_grts, proportional_random, proportional_systematic, proportional_grts)
    - the sample sizes to compare for training (100, 500, 1000)
    - the method of tuning the mdoels (no_tuning, bayesian_optimization)

### the experiment constants

    - the names of the case studies to use (valencia, danube, oder)
    - the sample sizes to use for testing (1000)
    - the sample_strategies to use for testing (balanced_systematic, proportional_systematic)
    - the parameter grid for tuning the models
    - the number of iterations to run the whole experiment (eg 10)
    - the sar features (columns) to use (VV_POST, VV_PRE, VH_POST, VH_PRE, VV_CHANGE, VH_CHANGE, VV_VH_RATIO_PRE, VV_VH_RATIO_POST, VV_VH_RATIO_CHANGE)
    - the contextual features (columns) to use (SLOPE, LAND_COVER, HAND, EDTW)
    - the training column to use for the evaluation (LABEL)
    - the testing column to use for the evaluation (LABEL)
    - the methods to investigate the feature importance (mdi, mda, shap_importance)

## I want to use both a LOGO-CV accross three different case studies and investigate the performance when the model is trained and evaluated on the same casestuddy

## I want to evaluate every possible combination of my experiment variables accross multiple iterations

### the reuslts should be stored in a csv file

### I want to compute and store OA F1 and the relative Comfusion matrix use the defined sample strategies for testing

### I want to compute the feature imprtance using three different methods defiend in the experiment constants I want to scale them from 0 to 1 to make them omparable

### I want to tune the params with BaysianSerach CV I want to use a custom scoring function

#### I want to use a combined metric score for validation 50% prportional_OA and 50% proportional_F1 defined in config json as well as the number of iterations and cvs

### I want to have cross site and same site results, cross site means within the LOGO-CV loop the model is trained on two of ththree case studies and the other one is used for testing

### I want to keep track on witch sitees we trained on and on withc we tested on

### but in addition to the testing with the sample set of the left out side I want to test the model predicting also on a combined dataset of the two sites that were trained on (500+500)

## thee csv should contain the following columns for each iteration, for each configuratio (combination of experiment variables), for each testing site used in the logo_cv loop

- iteration
- configuration_name
- model
- training sample size
- training sample strategy
- best params (if tuned otherwise NAN)
- testing_site
- training_sites
- proportional_OA_same_site
- proportional_F1_same_site
- proportional_Confusion_Matrix_same_site
- balanced_OA_same_site
- balanced_F1_same_site
- balanced_Confusion_Matrix_same_site
- proportional_OA_cross_site
- proportional_F1_cross_site
- proportional_Confusion_Matrix_cross_site
- balanced_OA_cross_site
- balanced_F1_cross_site
- balanced_Confusion_Matrix_cross_site
- feature_importance_mdi
- feature_importance_mda
- feature_importance_shap

## I also want to store all of the trained models

## I aslo want to visulaize the results

### One plot should plot boxplots of balanced and proportional OA and F1 in a 2x2 subplot (top OA botoom F1, left proportional, roght balanced) I want to have the following parameters for this function, what to compare (x axis labels, the same for each subplot) eg. all (for all confgurations) or se selaction by config names or a differnt type of experiment variable to compare eg. tuned vs untuned, or sample sizes, the

### also specify whataccross what distirbution the boxplot should be plotted eg accross the iteations, accross the logo folds

### I want to have one plot that ompares same site with cross site pertfromance for each testing site in one plot

### I want to have a function that plots the confusion matrix comparing two configurations it should be 2x2 subplots on the left config 1 on the irght config 2 s cfm, the top should be the proportional ccfm and bottom should be the balanced cfm

#### one plot should be a boxplots plots of the feature importance for each method and each feature, the features should be sorted by the mean importance accross all iterations and methods

#### x axis features grouped by methods, yaxis importance 0-1 combine featureswith a very very low feature importance as others

## the reults should be stored to data/experiments/experiment_name/results.csv

## the models should be stored to data/experiments/experiment_name/models/

## the plots should be stored to data/experiments/experiment_name/plots/

## Implementation Approach

Focus on modular, reusable components
Use object-oriented approach for configuration and experiment tracking
Implement proper error handling for files/directories
Optimize for parallel processing where possible
Use interactive Python cells for easy execution and debugging
This structured approach will ensure a clean, maintainable codebase that addresses all your requirements.
give meaningfull variable, function and filenames
do not create to large code files max 200 lines create more individual files in a well strutured dir with meaningfull folders
