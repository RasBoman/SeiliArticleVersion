# SeiliArticleVersion

To be modified before running the final models:
 - Adding link strength measurement?
 - Season-variable as categorical (1:Winter, 2:Spring, 3: Summer, 4:Fall)
 
Data for Bayesian model: 
- SLICED version for testing functionality (10 obs)
- Full data set (log10 and scaled) with ~240 observations for running the final model.

MATLAB-models with hidden variables:
- Naive Bayes 
  -> Links from Season & Hidden variable to plankton
  -> Environmental variables ignored but still in dataset as focus on biological interactions (see e-mail)

- "Medium" Bayes
  -> More links between env. variables and plankton groups but not between time slices
  
- Dynamic Bayes
  -> Links between time slices added


After running the basic models:
  - Predictions of each model
  - Some other validation?
