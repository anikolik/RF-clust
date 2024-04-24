# project
project_name = "RF+Clust"

# data 
suite_name = "cec2014"
features = "all"
target = "log_precision"
transformation = "std"
n_folds = 30
crossval_column = 0
budgets = [5000]
algorithms = ["DE1", "DE2", "DE3"]

# model
model_name = "random_forest"
explainer_type = "tree"

# Rf+Clus 
metric = "cosine"
calibration_method = "weighted"
similarity_thresholds = [0.5, 0.7, 0.9]
weights_method = None

# visualization 
logscale=False
vmax = 8