# ML4MI
This is the project repository for Team B.C.'s class project.

# Preprocesing
To preprocess data for use by the models, use the preprocessing.py script or for a more visual interaction preprocessing.ipynb

# Model Creation
To Create a new model use the model_builder.py script or for a more visual interaction model_builder.ipynb

## Model_Builder.py
-h can be used in model_builder.py to see the list of command line options

-m should be used to specify the model to run: DT, ANN, SVM

-i should be used to specify the input path to directory with TRAIN, TEST, VALIDATION directories

-o should be used to specify the path to place output artifacts, needs METRICS and MODEL subdirs

-c can be used when the model is DT to specify the criteria

-d can be used when the model is DT to specify the depth

-k can be used when the model is SVM to specify the kernel type: linear, poly, rbf, sigmoid, precomputed

-r can be used when the model is SVM to specify the C-value (regularization)

-s can be used when the model is ANN to set earlystopping to "true", to omit early stopping, do not add this flag

-a can be used when the model is ANN to specify an activation function: relu, sigmoid, tanh

-l can be used when the model is ANN to specify a learning rate


## Model_Builder.ipynb
The last 3 sections correspond to each of the models. 

The notebook pulls down the data from the repo at the start of the run.

Each model can be configured to use a new input and output directory at the top of its section.

The ANN model can additionally be reconfigured to toggle early stopping on or off, change the activation function, or update the learning rate.

earlystop = True
activation = "relu"
learningrate = 0.00025
output = "/content/ML4MI/OUTPUT/"
input = "/content/ML4MI/INPUT/"

The Decision Tree model can be reconfigured to change the criteria and the maximum depth.

criteria = "entropy" # criteria to use for Decision Tree: entropy or gini
depth = 5 # cutoff depth of tree
output = "/content/ML4MI/OUTPUT"
input = "/content/ML4MI/INPUT/DT"

The SVM can be reconfigured to change the kernel and the regularization (C value). 

kernel = "linear"  # kernel type for SVM: linear, poly, rbf, sigmoid, precomputed
regularization = 1.0 # regularization amount for SVM (must be positive)
output = "/content/ML4MI/OUTPUT/"
input = "/content/ML4MI/INPUT/"

# Model Loading
To load and use a saved model use the model_runner.py script
