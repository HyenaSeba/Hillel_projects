# Hillel_projects

Projects from Hillel Machine Learning class

During the classes several Machine Learning projects were completed.

**1. The formula for the flight range of a bullet fired at an angle to the horizon.**

- Generate data arrays for angle and initial velocity from a normal and uniform distribution
- Calculate the flight range distribution
- Construct a histogram of the flight range
- Fill out a report based on research results

**2. Experiments with minimization**

- Plot loss function value (should drop over the fitting, loss = f(epoch))
- Try RMSE, MAE and maybe other losses for linear regression
- Make animation for fitting: plots of changing fitting curve (line) over the data (see slides)
- Experiment with non-linear data, for example:
      y = 2 * x**2 + x + 3.5 + noise
- Experiment with number of samples, sigma, and optimization algorithms

**3. Computation Graph. Derivative, gradient, learning rate, gradient descent**

- Take analytical derivative of sigmoid function (on a paper with pencil, take a photo of your calculation and attach it to pdf report)
- Experiments with   demo code (gradient descent)
      * Vary learning rate
      * Vary epochs
      * Plot MSE over training (over epochs for specific learning rate)
- Make one forward and backward steps for L = (2a + b)(c - d), where a, b, c, d are arbitrary numbers

**4. Experiments with NN**

--== Pytorch NN ==--
- Plot loss curve (both train and test)
- Print model weights (before and after training) and compare with original ones
- Plot results (predicted line along with dataset data)

**5. Classification Training and Tuning**

Make 3 jupyter notebooks for classification of 3 datasets (one notebook for one dataset). Achieve at least 95% accuracy on each dataset, and if possible even 100% (accuracy is measured on the test set) with as few neural network parameters as possible.

- make_blobs() -- classification into 4 classes
- make_circles() -- classification into 2 classes
- make_moon() -- classification into 2 classes

**6. Experiment with multiclass classification (MLP) for MNIST data set**

- Include dropout layers
- Include batch normalization layers
- Include more layers
- Experiment with activation functions
- Experiment presence / absence of dropout and batch normalization
- Experiment with batch size
- Plot loss = f(epochs) for each experiment

**7. PCA use cases**

- Get dataset from Kaggle (any tabular dataset you want)
- Make simple classifier / regressor on the dataset
- Reasonably reduce dataset dimensionality
      * Plot explained variance
      * Explain chosen number of components
- Retrain the same classifier / regressor on the dataset with reduced dimensionality
- Compare accuracies / MSEs and speed of the two approaches (with and without dimensionality reduction)

**8. Classification metrics on MNIST**

- Accuracy (per class and general)
- Precision (per class and general)
- Recall (per class and general)
- F1-score (per class and general)
- Confusion matrix
- Classification report

**9. Convolutional neural network (CNN)**

- Calculate number of weights on each layer
- Calculate shape of tensors before and after each layer
- Make model overfit the data. Show loss curves with overfit
- Reduce model complexity (number parameters) with keeping accuracy
- Add batch norm as well
  (add loss curve plot, accuracy plot, classification report and confusion matrix as well)

**10. CV Architectures, fine-tuning**

- Transfer learning for your dataset on a chosen pretrained model
- Construct your own simple architecture and train it on your dataset

**11. Ensemble methods**

- Train classifiers on the dataset
- There should be 3 classifiers (stacking, boosting, bagging)
  Mandatory steps:
    - primary data analysis (gap distance, presence of categorical features, ...)
    - feature engineering (build 1-2 new features)
    - scaling feature
    - division of the dataset into training, validation and test parts
    - training the base model with default hyperparameters
    - selection of hyperparameters
    - evaluation of results

**12. Text classification**

- Negative/Positive review classification

