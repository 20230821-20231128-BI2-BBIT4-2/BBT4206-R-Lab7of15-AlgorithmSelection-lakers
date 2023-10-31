Business Intelligence Lab Submission Markdown
================
<Lakers>
\<30-10-2023\>

- [Student Details](#student-details)
- [Setup Chunk](#setup-chunk)
- [STEP 1. Install and Load the Required Packages
  —-](#step-1-install-and-load-the-required-packages--)
  - [arules —-](#arules--)
  - [arulesViz —-](#arulesviz--)
  - [tidyverse —-](#tidyverse--)
  - [readxl —-](#readxl--)
  - [knitr —-](#knitr--)
  - [ggplot2 —-](#ggplot2--)
  - [lubridate —-](#lubridate--)
  - [plyr —-](#plyr--)
  - [dplyr —-](#dplyr--)
  - [naniar —-](#naniar--)
  - [RColorBrewer —-](#rcolorbrewer--)
    - [7a - Algorithm Selection for Classification and
      Regression](#7a---algorithm-selection-for-classification-and-regression)
- [A. Linear Algorithms](#a-linear-algorithms)
  - [1. Linear Regression](#1-linear-regression)
- [Define an 80:20 train:test data split of the diamonds
  dataset](#define-an-8020-traintest-data-split-of-the-diamonds-dataset)
- [Train the linear regression
  model](#train-the-linear-regression-model)
- [Display the model’s details](#display-the-models-details)
- [Make predictions on the test set](#make-predictions-on-the-test-set)
- [SSR](#ssr)
- [SST](#sst)
- [R Squared](#r-squared)
- [MAE](#mae)
- [Load the Sonar dataset from the mlbench
  package](#load-the-sonar-dataset-from-the-mlbench-package)
- [Define a 70:30 train:test data
  split](#define-a-7030-traintest-data-split)
- [Define train control with 5-fold
  cross-validation](#define-train-control-with-5-fold-cross-validation)
- [Train the logistic regression
  model](#train-the-logistic-regression-model)
- [Display the model’s details](#display-the-models-details-1)
- [Make predictions](#make-predictions)
- [Display the model’s evaluation
  metrics](#display-the-models-evaluation-metrics)
- [Plot the confusion matrix](#plot-the-confusion-matrix)
  - [3. Linear Discriminant Analysis](#3-linear-discriminant-analysis)
- [Define a 70:30 train:test data split for
  LDA](#define-a-7030-traintest-data-split-for-lda)
- [Train the LDA model](#train-the-lda-model)
- [Display the model’s details](#display-the-models-details-2)
- [Make predictions using the LDA model on the test
  set](#make-predictions-using-the-lda-model-on-the-test-set)
- [Display the evaluation metrics (confusion
  matrix)](#display-the-evaluation-metrics-confusion-matrix)
  - [4a Regularized Linear Regression
    Classification](#4a-regularized-linear-regression-classification)
- [Train the regularized logistic regression model using
  glmnet](#train-the-regularized-logistic-regression-model-using-glmnet)
- [Display model’s details](#display-models-details)
- [Make predictions on the Sonar dataset using the trained
  model](#make-predictions-on-the-sonar-dataset-using-the-trained-model)
- [Display evaluation metrics (confusion
  matrix)](#display-evaluation-metrics-confusion-matrix)
  - [4b Regularized Linear Regression
    Regression](#4b-regularized-linear-regression-regression)
- [Define a 70:30 train:test data split of the
  dataset](#define-a-7030-traintest-data-split-of-the-dataset)
- [Train the model using glmnet (regularized linear
  regression)](#train-the-model-using-glmnet-regularized-linear-regression)
- [Make predictions on the test
  set](#make-predictions-on-the-test-set-1)
- [Calculate evaluation metrics](#calculate-evaluation-metrics)
- [B. Non-Linear Algorithms](#b-non-linear-algorithms)
  - [1. Classification and Regression
    Trees](#1-classification-and-regression-trees)
- [Define a 70:30 train:test data split of the
  dataset](#define-a-7030-traintest-data-split-of-the-dataset-1)
- [Train the model using the rpart method (decision
  tree)](#train-the-model-using-the-rpart-method-decision-tree)
- [Display the model’s details](#display-the-models-details-3)
- [Make predictions on the test
  set](#make-predictions-on-the-test-set-2)
- [Display the model’s evaluation metrics (confusion
  matrix)](#display-the-models-evaluation-metrics-confusion-matrix)
- [Plot the confusion matrix](#plot-the-confusion-matrix-1)
  - [1b Decision tree for a regression problem with
    CARET](#1b-decision-tree-for-a-regression-problem-with-caret)
- [Load the mtcars dataset](#load-the-mtcars-dataset)
- [Split the mtcars dataset into training and testing sets (70:30
  split)](#split-the-mtcars-dataset-into-training-and-testing-sets-7030-split)
- [Train the model with decision tree using the rpart
  method](#train-the-model-with-decision-tree-using-the-rpart-method)
- [Display the model’s details](#display-the-models-details-4)
- [Calculate evaluation metrics](#calculate-evaluation-metrics-1)
- [Define a 70:30 train:test data split of the
  dataset](#define-a-7030-traintest-data-split-of-the-dataset-2)
- [Train the Naive Bayes model using 5-fold
  cross-validation](#train-the-naive-bayes-model-using-5-fold-cross-validation)
- [Display the model’s details](#display-the-models-details-5)
- [Make predictions on the test
  set](#make-predictions-on-the-test-set-3)
- [Display the model’s evaluation metrics (confusion
  matrix)](#display-the-models-evaluation-metrics-confusion-matrix-1)
- [Plot the confusion matrix](#plot-the-confusion-matrix-2)
  - [3. k-Nearest Neighbours](#3-k-nearest-neighbours)
    - [3a kNN for a classification problem with CARET’s train
      function](#3a-knn-for-a-classification-problem-with-carets-train-function)
- [Define a 70:30 train:test data split of the
  dataset](#define-a-7030-traintest-data-split-of-the-dataset-3)
- [Train the kNN model with 10-fold cross-validation and data
  standardization](#train-the-knn-model-with-10-fold-cross-validation-and-data-standardization)
- [Display the model’s details](#display-the-models-details-6)
- [Make predictions on the test
  set](#make-predictions-on-the-test-set-4)
- [Display the model’s evaluation metrics (confusion
  matrix)](#display-the-models-evaluation-metrics-confusion-matrix-2)
- [Plot the confusion matrix](#plot-the-confusion-matrix-3)
- [Replace missing values with mean for the target variable
  “Ozone”](#replace-missing-values-with-mean-for-the-target-variable-ozone)
- [Define a target variable and predictors for
  regression](#define-a-target-variable-and-predictors-for-regression)
- [Split the data into train and test
  sets](#split-the-data-into-train-and-test-sets)
- [Apply the 5-fold cross-validation resampling method and standardize
  data](#apply-the-5-fold-cross-validation-resampling-method-and-standardize-data)
- [Display the model’s details](#display-the-models-details-7)
- [Make predictions on the test
  set](#make-predictions-on-the-test-set-5)
- [Calculate evaluation metrics for the
  model](#calculate-evaluation-metrics-for-the-model)
- [Display the model’s evaluation
  metrics](#display-the-models-evaluation-metrics-1)
  - [4. Support Vector Machine](#4-support-vector-machine)
    - [4a SVM Classifier for a classification problem with
      CARET](#4a-svm-classifier-for-a-classification-problem-with-caret)
- [Load the necessary libraries](#load-the-necessary-libraries)
- [Define target variable and predictors for the iris
  dataset](#define-target-variable-and-predictors-for-the-iris-dataset)
- [Define a 70:30 train:test data split of the iris
  dataset](#define-a-7030-traintest-data-split-of-the-iris-dataset)
- [Train the LSVM model with radial
  kernel](#train-the-lsvm-model-with-radial-kernel)
- [Display the model’s details](#display-the-models-details-8)
- [Make predictions](#make-predictions-1)
- [Display the model’s evaluation metrics - Confusion
  Matrix](#display-the-models-evaluation-metrics---confusion-matrix)
- [Visualize the confusion matrix using a heat
  map](#visualize-the-confusion-matrix-using-a-heat-map)
  - [4b SVM classifier for a regression problem with
    CARET](#4b-svm-classifier-for-a-regression-problem-with-caret)
- [Load the ‘swiss’ dataset](#load-the-swiss-dataset)
- [Define predictors and target
  variable](#define-predictors-and-target-variable)
- [Define an 80:20 train:test data split of the
  dataset](#define-an-8020-traintest-data-split-of-the-dataset)
- [Train the model using SVM
  regression](#train-the-model-using-svm-regression)
- [Display the model’s details](#display-the-models-details-9)
- [Make predictions](#make-predictions-2)
- [Calculate evaluation metrics](#calculate-evaluation-metrics-2)
- [RMSE](#rmse)
- [SSR](#ssr-1)
- [SST](#sst-1)
- [R Squared](#r-squared-1)
- [MAE](#mae-1)
- [7b Algorithm Selection for
  Clustering](#7b-algorithm-selection-for-clustering)
- [Check for missing values (There are no missing values in the iris
  dataset)](#check-for-missing-values-there-are-no-missing-values-in-the-iris-dataset)
- [Compute the correlations between
  variables](#compute-the-correlations-between-variables)
- [Basic Table (View the correlation
  matrix)](#basic-table-view-the-correlation-matrix)
- [Basic Plot (Correlation matrix using corrplot
  package)](#basic-plot-correlation-matrix-using-corrplot-package)
- [Fancy Plot using ggplot2 (Heatmap visualization of the correlation
  matrix)](#fancy-plot-using-ggplot2-heatmap-visualization-of-the-correlation-matrix)
- [Scatter plot comparing Sepal Length vs. Sepal Width with Species
  differentiation](#scatter-plot-comparing-sepal-length-vs-sepal-width-with-species-differentiation)
- [Scatter plot comparing Petal Length vs. Petal Width with Species
  differentiation](#scatter-plot-comparing-petal-length-vs-petal-width-with-species-differentiation)
- [Scatter plot comparing Sepal Length vs. Petal Length with Species
  differentiation](#scatter-plot-comparing-sepal-length-vs-petal-length-with-species-differentiation)
- [Scatter plot comparing Sepal Width vs. Petal Width with Species
  differentiation](#scatter-plot-comparing-sepal-width-vs-petal-width-with-species-differentiation)
- [Scatter plot comparing Sepal Length vs. Petal Width with Species
  differentiation](#scatter-plot-comparing-sepal-length-vs-petal-width-with-species-differentiation)
  - [Select the features to use to create the
    clusters](#select-the-features-to-use-to-create-the-clusters)
- [Create the clusters using the K-Means Clustering
  Algorithm](#create-the-clusters-using-the-k-means-clustering-algorithm)
- [Define the maximum number of clusters to
  investigate](#define-the-maximum-number-of-clusters-to-investigate)
- [Initialize the total within sum of squares error
  (wss)](#initialize-the-total-within-sum-of-squares-error-wss)
- [Investigate 1 to n possible
  clusters](#investigate-1-to-n-possible-clusters)
- [Add the cluster number as a label for each
  observation](#add-the-cluster-number-as-a-label-for-each-observation)
  - [View the results by plotting scatter plots with the labelled
    cluster](#view-the-results-by-plotting-scatter-plots-with-the-labelled-cluster)
- [Load the Groceries dataset](#load-the-groceries-dataset)
- [View the structure of the Groceries
  dataset](#view-the-structure-of-the-groceries-dataset)
- [Show the first few transactions](#show-the-first-few-transactions)
- [Create a transactions object](#create-a-transactions-object)
- [Get item frequency](#get-item-frequency)
- [Sort item frequency in descending
  order](#sort-item-frequency-in-descending-order)
- [Plotting the top 10 absolute item
  frequencies](#plotting-the-top-10-absolute-item-frequencies)
- [Plotting the top 10 relative item
  frequencies](#plotting-the-top-10-relative-item-frequencies)
- [Print the summary and inspect the association rules (Option
  1)](#print-the-summary-and-inspect-the-association-rules-option-1)
- [Remove redundant rules](#remove-redundant-rules)
- [Display summary and inspect non-redundant
  rules](#display-summary-and-inspect-non-redundant-rules)
- [Write the non-redundant rules to a CSV
  file](#write-the-non-redundant-rules-to-a-csv-file)
- [Create a transactions object](#create-a-transactions-object-1)
- [Set the minimum support, confidence, and
  maxlen](#set-the-minimum-support-confidence-and-maxlen)
- [Option 1: Create association rules based on stock
  code](#option-1-create-association-rules-based-on-stock-code)
- [Option 2: Create association rules based on product
  name](#option-2-create-association-rules-based-on-product-name)

# Student Details

|                                                                                                                                                                                                                                   |                                                              |     |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|-----|
| **Student ID Numbers and Names of Group Members** \| \| \| 1. 134780 - C - Trevor Okinda \| \| \| \| 2. 132840 - C - Sheila Wangui \| \| \| \| 3. 131749 - C - Teresia Nungari \| \| \| 4. 135203 - C - Tom Arnold \| \| \| \| \| |                                                              |     |
| **GitHub Classroom Group Name**                                                                                                                                                                                                   | Lakers                                                       |     |
| **Course Code**                                                                                                                                                                                                                   | BBT4206                                                      |     |
| **Course Name**                                                                                                                                                                                                                   | Business Intelligence II                                     |     |
| **Program**                                                                                                                                                                                                                       | Bachelor of Business Information Technology                  |     |
| **Semester Duration**                                                                                                                                                                                                             | 21<sup>st</sup> August 2023 to 28<sup>th</sup> November 2023 |     |

# Setup Chunk

**Note:** the following KnitR options have been set as the global
defaults: <BR>
`knitr::opts_chunk$set(echo = TRUE, warning = FALSE, eval = TRUE, collapse = FALSE, tidy = TRUE)`.

More KnitR options are documented here
<https://bookdown.org/yihui/rmarkdown-cookbook/chunk-options.html> and
here <https://yihui.org/knitr/options/>.

# STEP 1. Install and Load the Required Packages —-

## arules —-

if (require(“arules”)) { require(“arules”) } else {
install.packages(“arules”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## arulesViz —-

if (require(“arulesViz”)) { require(“arulesViz”) } else {
install.packages(“arulesViz”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## tidyverse —-

if (require(“tidyverse”)) { require(“tidyverse”) } else {
install.packages(“tidyverse”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## readxl —-

if (require(“readxl”)) { require(“readxl”) } else {
install.packages(“readxl”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## knitr —-

if (require(“knitr”)) { require(“knitr”) } else {
install.packages(“knitr”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## ggplot2 —-

if (require(“ggplot2”)) { require(“ggplot2”) } else {
install.packages(“ggplot2”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## lubridate —-

if (require(“lubridate”)) { require(“lubridate”) } else {
install.packages(“lubridate”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## plyr —-

if (require(“plyr”)) { require(“plyr”) } else { install.packages(“plyr”,
dependencies = TRUE, repos = “<https://cloud.r-project.org>”) }

## dplyr —-

if (require(“dplyr”)) { require(“dplyr”) } else {
install.packages(“dplyr”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## naniar —-

if (require(“naniar”)) { require(“naniar”) } else {
install.packages(“naniar”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## RColorBrewer —-

if (require(“RColorBrewer”)) { require(“RColorBrewer”) } else {
install.packages(“RColorBrewer”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

#### 7a - Algorithm Selection for Classification and Regression

# A. Linear Algorithms

## 1. Linear Regression

\#Load dataset

library(ggplot2) data(“diamonds”)

library(caret)

# Define an 80:20 train:test data split of the diamonds dataset

set.seed(123) \# For reproducibility train_index \<-
createDataPartition(diamonds\$price, p = 0.8, list = FALSE)
diamonds_train \<- diamonds\[train_index, \] diamonds_test \<-
diamonds\[-train_index, \]

# Train the linear regression model

diamonds_model_lm \<- lm(price ~ ., diamonds_train)

# Display the model’s details

summary(diamonds_model_lm)

# Make predictions on the test set

predictions \<- predict(diamonds_model_lm, newdata = diamonds_test)

# SSR

ssr \<- sum((diamonds_test\$price - predictions)^2) print(paste(“SSR =”,
sprintf(ssr, fmt = “%#.4f”)))

# SST

sst \<- sum((diamonds_test$price - mean(diamonds_test$price))^2)
print(paste(“SST =”, sprintf(sst, fmt = “%#.4f”)))

# R Squared

r_squared \<- 1 - (ssr / sst) print(paste(“R Squared =”,
sprintf(r_squared, fmt = “%#.4f”)))

# MAE

absolute_errors \<- abs(predictions - diamonds_test\$price) mae \<-
mean(absolute_errors) print(paste(“MAE =”, sprintf(mae, fmt = “%#.4f”)))

\##2. Logistic Regression library(datasets)

# Load the Sonar dataset from the mlbench package

library(mlbench) data(Sonar)

# Define a 70:30 train:test data split

set.seed(123) \# For reproducibility train_index \<-
createDataPartition(Sonar\$Class, p = 0.7, list = FALSE) sonar_train \<-
Sonar\[train_index, \] sonar_test \<- Sonar\[-train_index, \]

# Define train control with 5-fold cross-validation

train_control \<- trainControl(method = “cv”, number = 5)

# Train the logistic regression model

set.seed(7) sonar_caret_model_logistic \<- train(Class ~ ., data =
sonar_train, method = “glm”, metric = “Accuracy”, preProcess =
c(“center”, “scale”), trControl = train_control)

# Display the model’s details

print(sonar_caret_model_logistic)

# Make predictions

predictions \<- predict(sonar_caret_model_logistic, sonar_test\[,
-ncol(sonar_test)\]) \# Exclude the last column which is the target
variable

# Display the model’s evaluation metrics

confusion_matrix \<- confusionMatrix(predictions, sonar_test\$Class)
print(confusion_matrix)

# Plot the confusion matrix

fourfoldplot(as.table(confusion_matrix\$table), color = c(“grey”,
“lightblue”), main = “Confusion Matrix”)

## 3. Linear Discriminant Analysis

\#Load dataset data(Sonar)

# Define a 70:30 train:test data split for LDA

set.seed(123) \# For reproducibility train_index \<-
createDataPartition(Sonar\$Class, p = 0.7, list = FALSE) sonar_train \<-
Sonar\[train_index, \] sonar_test \<- Sonar\[-train_index, \]

# Train the LDA model

library(MASS) \# Ensure MASS library is loaded for lda function
sonar_model_lda \<- lda(Class ~ ., data = sonar_train)

# Display the model’s details

print(sonar_model_lda)

# Make predictions using the LDA model on the test set

predictions \<- predict(sonar_model_lda, newdata = sonar_test)\$class

# Display the evaluation metrics (confusion matrix)

table(predictions, sonar_test\$Class)

## 4a Regularized Linear Regression Classification

\#Load dataset data(“Sonar”)

library(glmnet) \# Separate predictors (X) and target variable (y) X \<-
as.matrix(Sonar\[, -ncol(Sonar)\]) \# Features (all columns except the
last one) y \<- ifelse(Sonar\$Class == “M”, 1, 0) \# Target variable
with binary encoding

# Train the regularized logistic regression model using glmnet

sonar_model_glm \<- glmnet(X, y, family = “binomial”, alpha = 0.5,
lambda = 0.001)

# Display model’s details

print(sonar_model_glm)

# Make predictions on the Sonar dataset using the trained model

predictions \<- ifelse(predict(sonar_model_glm, newx = X, type =
“response”) \> 0.5, “M”, “R”)

# Display evaluation metrics (confusion matrix)

table(predictions, Sonar\$Class)

### 4b Regularized Linear Regression Regression

\#Load dataset library(ggplot2) data(“diamonds”)

# Define a 70:30 train:test data split of the dataset

set.seed(123) \# For reproducibility train_index \<-
createDataPartition(diamonds\$price, p = 0.7, list = FALSE)
diamonds_train \<- diamonds\[train_index, \] diamonds_test \<-
diamonds\[-train_index, \]

# Train the model using glmnet (regularized linear regression)

train_control \<- trainControl(method = “cv”, number = 5)
diamonds_caret_model_glmnet \<- train(price ~ ., data = diamonds_train,
method = “glmnet”, metric = “RMSE”, preProcess = c(“center”, “scale”),
trControl = train_control) \# Display the model’s details
print(diamonds_caret_model_glmnet)

# Make predictions on the test set

predictions \<- predict(diamonds_caret_model_glmnet, newdata =
diamonds_test)

# Calculate evaluation metrics

rmse \<- sqrt(mean((diamonds_test\$price - predictions)^2))
print(paste(“RMSE =”, sprintf(rmse, fmt = “%#.4f”)))

ssr \<- sum((diamonds_test\$price - predictions)^2) print(paste(“SSR =”,
sprintf(ssr, fmt = “%#.4f”)))

sst \<- sum((diamonds_test$price - mean(diamonds_test$price))^2)
print(paste(“SST =”, sprintf(sst, fmt = “%#.4f”)))

r_squared \<- 1 - (ssr / sst) print(paste(“R Squared =”,
sprintf(r_squared, fmt = “%#.4f”)))

absolute_errors \<- abs(predictions - diamonds_test\$price) mae \<-
mean(absolute_errors) print(paste(“MAE =”, sprintf(mae, fmt = “%#.4f”)))

# B. Non-Linear Algorithms

## 1. Classification and Regression Trees

\#1.a. Decision tree for a classification problem with caret

\#Load dataset library(mlbench) data(“Sonar”)

# Define a 70:30 train:test data split of the dataset

set.seed(123) \# For reproducibility train_index \<-
createDataPartition(Sonar\$Class, p = 0.7, list = FALSE) sonar_train \<-
Sonar\[train_index, \] sonar_test \<- Sonar\[-train_index, \]

# Train the model using the rpart method (decision tree)

train_control \<- trainControl(method = “cv”, number = 5)
sonar_caret_model_rpart \<- train(Class ~ ., data = sonar_train, method
= “rpart”, metric = “Accuracy”, trControl = train_control)

# Display the model’s details

print(sonar_caret_model_rpart)

# Make predictions on the test set

predictions \<- predict(sonar_caret_model_rpart, newdata = sonar_test)

# Display the model’s evaluation metrics (confusion matrix)

confusion_matrix \<- confusionMatrix(predictions, sonar_test\$Class)
print(confusion_matrix)

# Plot the confusion matrix

fourfoldplot(as.table(confusion_matrix\$table), color = c(“grey”,
“lightblue”), main = “Confusion Matrix”)

## 1b Decision tree for a regression problem with CARET

# Load the mtcars dataset

data(“mtcars”)

# Split the mtcars dataset into training and testing sets (70:30 split)

set.seed(123) \# For reproducibility train_index \<-
createDataPartition(mtcars\$mpg, p = 0.7, list = FALSE) mtcars_train \<-
mtcars\[train_index, \] mtcars_test \<- mtcars\[-train_index, \]

# Train the model with decision tree using the rpart method

train_control \<- trainControl(method = “repeatedcv”, number = 5,
repeats = 3) model_cart \<- train(mpg ~ ., data = mtcars, method =
“rpart”, metric = “RMSE”, trControl = train_control)

# Display the model’s details

print(model_cart)

\#Make predictions predictions \<- predict(model_cart, mtcars_test)

# Calculate evaluation metrics

rmse \<- sqrt(mean((mtcars_test\$mpg - predictions)^2))
print(paste(“RMSE =”, sprintf(rmse, fmt = “%#.4f”)))

ssr \<- sum((mtcars_test\$mpg - predictions)^2) print(paste(“SSR =”,
sprintf(ssr, fmt = “%#.4f”)))

sst \<- sum((mtcars_test$mpg - mean(mtcars_test$mpg))^2)
print(paste(“SST =”, sprintf(sst, fmt = “%#.4f”)))

r_squared \<- 1 - (ssr / sst) print(paste(“R Squared =”,
sprintf(r_squared, fmt = “%#.4f”)))

absolute_errors \<- abs(predictions - mtcars_test\$mpg) mae \<-
mean(absolute_errors) print(paste(“MAE =”, sprintf(mae, fmt = “%#.4f”)))

\##2a. Naïve Bayes Classifier for a Classification Problem with CARET
\## Load and split the dataset library(mlbench) data(“Sonar”)

# Define a 70:30 train:test data split of the dataset

set.seed(123) \# For reproducibility train_index \<-
createDataPartition(Sonar\$Class, p = 0.7, list = FALSE) sonar_train \<-
Sonar\[train_index, \] sonar_test \<- Sonar\[-train_index, \]

# Train the Naive Bayes model using 5-fold cross-validation

train_control \<- trainControl(method = “cv”, number = 5) model_nb \<-
train(Class ~ ., data = sonar_train, method = “nb”, metric = “Accuracy”,
trControl = train_control)

# Display the model’s details

print(model_nb)

# Make predictions on the test set

predictions \<- predict(model_nb, newdata = sonar_test)

# Display the model’s evaluation metrics (confusion matrix)

confusion_matrix \<- confusionMatrix(predictions, sonar_test\$Class)
print(confusion_matrix)

# Plot the confusion matrix

fourfoldplot(as.table(confusion_matrix\$table), color = c(“grey”,
“lightblue”), main = “Confusion Matrix”)

## 3. k-Nearest Neighbours

### 3a kNN for a classification problem with CARET’s train function

\#Load dataset library(mlbench) data(“Sonar”)

# Define a 70:30 train:test data split of the dataset

set.seed(123) \# For reproducibility train_index \<-
createDataPartition(Sonar\$Class, p = 0.7, list = FALSE) sonar_train \<-
Sonar\[train_index, \] sonar_test \<- Sonar\[-train_index, \]

# Train the kNN model with 10-fold cross-validation and data standardization

train_control \<- trainControl(method = “cv”, number = 10) model_knn \<-
train(Class ~ ., data = sonar_train, method = “knn”, metric =
“Accuracy”, preProcess = c(“center”, “scale”), trControl =
train_control)

# Display the model’s details

print(model_knn)

# Make predictions on the test set

predictions \<- predict(model_knn, newdata = sonar_test)

# Display the model’s evaluation metrics (confusion matrix)

confusion_matrix \<- confusionMatrix(predictions, sonar_test\$Class)
print(confusion_matrix)

# Plot the confusion matrix

fourfoldplot(as.table(confusion_matrix\$table), color = c(“grey”,
“lightblue”), main = “Confusion Matrix”)

\###3b kNN for a regression problem with CARET’s train function \#Load
dataset

data(“airquality”)

# Replace missing values with mean for the target variable “Ozone”

airquality$Ozone[is.na(airquality$Ozone)\] \<- mean(airquality\$Ozone,
na.rm = TRUE)

# Define a target variable and predictors for regression

target_var \<- “Ozone” \# Target variable predictors \<- c(“Solar.R”,
“Wind”, “Temp”) \# Predictors for the regression

# Split the data into train and test sets

set.seed(123) trainIndex \<-
createDataPartition(airquality\[\[target_var\]\], p = 0.8, list = FALSE)
airq_train \<- airquality\[trainIndex, \] airq_test \<-
airquality\[-trainIndex, \]

# Apply the 5-fold cross-validation resampling method and standardize data

train_control \<- trainControl(method = “cv”, number = 5) model_knn \<-
train(airq_train\[predictors\], airq_train\[\[target_var\]\], method =
“knn”, metric = “RMSE”, preProcess = c(“center”, “scale”), trControl =
train_control)

# Display the model’s details

print(model_knn)

# Make predictions on the test set

predictions \<- predict(model_knn, newdata = airq_test\[predictors\])

# Calculate evaluation metrics for the model

rmse \<- sqrt(mean((airq_test\[\[target_var\]\] - predictions)^2)) ssr
\<- sum((airq_test\[\[target_var\]\] - predictions)^2) sst \<-
sum((airq_test\[\[target_var\]\] - mean(airq_test\[\[target_var\]\]))^2)
r_squared \<- 1 - (ssr / sst) absolute_errors \<- abs(predictions -
airq_test\[\[target_var\]\]) mae \<- mean(absolute_errors)

# Display the model’s evaluation metrics

print(paste(“RMSE =”, sprintf(rmse, fmt = “%#.4f”))) print(paste(“SSR
=”, sprintf(ssr, fmt = “%#.4f”))) print(paste(“SST =”, sprintf(sst, fmt
= “%#.4f”))) print(paste(“R Squared =”, sprintf(r_squared, fmt =
“%#.4f”))) print(paste(“MAE =”, sprintf(mae, fmt = “%#.4f”)))

## 4. Support Vector Machine

### 4a SVM Classifier for a classification problem with CARET

# Load the necessary libraries

library(caret) \# Load the iris dataset (it’s a built-in dataset)
data(iris)

# Define target variable and predictors for the iris dataset

target_var \<- “Species” \# Target variable predictors \<-
c(“Sepal.Length”, “Sepal.Width”, “Petal.Length”, “Petal.Width”) \#
Predictors

# Define a 70:30 train:test data split of the iris dataset

train_index \<- createDataPartition(iris\[\[target_var\]\], p = 0.7,
list = FALSE) iris_train \<- iris\[train_index, \] iris_test \<-
iris\[-train_index, \]

# Train the LSVM model with radial kernel

set.seed(7) train_control \<- trainControl(method = “cv”, number = 5)
model_svm_radial \<- train(iris_train\[, predictors\],
iris_train\[\[target_var\]\], method = “svmRadial”, metric = “Accuracy”,
trControl = train_control)

# Display the model’s details

print(model_svm_radial)

# Make predictions

predictions \<- predict(model_svm_radial, iris_test\[, predictors\])

# Display the model’s evaluation metrics - Confusion Matrix

confusion_matrix \<- confusionMatrix(predictions,
iris_test\[\[target_var\]\]) print(confusion_matrix)

# Visualize the confusion matrix using a heat map

caret::plot(confusion_matrix\$table, col =
colorRampPalette(c(“lightblue”, “grey”))(20))

### 4b SVM classifier for a regression problem with CARET

# Load the ‘swiss’ dataset

data(“swiss”)

# Define predictors and target variable

predictors \<- names(swiss)\[1:5\] \# Selecting the first 5 columns as
predictors target_var \<- “Fertility” \# Target variable for regression

# Define an 80:20 train:test data split of the dataset

set.seed(123) train_index \<-
createDataPartition(swiss\[\[target_var\]\], p = 0.8, list = FALSE)
swiss_train \<- swiss\[train_index, \] swiss_test \<-
swiss\[-train_index, \]

# Train the model using SVM regression

train_control \<- trainControl(method = “cv”, number = 5) model_svm_reg
\<- train(swiss_train\[, predictors\], swiss_train\[\[target_var\]\],
method = “svmLinear”, trControl = train_control)

# Display the model’s details

print(model_svm_reg)

# Make predictions

predictions \<- predict(model_svm_reg, newdata = swiss_test\[,
predictors\])

# Calculate evaluation metrics

# RMSE

rmse \<- sqrt(mean((swiss_test\[\[target_var\]\] - predictions)^2))
print(paste(“RMSE =”, sprintf(rmse, fmt = “%#.4f”)))

# SSR

ssr \<- sum((swiss_test\[\[target_var\]\] - predictions)^2)
print(paste(“SSR =”, sprintf(ssr, fmt = “%#.4f”)))

# SST

sst \<- sum((swiss_test\[\[target_var\]\] -
mean(swiss_test\[\[target_var\]\]))^2) print(paste(“SST =”, sprintf(sst,
fmt = “%#.4f”)))

# R Squared

r_squared \<- 1 - (ssr / sst) print(paste(“R Squared =”,
sprintf(r_squared, fmt = “%#.4f”)))

# MAE

absolute_errors \<- abs(predictions - swiss_test\[\[target_var\]\]) mae
\<- mean(absolute_errors) print(paste(“MAE =”, sprintf(mae, fmt =
“%#.4f”)))

# 7b Algorithm Selection for Clustering

\#Load dataset data(iris) \# Display structure of Iris dataset str(iris)
\# Get dimensions of Iris dataset dim(iris) \# Display the first few
rows of the Iris dataset head(iris) \# Summary statistics of Iris
dataset summary(iris)

# Check for missing values (There are no missing values in the iris dataset)

any_na(iris) n_miss(iris) prop_miss(iris) miss_var_summary(iris)
gg_miss_var(iris)

# Compute the correlations between variables

cor_matrix \<- cor(iris\[, -5\]) \# Calculate the correlation matrix
(excluding the ‘Species’ column)

# Basic Table (View the correlation matrix)

View(cor_matrix)

# Basic Plot (Correlation matrix using corrplot package)

corrplot::corrplot(cor_matrix, method = “square”)

# Fancy Plot using ggplot2 (Heatmap visualization of the correlation matrix)

library(ggplot2) library(reshape2)

cor_matrix_melted \<- melt(cor_matrix)

p \<- ggplot(cor_matrix_melted, aes(Var1, Var2, fill = value)) +
geom_tile() + geom_text(aes(label = round(value, 2)), size = 4) +
theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust =
1))

print(p)

# Scatter plot comparing Sepal Length vs. Sepal Width with Species differentiation

library(ggplot2) ggplot(iris, aes(Sepal.Length, Sepal.Width, color =
Species, shape = Species)) + geom_point(alpha = 0.5) + xlab(“Sepal
Length”) + ylab(“Sepal Width”)

# Scatter plot comparing Petal Length vs. Petal Width with Species differentiation

ggplot(iris, aes(Petal.Length, Petal.Width, color = Species, shape =
Species)) + geom_point(alpha = 0.5) + xlab(“Petal Length”) + ylab(“Petal
Width”)

# Scatter plot comparing Sepal Length vs. Petal Length with Species differentiation

ggplot(iris, aes(Sepal.Length, Petal.Length, color = Species, shape =
Species)) + geom_point(alpha = 0.5) + xlab(“Sepal Length”) + ylab(“Petal
Length”)

# Scatter plot comparing Sepal Width vs. Petal Width with Species differentiation

ggplot(iris, aes(Sepal.Width, Petal.Width, color = Species, shape =
Species)) + geom_point(alpha = 0.5) + xlab(“Sepal Width”) + ylab(“Petal
Width”)

# Scatter plot comparing Sepal Length vs. Petal Width with Species differentiation

ggplot(iris, aes(Sepal.Length, Petal.Width, color = Species, shape =
Species)) + geom_point(alpha = 0.5) + xlab(“Sepal Length”) + ylab(“Petal
Width”)

\#Transform the data summary(iris) model_of_the_transform \<-
preProcess(iris, method = c(“scale”, “center”))
print(model_of_the_transform) iris_std \<-
predict(model_of_the_transform, iris) summary(iris_std)
sapply(iris_std\[, sapply(iris_std, is.numeric)\], sd)

## Select the features to use to create the clusters

iris_vars \<- iris\[, c(“Sepal.Length”, “Sepal.Width”, “Petal.Length”,
“Petal.Width”)\]

# Create the clusters using the K-Means Clustering Algorithm

set.seed(7) iris_kmeans \<- kmeans(iris_vars, centers = 3, nstart = 20)

# Define the maximum number of clusters to investigate

n_clusters \<- 8

# Initialize the total within sum of squares error (wss)

wss \<- numeric(n_clusters)

set.seed(7)

# Investigate 1 to n possible clusters

for (i in 1:n_clusters) { \# Apply the K Means clustering algorithm for
each potential cluster count kmeans_cluster \<- kmeans(iris_vars,
centers = i, nstart = 20) \# Store the within-cluster sum of squares
wss\[i\] \<- kmeans_cluster\$tot.withinss }

\#Scree plot wss_df \<- data.frame(clusters = 1:n_clusters, wss = wss)

scree_plot \<- ggplot(wss_df, aes(x = clusters, y = wss, group = 1)) +
geom_point(size = 4) + geom_line() + scale_x_continuous(breaks = c(2, 4,
6, 8)) + xlab(“Number of Clusters”)

scree_plot

scree_plot + geom_hline( yintercept = wss, linetype = “dashed”, col =
c(rep(“\#000000”, 5), “\#FF0000”, rep(“\#000000”, 2)) ) \# The plateau
is reached at 6 clusters. \# We therefore create the final cluster with
6 clusters \# (not the initial 3 used at the beginning of this STEP.) k
\<- 6 set.seed(7) \# Build model with k clusters: kmeans_cluster
kmeans_cluster \<- kmeans(iris\[, -5\], centers = k, nstart = 20)

# Add the cluster number as a label for each observation

iris$cluster_id <- factor(kmeans_cluster$cluster)

## View the results by plotting scatter plots with the labelled cluster

ggplot(iris, aes(Petal.Length, Petal.Width, color = cluster_id)) +
geom_point(alpha = 0.5) + xlab(“Petal Length”) + ylab(“Petal Width”)

ggplot(iris, aes(Sepal.Length, Sepal.Width, color = cluster_id)) +
geom_point(alpha = 0.5) + xlab(“Sepal Length”) + ylab(“Sepal Width”)

\###7c: Algorithm Selection for Association Rule Learning \# Load the
arules package library(arules)

# Load the Groceries dataset

data(“Groceries”)

# View the structure of the Groceries dataset

str(Groceries)

# Show the first few transactions

inspect(head(Groceries))

# Create a transactions object

groceries_transactions \<- as(Groceries, “transactions”)

# Get item frequency

item_freq \<- itemFrequency(groceries_transactions)

# Sort item frequency in descending order

sorted_freq \<- sort(item_freq, decreasing = TRUE)

# Plotting the top 10 absolute item frequencies

itemFrequencyPlot(groceries_transactions, topN = 10, type = “absolute”,
col = brewer.pal(8, “Pastel2”), main = “Absolute Item Frequency Plot”,
horiz = TRUE)

# Plotting the top 10 relative item frequencies

itemFrequencyPlot(groceries_transactions, topN = 10, type = “relative”,
col = brewer.pal(8, “Pastel2”), main = “Relative Item Frequency Plot”,
horiz = TRUE)

# Print the summary and inspect the association rules (Option 1)

install.packages(“arulesViz”) \# Install the arulesViz package
library(arulesViz) \# Load the arulesViz package
summary(association_rules_stock_code)
inspect(association_rules_stock_code)
inspect(head(association_rules_stock_code, 10)) \# Viewing the top 10
rules plot(association_rules_stock_code)

# Remove redundant rules

redundant_rules \<-
which(colSums(is.subset(association_rules_stock_code,
association_rules_stock_code)) \> 1) length(redundant_rules)
association_rules_non_redundant \<-
association_rules_stock_code\[-redundant_rules\]

# Display summary and inspect non-redundant rules

summary(association_rules_non_redundant)
inspect(association_rules_non_redundant)

# Write the non-redundant rules to a CSV file

write(association_rules_non_redundant, file =
“rules/association_rules_based_on_stock_code.csv”)

# Create a transactions object

groceries_transactions \<- as(Groceries, “transactions”)

# Set the minimum support, confidence, and maxlen

min_support \<- 0.01 min_confidence \<- 0.8 maxlen \<- 10

# Option 1: Create association rules based on stock code

association_rules_stock_code \<- apriori(groceries_transactions,
parameter = list(support = min_support, confidence = min_confidence,
maxlen = maxlen))

# Option 2: Create association rules based on product name

association_rules_product_name \<- apriori(groceries_transactions,
parameter = list(support = min_support, confidence = min_confidence,
maxlen = maxlen))
