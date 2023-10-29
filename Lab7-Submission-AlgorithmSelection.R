# A. Linear Algorithms
## 1. Linear Regression

#Load dataset

library(ggplot2)
data("diamonds")

library(caret)

# Define an 80:20 train:test data split of the diamonds dataset
set.seed(123)  # For reproducibility
train_index <- createDataPartition(diamonds$price,
                                   p = 0.8,
                                   list = FALSE)
diamonds_train <- diamonds[train_index, ]
diamonds_test <- diamonds[-train_index, ]

# Train the linear regression model
diamonds_model_lm <- lm(price ~ ., diamonds_train)

# Display the model's details
summary(diamonds_model_lm)

# Make predictions on the test set
predictions <- predict(diamonds_model_lm, newdata = diamonds_test)

# SSR
ssr <- sum((diamonds_test$price - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

# SST
sst <- sum((diamonds_test$price - mean(diamonds_test$price))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

# R Squared
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

# MAE
absolute_errors <- abs(predictions - diamonds_test$price)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))

##2. Logistic Regression
library(datasets)

# Load the Sonar dataset from the mlbench package
library(mlbench)
data(Sonar)

# Define a 70:30 train:test data split
set.seed(123)  # For reproducibility
train_index <- createDataPartition(Sonar$Class,
                                   p = 0.7,
                                   list = FALSE)
sonar_train <- Sonar[train_index, ]
sonar_test <- Sonar[-train_index, ]

# Define train control with 5-fold cross-validation
train_control <- trainControl(method = "cv", number = 5)

# Train the logistic regression model
set.seed(7)
sonar_caret_model_logistic <- train(Class ~ ., data = sonar_train,
                                    method = "glm", metric = "Accuracy",
                                    preProcess = c("center", "scale"),
                                    trControl = train_control)

# Display the model's details
print(sonar_caret_model_logistic)

# Make predictions
predictions <- predict(sonar_caret_model_logistic,
                       sonar_test[, -ncol(sonar_test)])  # Exclude the last column which is the target variable

# Display the model's evaluation metrics
confusion_matrix <- confusionMatrix(predictions, sonar_test$Class)
print(confusion_matrix)

# Plot the confusion matrix
fourfoldplot(as.table(confusion_matrix$table), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

## 3. Linear Discriminant Analysis
#Load dataset
data(Sonar)

# Define a 70:30 train:test data split for LDA
set.seed(123)  # For reproducibility
train_index <- createDataPartition(Sonar$Class,
                                   p = 0.7,
                                   list = FALSE)
sonar_train <- Sonar[train_index, ]
sonar_test <- Sonar[-train_index, ]

# Train the LDA model
library(MASS)  # Ensure MASS library is loaded for lda function
sonar_model_lda <- lda(Class ~ ., data = sonar_train)

# Display the model's details
print(sonar_model_lda)

# Make predictions using the LDA model on the test set
predictions <- predict(sonar_model_lda, newdata = sonar_test)$class

# Display the evaluation metrics (confusion matrix)
table(predictions, sonar_test$Class)

## 4a Regularized Linear Regression Classification
#Load dataset
data("Sonar")

library(glmnet)
# Separate predictors (X) and target variable (y)
X <- as.matrix(Sonar[, -ncol(Sonar)])  # Features (all columns except the last one)
y <- ifelse(Sonar$Class == "M", 1, 0)  # Target variable with binary encoding

# Train the regularized logistic regression model using glmnet
sonar_model_glm <- glmnet(X, y, family = "binomial", alpha = 0.5, lambda = 0.001)

# Display model's details
print(sonar_model_glm)

# Make predictions on the Sonar dataset using the trained model
predictions <- ifelse(predict(sonar_model_glm, newx = X, type = "response") > 0.5, "M", "R")

# Display evaluation metrics (confusion matrix)
table(predictions, Sonar$Class)

### 4b Regularized Linear Regression Regression
#Load dataset
library(ggplot2)
data("diamonds")

# Define a 70:30 train:test data split of the dataset
set.seed(123)  # For reproducibility
train_index <- createDataPartition(diamonds$price, p = 0.7, list = FALSE)
diamonds_train <- diamonds[train_index, ]
diamonds_test <- diamonds[-train_index, ]

# Train the model using glmnet (regularized linear regression)
train_control <- trainControl(method = "cv", number = 5)
diamonds_caret_model_glmnet <- train(price ~ ., data = diamonds_train, method = "glmnet",
                                     metric = "RMSE", preProcess = c("center", "scale"),
                                     trControl = train_control)
# Display the model's details
print(diamonds_caret_model_glmnet)

# Make predictions on the test set
predictions <- predict(diamonds_caret_model_glmnet, newdata = diamonds_test)

# Calculate evaluation metrics
rmse <- sqrt(mean((diamonds_test$price - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

ssr <- sum((diamonds_test$price - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

sst <- sum((diamonds_test$price - mean(diamonds_test$price))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

absolute_errors <- abs(predictions - diamonds_test$price)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))

# B. Non-Linear Algorithms#
## 1.  Classification and Regression Trees
#1.a. Decision tree for a classification problem with caret

#Load dataset
library(mlbench)
data("Sonar")

# Define a 70:30 train:test data split of the dataset
set.seed(123)  # For reproducibility
train_index <- createDataPartition(Sonar$Class, p = 0.7, list = FALSE)
sonar_train <- Sonar[train_index, ]
sonar_test <- Sonar[-train_index, ]

# Train the model using the rpart method (decision tree)
train_control <- trainControl(method = "cv", number = 5)
sonar_caret_model_rpart <- train(Class ~ ., data = sonar_train, method = "rpart",
                                 metric = "Accuracy", trControl = train_control)

# Display the model's details
print(sonar_caret_model_rpart)

# Make predictions on the test set
predictions <- predict(sonar_caret_model_rpart, newdata = sonar_test)

# Display the model's evaluation metrics (confusion matrix)
confusion_matrix <- confusionMatrix(predictions, sonar_test$Class)
print(confusion_matrix)

# Plot the confusion matrix
fourfoldplot(as.table(confusion_matrix$table), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

## 1b Decision tree for a regression problem with CARET
# Load the mtcars dataset
data("mtcars")

# Split the mtcars dataset into training and testing sets (70:30 split)
set.seed(123)  # For reproducibility
train_index <- createDataPartition(mtcars$mpg, p = 0.7, list = FALSE)
mtcars_train <- mtcars[train_index, ]
mtcars_test <- mtcars[-train_index, ]

# Train the model with decision tree using the rpart method
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
model_cart <- train(mpg ~ ., data = mtcars, method = "rpart", metric = "RMSE", trControl = train_control)

# Display the model's details
print(model_cart)

#Make predictions
predictions <- predict(model_cart, mtcars_test)

# Calculate evaluation metrics
rmse <- sqrt(mean((mtcars_test$mpg - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

ssr <- sum((mtcars_test$mpg - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

sst <- sum((mtcars_test$mpg - mean(mtcars_test$mpg))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

absolute_errors <- abs(predictions - mtcars_test$mpg)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))


##2a. Naïve Bayes Classifier for a Classification Problem with CARET
## Load and split the dataset
library(mlbench)
data("Sonar")

# Define a 70:30 train:test data split of the dataset
set.seed(123)  # For reproducibility
train_index <- createDataPartition(Sonar$Class, p = 0.7, list = FALSE)
sonar_train <- Sonar[train_index, ]
sonar_test <- Sonar[-train_index, ]

# Train the Naive Bayes model using 5-fold cross-validation
train_control <- trainControl(method = "cv", number = 5)
model_nb <- train(Class ~ ., data = sonar_train, method = "nb", metric = "Accuracy", trControl = train_control)

# Display the model's details
print(model_nb)

# Make predictions on the test set
predictions <- predict(model_nb, newdata = sonar_test)

# Display the model's evaluation metrics (confusion matrix)
confusion_matrix <- confusionMatrix(predictions, sonar_test$Class)
print(confusion_matrix)

# Plot the confusion matrix
fourfoldplot(as.table(confusion_matrix$table), color = c("grey", "lightblue"), main = "Confusion Matrix")

## 3.  k-Nearest Neighbours
### 3a kNN for a classification problem with CARET's train function

#Load dataset
library(mlbench)
data("Sonar")

# Define a 70:30 train:test data split of the dataset
set.seed(123)  # For reproducibility
train_index <- createDataPartition(Sonar$Class, p = 0.7, list = FALSE)
sonar_train <- Sonar[train_index, ]
sonar_test <- Sonar[-train_index, ]

# Train the kNN model with 10-fold cross-validation and data standardization
train_control <- trainControl(method = "cv", number = 10)
model_knn <- train(Class ~ ., data = sonar_train, method = "knn", metric = "Accuracy",
                   preProcess = c("center", "scale"), trControl = train_control)

# Display the model's details
print(model_knn)

# Make predictions on the test set
predictions <- predict(model_knn, newdata = sonar_test)

# Display the model's evaluation metrics (confusion matrix)
confusion_matrix <- confusionMatrix(predictions, sonar_test$Class)
print(confusion_matrix)

# Plot the confusion matrix
fourfoldplot(as.table(confusion_matrix$table), color = c("grey", "lightblue"), main = "Confusion Matrix")

###3b kNN for a regression problem with CARET's train function
#Load dataset

data("airquality")

# Replace missing values with mean for the target variable "Ozone"
airquality$Ozone[is.na(airquality$Ozone)] <- mean(airquality$Ozone, na.rm = TRUE)

# Define a target variable and predictors for regression
target_var <- "Ozone"  # Target variable
predictors <- c("Solar.R", "Wind", "Temp")  # Predictors for the regression

# Split the data into train and test sets
set.seed(123)
trainIndex <- createDataPartition(airquality[[target_var]], p = 0.8, list = FALSE)
airq_train <- airquality[trainIndex, ]
airq_test <- airquality[-trainIndex, ]

# Apply the 5-fold cross-validation resampling method and standardize data
train_control <- trainControl(method = "cv", number = 5)
model_knn <- train(airq_train[predictors], airq_train[[target_var]], method = "knn",
                   metric = "RMSE", preProcess = c("center", "scale"), trControl = train_control)

# Display the model's details
print(model_knn)

# Make predictions on the test set
predictions <- predict(model_knn, newdata = airq_test[predictors])

# Calculate evaluation metrics for the model
rmse <- sqrt(mean((airq_test[[target_var]] - predictions)^2))
ssr <- sum((airq_test[[target_var]] - predictions)^2)
sst <- sum((airq_test[[target_var]] - mean(airq_test[[target_var]]))^2)
r_squared <- 1 - (ssr / sst)
absolute_errors <- abs(predictions - airq_test[[target_var]])
mae <- mean(absolute_errors)

# Display the model's evaluation metrics
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))

## 4.  Support Vector Machine
### 4a SVM Classifier for a classification problem with CARET

# Load the necessary libraries
library(caret)
# Load the iris dataset (it's a built-in dataset)
data(iris)

# Define target variable and predictors for the iris dataset
target_var <- "Species"  # Target variable
predictors <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")  # Predictors

# Define a 70:30 train:test data split of the iris dataset
train_index <- createDataPartition(iris[[target_var]], p = 0.7, list = FALSE)
iris_train <- iris[train_index, ]
iris_test <- iris[-train_index, ]

# Train the LSVM model with radial kernel
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
model_svm_radial <- train(iris_train[, predictors], iris_train[[target_var]],
                          method = "svmRadial", metric = "Accuracy", trControl = train_control)

# Display the model's details
print(model_svm_radial)

# Make predictions
predictions <- predict(model_svm_radial, iris_test[, predictors])

# Display the model's evaluation metrics - Confusion Matrix
confusion_matrix <- confusionMatrix(predictions, iris_test[[target_var]])
print(confusion_matrix)

# Visualize the confusion matrix using a heat map
caret::plot(confusion_matrix$table, col = colorRampPalette(c("lightblue", "grey"))(20))

### 4b SVM classifier for a regression problem with CARET
# Load the 'swiss' dataset
data("swiss")

# Define predictors and target variable
predictors <- names(swiss)[1:5]  # Selecting the first 5 columns as predictors
target_var <- "Fertility"  # Target variable for regression

# Define an 80:20 train:test data split of the dataset
set.seed(123)
train_index <- createDataPartition(swiss[[target_var]], p = 0.8, list = FALSE)
swiss_train <- swiss[train_index, ]
swiss_test <- swiss[-train_index, ]

# Train the model using SVM regression
train_control <- trainControl(method = "cv", number = 5)
model_svm_reg <- train(swiss_train[, predictors], swiss_train[[target_var]], 
                       method = "svmLinear", trControl = train_control)

# Display the model's details
print(model_svm_reg)

# Make predictions
predictions <- predict(model_svm_reg, newdata = swiss_test[, predictors])

# Calculate evaluation metrics
# RMSE
rmse <- sqrt(mean((swiss_test[[target_var]] - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

# SSR
ssr <- sum((swiss_test[[target_var]] - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

# SST
sst <- sum((swiss_test[[target_var]] - mean(swiss_test[[target_var]]))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

# R Squared
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

# MAE
absolute_errors <- abs(predictions - swiss_test[[target_var]])
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))

