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

