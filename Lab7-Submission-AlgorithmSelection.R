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

