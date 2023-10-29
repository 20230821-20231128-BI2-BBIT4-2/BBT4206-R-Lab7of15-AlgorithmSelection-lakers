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
