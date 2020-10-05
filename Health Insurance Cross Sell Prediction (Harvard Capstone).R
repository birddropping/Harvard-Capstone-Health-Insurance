#Health Insurance Cross Sell Prediction 

#This project is adapted from a challenge on Kaggle (https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction). The client is an Insurance company that has provided Health Insurance to its customers and is looking to build a model to predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company.

#An insurance policy is an arrangement by which a company undertakes to provide a guarantee of compensation for specified loss, damage, illness, or death in return for the payment of a specified premium. A premium is a sum of money that the customer needs to pay regularly to an insurance company for this guarantee.

#Just like medical insurance, there is vehicle insurance where every year customer needs to pay a premium of certain amount to insurance provider company so that in case of unfortunate accident by the vehicle, the insurance provider company will provide a compensation (called ‘sum assured’) to the customer.

#Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimise its business model and revenue.

#In this project, we aim to build supervised machine learningmodel to predict if a customer would be interested in Vehicle insurance, based on information about demographics (gender, age, region code type), Vehicles (Vehicle Age, Damage), Policy (Premium, sourcing channel) etc.

#Dataset found from https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction?select=train.csv

################################

### REQUIRED LIBRARIES

################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(RCurl)) install.packages("RCurl", repos = "http://cran.us.r-project.org")
if(!require(summarytools)) install.packages("summarytools", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(RCurl)
library(summarytools)
library(RColorBrewer)
library(ggplot2)
library(caret)
library(randomForest)
library(corrplot)
library(pROC)


################################

### LOADING DATA

################################


# This dataset comes with both test and training datasets. However, as the files were taken from a kaggle competition, the test dataset did not contain the dependent variable. Hence, I split the training dataset into my own testing and training sets. 

# The total number of observations in the set was 381,109, which was quite big and difficult for my computer to process. I reduced the total dataset to 100000 and split the dataset 80:20 such that caret training models could be used.

data_source <- getURL("https://raw.githubusercontent.com/birddropping/Harvard-Capstone-Health-Insurance/main/train.csv")
data <- read.csv(text = data_source)

data <- data[1:150000,]


################################

# DATA CLEANING

################################

# Converting all inputs to numeric values

# Gender #
data$Gender[data$Gender == "Male"] <- 1
data$Gender[data$Gender == "Female"] <- 2

# Vehicle Age #
data$Vehicle_Age[data$Vehicle_Age == "< 1 Year"] <- 1
data$Vehicle_Age[data$Vehicle_Age == "1-2 Year"] <- 2
data$Vehicle_Age[data$Vehicle_Age == "> 2 Years"] <- 3

# Vehicle Damage #
data$Vehicle_Damage[data$Vehicle_Damage == "No"] <- 0
data$Vehicle_Damage[data$Vehicle_Damage == "Yes"] <- 1

# Converting Responses #
data$Response <- as.numeric(data$Response)
data$Response[data$Response == 0] <- "N"
data$Response[data$Response == 1] <- "Y"

# Converting all columns to factor type

data <- lapply(data, as.factor)
data <- as.data.frame(data)

# Change Age, Annual Premiums and Vintage to numeric as they are continuous data

data$Age <- as.numeric(data$Age)
data$Annual_Premium <- as.numeric(data$Annual_Premium)
data$Region_Code <- as.numeric(data$Region_Code)
data$Policy_Sales_Channel <- as.numeric(data$Policy_Sales_Channel)
data$Vintage <- as.numeric(data$Vintage)


################################

# DATA PARTITIONING

################################


set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(data$Response, times = 1, p = 0.2, list = FALSE)

validation <- data[test_index, ]
edx <- data[-test_index, ]

validation <- validation %>% 
  semi_join(edx, by = "Gender") %>%
  semi_join(edx, by = "Driving_License") %>%
  semi_join(edx, by = "Region_Code") %>%
  semi_join(edx, by = "Previously_Insured") %>%
  semi_join(edx, by = "Vehicle_Damage") %>%
  semi_join(edx, by = "Policy_Sales_Channel")


set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(edx$Response, times = 1, p = 0.2, list = FALSE)

edx_test <- edx[test_index, ]

edx_train <- edx[-test_index, ]

edx_test <- edx_test %>% 
  semi_join(edx_train, by = "Gender") %>%
  semi_join(edx_train, by = "Driving_License") %>%
  semi_join(edx_train, by = "Region_Code") %>%
  semi_join(edx_train, by = "Previously_Insured") %>%
  semi_join(edx_train, by = "Vehicle_Damage") %>%
  semi_join(edx_train, by = "Policy_Sales_Channel")

rm(data_source, test_index, data)


################################

# DATA EXPLORATION

################################


### Correlation matrix of various factors

train_cor <- edx_train %>% mutate_if(is.factor, as.numeric)

cor(train_cor, use="pairwise.complete.obs", method = c("pearson", "kendall", "spearman"))  %>% 
  corrplot( method = "pie", type= "upper", outline = T, tl.col = "black",tl.srt=50,col=c("red", "blue"))


### From this correlation plot, we see that Gender, Age, ANnual Premium and the Policy Sales Channel used to attract the customer has a mild correlation with the response of the customer, and Vehicle Age, previous Vehicle Damage and whether they have been previously insured hhas a moderate correlation with the response. We will examine these factors in the next few steps

### Number of people that also take up the vehicle insurance on top of their health insurance

edx_train %>% 
  group_by(Response) %>%
  count()


# Only 13.9% of policy holders were interested to also take up the vehicle insurance

### Distribution of gender ###

edx_train %>% 
  ggplot(aes(x=Gender)) +
  geom_bar(aes(fill=Response)) + 
  geom_text(stat='count', aes(label=..count..), vjust=3) + 
  ggtitle("Distribution of customers by gender")

edx_train %>%
  group_by(Gender, Response) %>% 
  summarise(n = n()) %>%
  mutate(percentage = paste0(round(100 * n/sum(n), 0), "%"))

# Slightly higher percentage of males were correlated with taking up the insurance

### Distribution of Age ###

edx_train %>% 
  ggplot(aes(x=Age)) + 
  geom_density(aes(fill=Response), alpha = 0.5)+ 
  ggtitle("Distribution of customers by age, grouped by response")


### Distribution of policy owners that have been previously insured ###

edx_train %>%
  ggplot(aes(x=Previously_Insured)) +
  geom_bar(aes(fill=Response)) + 
  ggtitle("Number of customers who have been previously insured")

edx_train %>%
  group_by(Previously_Insured, Response) %>%
  summarise(n = n()) %>%
  mutate(percentage = paste0(round(100 * n/sum(n), 0), "%"))

# Most people who were Previously Insured chose not to take up the insurance. However, 23% of people who were not previously insured took up the insurance. 

### Number of individuals from the various regions ###

edx_train %>%
  ggplot(aes(y=Region_Code)) +
  geom_bar(aes(fill = Region_Code)) +
  ggtitle("Total number of customers by region")

region <- edx_train %>%
  group_by(Region_Code, Response) %>%
  summarise(n = n()) %>%
  mutate(percentage = round(100 * n/sum(n), 0)) %>% 
  filter(Response == "Y")

region %>%
  ggplot(aes(x = Region_Code, y = as.numeric(percentage))) +
  geom_point() + 
  ggtitle("Percentage of customers that take up insurance by region code") + 
  labs(y = "Percentage Uptake of Insurance")

region <- edx_train %>%
  group_by(Region_Code, Response) %>%
  summarise(n = n()) %>%
  mutate(percentage = round(100 * n/sum(n), 0)) 

max(region$n)
min(region$n)
### Distribution of length of car ownership ###

edx_train %>%
  ggplot(aes(x=Vehicle_Age)) +
  geom_bar(aes(fill = Response)) +
  geom_text(stat='count', aes(label=..count..), vjust=-0.8)

edx_train %>%
  group_by(Vehicle_Age, Response) %>%
  summarise(n = n()) %>%
  mutate(percentage = round(100 * n/sum(n), 0)) %>%
  filter(Response == "Y")

### Distribution of past vehicle damage ###

edx_train %>%
  ggplot(aes(x=Vehicle_Damage)) + 
  geom_bar(aes(fill=Response))

### Distribution of Annual Premiums ###

edx_train %>%
  ggplot(aes(x=Annual_Premium)) +
  geom_density(aes(fill = Response), alpha = 0.3) 

# Amount of annual premiums paid do not seem to make a significant difference in whether a driver takes up the insurance, although it does seem that drivers paying larger amounts in insurance tend to be more likely to take up the insurance


### Distribution of Policy Sales Channels

edx_train %>% 
  ggplot(aes(as.numeric(Policy_Sales_Channel))) +
  geom_bar(aes(fill = Response), alpha = 0.3) + 
  ggtitle("Distribution of customers by policy sales channel")
  
edx_train %>% 
  group_by(Policy_Sales_Channel, Response) %>%
  summarise(n = n()) %>% 
  mutate(percentage = round(100 * n/sum(n), 0)) %>%
  filter(Response == "Y") %>%
  arrange(desc(percentage)) %>% 
  head()

edx_train %>% 
  group_by(Policy_Sales_Channel, Response) %>%
  summarise(n = n()) %>% 
  mutate(percentage = round(100 * n/sum(n), 0)) %>%
  filter(Response == "Y") %>%
  arrange(percentage) %>% 
  head()


################################

# MODEL BUILDING

################################

###
### Naive Bayes ###
###

# Set control parameters for cross validation
control_bayes <- trainControl(method = "cv", number = 5, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary)

# Train fit using caret package. Metric set to ROC
fit_bayes <- train(Response ~ Gender + Age + Previously_Insured + Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel, method = "naive_bayes", data = edx_train, trControl = control_bayes, metric = "ROC")


# Selecting best hyperparameters for Naive Bayes model to plot the ROC curve
index_bayes <- fit_bayes$pred$usekernel == as.logical(fit_bayes$bestTune[2])

roc_bayes <- roc(fit_bayes$pred$obs[index_bayes],
                 fit_bayes$pred$Y[index_bayes])

# Plotting ROC curve with Youden's index highlighted
plot(roc_bayes,print.thres="best",  print.thres.best.method="youden")

J_bayes <- coords(roc_bayes, x="best", input = c("threshold", "specificity", "sensitivity"), best.method = "youden", transpose=TRUE)

# Generating predictions on the edx_test set with the fit, optimised for probabilistic threshold set by Youden's statistic
y_hat_bayes  <- ifelse(predict(fit_bayes, edx_test, type="prob")> J_bayes[1], "Y", "N")
y_hat_bayes <- as.factor(y_hat_bayes[,2])
confusionMatrix(y_hat_bayes, edx_test$Response, positive = "Y", mode = "everything")

# Results of the model are added to a rolling list to compare models
model_AUC <- data.frame("Naive Bayes", round(roc_bayes$auc, digits=4), J = round((as.numeric(J_bayes[3]) + as.numeric(J_bayes[2]) - 1), digits = 4))
colnames(model_AUC) <- c("Model", "AUC", "J-Statistic")
model_AUC


###
### KNN - Using KNN to test
###

control_knn <- trainControl(method = "cv", number = 5, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary)

# Region_Code is added as an additional prediction feature because knn fails to work with the other 6 classifiers alone. Given that the other 6 features are categorical data, the error: 'too many ties in knn' was returned and caused the training to fail. 

fit_knn <- train(Response ~ Gender + Age + Previously_Insured + Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel + Region_Code, preProcess = c("center", "scale"), method = "knn", data = edx_train, trControl = control_knn, metric = "ROC")

y_hat_knn <- predict(fit_knn, edx_test)
confusionMatrix(y_hat_knn, edx_test$Response, positive = "Y", mode = "everything")

index_knn <- fit_knn$pred$k == as.numeric(fit_knn$bestTune)

roc_knn <- roc(fit_knn$pred$obs[index_knn],
               fit_knn$pred$Y[index_knn])

plot(roc_knn, print.thres="best", print.thres.best.method="youden")  

J_knn <- coords(roc_knn, x="best", input = c("threshold", "specificity", "sensitivity"), best.method = "youden", transpose=TRUE)

y_hat_knn <- ifelse(predict(fit_knn, edx_test, type="prob")> J_knn[1], "Y", "N")
y_hat_knn <- as.factor(y_hat_knn[,2])
confusionMatrix(y_hat_knn, edx_test$Response, positive = "Y", mode = "everything")

model_AUC <- rbind(model_AUC, c("kNN", round(roc_knn$auc, digits=4),J = round((as.numeric(J_knn[3]) + as.numeric(J_knn[2]) - 1), digits = 4)))
model_AUC





###
### GLM - Generalized Linear Model
###

control_glm <- trainControl(method = "cv", number = 5, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary)

fit_glm <- train(Response ~ Gender + Age + Previously_Insured + Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel, method = "glm", data = edx_train, trControl = control_glm, metric = "ROC")

# No hyperparameters for GLM

roc_glm <- roc(fit_glm$pred$obs,
                 fit_glm$pred$Y)

plot(roc_glm, print.thres="best", print.thres.best.method="youden")  

J_glm <- coords(roc_glm, x="best", input = c("threshold", "specificity", "sensitivity"), best.method = "youden", transpose=TRUE)

y_hat_glm <- ifelse(predict(fit_glm, edx_test, type="prob")> J_glm[1], "Y", "N")
y_hat_glm <- as.factor(y_hat_glm[,2])
confusionMatrix(y_hat_glm, edx_test$Response, positive = "Y", mode = "everything")

model_AUC <- rbind(model_AUC, c("Generalized Linear Model", round(roc_glm$auc, digits=4),J = round((as.numeric(J_glm[3]) + as.numeric(J_glm[2]) - 1), digits = 4)))
model_AUC





###
### LDA - Linear Discriminatory Analysis 
###

control_lda <- trainControl(method = "cv", number = 5, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary)

fit_lda <- train(Response ~ Gender + Age + Previously_Insured + Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel, method = "lda", data = edx_train, trControl = control_lda, metric = "ROC")

# No hyperparameters for LDA

roc_lda <- roc(fit_lda$pred$obs,
               fit_lda$pred$Y)

plot(roc_lda, print.thres="best", print.thres.best.method="youden")

J_lda <- coords(roc_lda, x="best", input = c("threshold", "specificity", "sensitivity"), best.method = "youden", transpose=TRUE)

y_hat_lda <- ifelse(predict(fit_lda, edx_test, type="prob")> J_lda[1], "Y", "N")
y_hat_lda <- as.factor(y_hat_lda[,2])
confusionMatrix(y_hat_lda, edx_test$Response, positive = "Y", mode = "everything")


model_AUC <- rbind(model_AUC, c("Quadratiic Discriminant Analysis", round(roc_lda$auc, digits=4),J = round((as.numeric(J_lda[3]) + as.numeric(J_lda[2]) - 1), digits = 4)))
model_AUC



###
### ada - Boosted classification trees 
###

control_ada <- trainControl(method = "cv", number = 3, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary)

fit_ada <- train(Response ~ Gender + Age + Previously_Insured + Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel, method = "ada", data = edx_train, trControl = control_ada, metric = "ROC")

index_ada <- (fit_ada$pred$iter == as.numeric(fit_ada$bestTune[1]) & fit_ada$pred$maxdepth == as.numeric(fit_ada$bestTune[2]) & fit_ada$pred$nu == as.numeric(fit_ada$bestTune[3]))

roc_ada <- roc(fit_ada$pred$obs[index_ada],
               fit_ada$pred$Y[index_ada])

plot(roc_ada, print.thres="best", print.thres.best.method="youden")  

J_ada <- coords(roc_ada, x="best", input = c("threshold", "specificity", "sensitivity"), best.method = "youden", transpose=TRUE)

y_hat_ada <- ifelse(predict(fit_ada, edx_test, type="prob")> J_ada[1], "Y", "N")
y_hat_ada <- as.factor(y_hat_ada[,2])
confusionMatrix(y_hat_ada, edx_test$Response, positive = "Y", mode = "everything")

model_AUC <- rbind(model_AUC, c("Boosted Classification Trees", round(roc_ada$auc, digits=4),J = round((as.numeric(J_ada[3]) + as.numeric(J_ada[2]) - 1), digits = 4)))
model_AUC




###
### GBM - Stochastic Gradient Boosting 
###

control_gbm <- trainControl(method = "cv", number = 3, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary)

fit_gbm <- train(Response ~ Gender + Age + Previously_Insured + Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel, method = "gbm", data = edx_train, trControl = control_ada, metric = "ROC")
fit_gbm$bestTune[4]

index_gbm <- (fit_gbm$pred$n.trees == as.numeric(fit_gbm$bestTune[1]) & fit_gbm$pred$interaction.depth == as.numeric(fit_gbm$bestTune[2]) & fit_gbm$pred$shrinkage == as.numeric(fit_gbm$bestTune[3]) & fit_gbm$pred$n.minobsinnode == as.numeric(fit_gbm$bestTune[4])) 

roc_gbm <- roc(fit_gbm$pred$obs[index_gbm],
               fit_gbm$pred$Y[index_gbm])

plot(roc_gbm, print.thres="best", print.thres.best.method="youden")  

J_gbm <- coords(roc_gbm, x="best", input = c("threshold", "specificity", "sensitivity"), best.method = "youden", transpose=TRUE)

y_hat_gbm <- ifelse(predict(fit_gbm, edx_test, type="prob")> J_gbm[1], "Y", "N")
y_hat_gbm <- as.factor(y_hat_gbm[,2])
confusionMatrix(y_hat_gbm, edx_test$Response, positive = "Y", mode = "everything")

model_AUC <- rbind(model_AUC, c("Stochastic Gradient Boostiing", round(roc_gbm$auc, digits=4),J = round((as.numeric(J_gbm[3]) + as.numeric(J_gbm[2]) - 1), digits = 4)))
model_AUC





###
### QDA - Quadratic Discriminatory Analysis
###

control_qda <- trainControl(method = "cv", number = 5, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary)

fit_qda <- train(Response ~ Gender + Age + Previously_Insured + Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel, method = "qda", data = edx_train, trControl = control_qda, metric = "ROC")

# No hyperparameters for QDA

roc_qda <- roc(fit_qda$pred$obs,
               fit_qda$pred$Y)

plot(roc_qda, print.thres="best", print.thres.best.method="youden")

J_qda <- coords(roc_qda, x="best", input = c("threshold", "specificity", "sensitivity"), best.method = "youden", transpose=TRUE)

y_hat_qda <- ifelse(predict(fit_qda, edx_test, type="prob")> J_qda[1], "Y", "N")
y_hat_qda <- as.factor(y_hat_qda[,2])
confusionMatrix(y_hat_qda, edx_test$Response, positive = "Y", mode = "everything")


model_AUC <- rbind(model_AUC, c("Quadratiic Discriminant Analysis", round(roc_qda$auc, digits=4),J = round((as.numeric(J_qda[3]) + as.numeric(J_qda[2]) - 1), digits = 4)))
model_AUC




###
### Random forest
###

control_rf <- trainControl(method = "cv", number = 5, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary)

fit_rf <- train(Response ~ Gender + Age + Previously_Insured + Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel, method = "rf", data = edx_train, trControl = control_rf, metric = "ROC")

index_rf <- fit_rf$pred$mtry == as.numeric(fit_rf$bestTune)

roc_rf <- roc(fit_rf$pred$obs[index_rf],
              fit_rf$pred$Y[index_rf])

plot(roc_rf, print.thres="best", print.thres.best.method="youden")

J_rf <- coords(roc_rf, x="best", input = c("threshold", "specificity", "sensitivity"), best.method = "youden", transpose=TRUE)

y_hat_rf <- ifelse(predict(fit_rf, edx_test, type="prob")> J_rf[1], "Y", "N")
y_hat_rf <- as.factor(y_hat_rf[,2])
confusionMatrix(y_hat_rf, edx_test$Response, positive = "Y", mode = "everything")

model_AUC <- rbind(model_AUC, c("Random Forests", round(roc_rf$auc, digits=4),J = round((as.numeric(J_rf[3]) + as.numeric(J_rf[2]) - 1), digits = 4)))
model_AUC




###
### LogitBoost - Boosted Logistic Regression
###

control_logboost <- trainControl(method = "cv", number= 5, summaryFunction = twoClassSummary, classProbs = TRUE, verboseIter = TRUE, savePredictions = TRUE)

fit_logboost <- train(Response ~ Gender + Age + Previously_Insured + Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel, method = "LogitBoost", data = edx_train, metric="ROC", trControl = control_logboost)

index_logboost <- fit_logboost$pred$nIter == as.numeric(fit_logboost$bestTune)

roc_logboost <- roc(fit_logboost$pred$obs[index_logboost],
                    fit_logboost$pred$Y[index_logboost])

plot(roc_logboost, print.thres="best", print.thres.best.method="youden")

J_logboost <- coords(roc_logboost, x="best", input = c("threshold", "specificity", "sensitivity"), best.method = "youden", transpose=TRUE)

y_hat_logboost <- ifelse(predict(fit_logboost, edx_test, type="prob")> J_logboost[1], "Y", "N")
y_hat_logboost <- as.factor(y_hat_logboost[,2])
confusionMatrix(y_hat_logboost, edx_test$Response, positive = "Y", mode = "everything")

model_AUC <- rbind(model_AUC, c("Boosted Logistic Regression", round(roc_logboost$auc, digits=4),J = round((as.numeric(J_logboost[3]) + as.numeric(J_logboost[2]) - 1), digits = 4)))
model_AUC




###
### XGB - eXtreme Gradient Boosting
###

control_xgb <- trainControl(method = "cv", number = 5, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary)

fit_xgb <- train(Response ~ Gender + Age + Previously_Insured + Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel, method = "xgbTree", data = edx_train, metric = "ROC", trControl = control_xgb)

index_xgb <- (fit_xgb$pred$nrounds == as.numeric(fit_xgb$bestTune[1]) & fit_xgb$pred$max_depth == as.numeric(fit_xgb$bestTune[2]) & fit_xgb$pred$eta == as.numeric(fit_xgb$bestTune[3]) & fit_xgb$pred$gamma == as.numeric(fit_xgb$bestTune[4]) & fit_xgb$pred$colsample_bytree == as.numeric(fit_xgb$bestTune[5]) & fit_xgb$pred$min_child_weight == as.numeric(fit_xgb$bestTune[6]) & fit_xgb$pred$subsample == as.numeric(fit_xgb$bestTune[7]))

fit_xgb$bestTune

roc_xgb <- roc(fit_xgb$pred$obs[index_xgb],
                    fit_xgb$pred$Y[index_xgb])

plot(roc_xgb, print.thres="best", print.thres.best.method="youden")

J_xgb <- coords(roc_xgb, x="best", input = c("threshold", "specificity", "sensitivity"), best.method = "youden", transpose=TRUE)

y_hat_xgb <- ifelse(predict(fit_xgb, edx_test, type="prob")> J_xgb[1], "Y", "N")
y_hat_xgb <- as.factor(y_hat_xgb[,2])
confusionMatrix(y_hat_xgb, edx_test$Response, positive = "Y", mode = "everything")

model_AUC <- rbind(model_AUC, c("eXtreme Gradient Boosting", round(roc_xgb$auc, digits=4),J = round((as.numeric(J_xgb[3]) + as.numeric(J_xgb[2]) - 1), digits = 4)))
model_AUC



################################

# FINAL MODEL

################################

### Using the chosen machine learning algorithm to solve for the validation set

final_control <- trainControl(method = "cv", number = 5, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary)

fit_final <- train(Response ~ Gender + Age + Previously_Insured + Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel, method = "xgbTree", data = edx, metric = "ROC", trControl = final_control)

# Determining best hyperparameters too use for the model

index_final <- (fit_final$pred$nrounds == as.numeric(fit_final$bestTune[1]) & fit_final$pred$max_depth == as.numeric(fit_final$bestTune[2]) & fit_final$pred$eta == as.numeric(fit_final$bestTune[3]) & fit_final$pred$gamma == as.numeric(fit_final$bestTune[4]) & fit_final$pred$colsample_bytree == as.numeric(fit_final$bestTune[5]) & fit_final$pred$min_child_weight == as.numeric(fit_final$bestTune[6])& fit_final$pred$subsample == as.numeric(fit_final$bestTune[7]))

roc_final <- roc(fit_final$pred$obs[index_final],
                 fit_final$pred$Y[index_final])

plot(roc_final, print.thres="best", print.thres.best.method="youden")
#plot(roc_final, print.thres="best", print.thres.best.method="closest.topleft", add=TRUE)

J_final <- coords(roc_final, x="best", input = c("threshold", "specificity", "sensitivity"), best.method = "youden", transpose=TRUE)

J_final

y_hat_final <- ifelse(predict(fit_final, validation, type="prob")> J_final[1], "Y", "N")
y_hat_final <- as.factor(y_hat_final[,2])

confusionMatrix(y_hat_final, validation$Response, mode = "everything", positive = "Y")

model_AUC <- rbind(model_AUC, c("Final Validation Set - eXtreme Gradient Boosting", round(roc_final$auc, digits=4),J = round((as.numeric(J_final[3]) + as.numeric(J_final[2]) - 1), digits = 4)))
model_AUC



