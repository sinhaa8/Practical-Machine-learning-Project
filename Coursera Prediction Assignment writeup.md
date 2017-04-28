---
title: "Coursera Prediction Assignment Writeup"
author: "Ashish Sinha"
date: "April 28, 2017"
output: html_document
---



## Introduction

It is now possible to collect a large amount of data about personal activity relatively inexpensively Using devices such as Jawbone Up, Nike FuelBand, and Fitbit . These type of devices are part of the quantified self movement a group of enthusiasts to find patterns in their behavior, or because they are tech geeks and who take measurements about themselves regularly to improve their health, . One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.
Data Preprocessing

```r
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
```
Read the Data

```r
traindata <- read.csv("C:/Ashish_Data/Work/coursera/Module8/pml-training.csv")
testdata <- read.csv("C:/Ashish_Data/Work/coursera/Module8/pml-testing.csv")
dim(traindata)
```

```
## [1] 19622   160
```

```r
dim(testdata)
```

```
## [1]  20 160
```
The training data set contains 19622 observations and 160 variables, while the testing data set contains 20 observations and 160 variables. The "classe" variable in the training set is the outcome to predict.

Clean the data

In this step, we will clean the data and get rid of observations with missing values as well as some meaningless variables

```r
sum(complete.cases(traindata))
```

```
## [1] 406
```
First, we remove columns that contain NA missing values.

```r
traindata <- traindata[, colSums(is.na(traindata)) == 0] 
testdata <- testdata[, colSums(is.na(testdata)) == 0] 
```
Next, we get rid of some columns that do not contribute much to the accelerometer measurements.

```r
classe <- traindata$classe
trainRemove <- grepl("^X|timestamp|window", names(traindata))
traindata <- traindata[, !trainRemove]
trainCleaned <- traindata[, sapply(traindata, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testdata))
testdata <- testdata[, !testRemove]
testCleaned <- testdata[, sapply(testdata, is.numeric)]
```
Now, the cleaned training data set contains 19622 observations and 53 variables, while the testing data set contains 20 observations and 53 variables. The "classe" variable is still in the cleaned training set.

Slice the data

Then, we can split the cleaned training set into a pure training data set (70%) and a validation data set (30%). We will use the validation data set to conduct cross validation in future steps.

```r
set.seed(21000) # For reproducibile purpose
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
traindata <- trainCleaned[inTrain, ]
testdata <- trainCleaned[-inTrain, ]
```

Data Modeling

We fit a predictive model for activity recognition using Random Forest algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general. We will use 5-fold cross validation when applying the algorithm.

```r
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=traindata, method="rf", trControl=controlRf, ntree=250)
modelRf
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10989, 10990, 10990, 10990, 10989 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9890806  0.9861847
##   27    0.9900268  0.9873845
##   52    0.9805638  0.9754123
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```
Then, we estimate the performance of the model on the validation data set.

```r
predictRf <- predict(modelRf, testdata)
confusionMatrix(testdata$classe, predictRf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    0    2    0    0
##          B   11 1127    1    0    0
##          C    0    5 1017    4    0
##          D    0    0    5  959    0
##          E    0    0    5    3 1074
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9939          
##                  95% CI : (0.9915, 0.9957)
##     No Information Rate : 0.286           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9923          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9935   0.9956   0.9874   0.9928   1.0000
## Specificity            0.9995   0.9975   0.9981   0.9990   0.9983
## Pos Pred Value         0.9988   0.9895   0.9912   0.9948   0.9926
## Neg Pred Value         0.9974   0.9989   0.9973   0.9986   1.0000
## Prevalence             0.2860   0.1924   0.1750   0.1641   0.1825
## Detection Rate         0.2841   0.1915   0.1728   0.1630   0.1825
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9965   0.9965   0.9928   0.9959   0.9992
```

```r
precision <- postResample(predictRf, testdata$classe)
precision
```

```
##  Accuracy     Kappa 
## 0.9938828 0.9922612
```

```r
error <- 1 - as.numeric(confusionMatrix(testdata$classe, predictRf)$overall[1])
error
```

```
## [1] 0.006117247
```
So, the estimated precision of the model is 99.42% and the estimated out-of-sample error is 0.58%.

Predicting for Test Data Set

Now, we apply the model to the original testing data set downloaded from the data source. We remove the problem_id column first.

```r
outcome <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
outcome
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
## Appendix: Figures
Correlation Matrix Visualization

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10-1.png)
Decision Tree Visualization
![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-11-1.png)
