Summary
-------

This report is part of the Practical Machine Learning Course. The
assignment is to use different machine learning models to predict
outcome of the different ways of barbell lifts.

The training data:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>
(see the section on the Weight Lifting Exercise Dataset).

Data loading and cleaning
-------------------------

    # Load packages
    library(caret)

    # Read the data
    training <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))
    testing  <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))


    # Cleaning
    # Get column names with NA
    na_columns <- colnames(training[, colSums(is.na(training)) > 0])

    # Get column namess with empty values
    emptyVal_columns <- colnames(training[, colSums(training[,]== "", na.rm = TRUE) > 0])

    # Pick names of the fist seven columns with only indexing values
    index_columns <- colnames(training[,1:7])

    # Remove columns from the datasets
    training <- training[ , -which(names(training) %in% c(na_columns,emptyVal_columns,index_columns))]
    testing <- testing[ , -which(names(testing) %in% c(na_columns,emptyVal_columns,index_columns))]

Data splitting
--------------

Split the data into training, testing and validation subsets.

    set.seed(125)
    inBuild  <- createDataPartition(training$classe, p=0.9, list=FALSE)

    buildData <- training[inBuild, ]
    validation <- training[-inBuild, ]

    inTrain <- createDataPartition(buildData$classe, p=0.7, list=FALSE)

    SubTrain <- buildData[inTrain, ]
    SubTest  <- buildData[-inTrain, ]

Check the dataset dimensions

    dim(SubTrain)

    ## [1] 12365    53

    dim(SubTest)

    ## [1] 5297   53

    dim(validation)

    ## [1] 1960   53

Prediction models
-----------------

We will be using two different prediction modelling methods: Random
Forest and Generalized Boosted Model.

Control parameters:

    control <- trainControl(method="cv", number=2) # K-fold cross-validation, 2-folds (saves calculation time)

### Random Forest

    set.seed(125)
    modelFitRandForest <- train(classe~ .,data=SubTrain,method="rf", trControl=control)

    # prediction on testing subdata
    predictRandForest <- predict(modelFitRandForest, newdata=SubTest)
    cMatrixRandForest <- confusionMatrix(predictRandForest, SubTest$classe)
    cMatrixRandForest

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1505    6    0    0    0
    ##          B    0 1018    8    0    0
    ##          C    0    1  915   14    6
    ##          D    1    0    1  852    2
    ##          E    0    0    0    2  966
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9923          
    ##                  95% CI : (0.9895, 0.9944)
    ##     No Information Rate : 0.2843          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9902          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9993   0.9932   0.9903   0.9816   0.9918
    ## Specificity            0.9984   0.9981   0.9952   0.9991   0.9995
    ## Pos Pred Value         0.9960   0.9922   0.9776   0.9953   0.9979
    ## Neg Pred Value         0.9997   0.9984   0.9979   0.9964   0.9982
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
    ## Detection Rate         0.2841   0.1922   0.1727   0.1608   0.1824
    ## Detection Prevalence   0.2853   0.1937   0.1767   0.1616   0.1827
    ## Balanced Accuracy      0.9989   0.9956   0.9927   0.9903   0.9957

    # Accuracy of the Random Forest
    cMatrixRandForest$overall[1]

    ##  Accuracy 
    ## 0.9922598

### Generalized Boosted Model

    set.seed(125)
    modelFitGbm  <- train(classe ~ ., data=SubTrain, method = "gbm", trControl = control, verbose = FALSE)

    # prediction on testing subdata
    predictGbm <- predict(modelFitGbm, newdata=SubTest)
    cMatixGbm <- confusionMatrix(predictGbm, SubTest$classe)
    cMatixGbm

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1487   35    0    0    1
    ##          B   11  963   27    4   11
    ##          C    4   22  883   21   13
    ##          D    1    4   13  835    9
    ##          E    3    1    1    8  940
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9643         
    ##                  95% CI : (0.959, 0.9692)
    ##     No Information Rate : 0.2843         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9549         
    ##  Mcnemar's Test P-Value : 1.999e-05      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9874   0.9395   0.9556   0.9620   0.9651
    ## Specificity            0.9905   0.9876   0.9863   0.9939   0.9970
    ## Pos Pred Value         0.9764   0.9478   0.9364   0.9687   0.9864
    ## Neg Pred Value         0.9950   0.9855   0.9906   0.9926   0.9922
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
    ## Detection Rate         0.2807   0.1818   0.1667   0.1576   0.1775
    ## Detection Prevalence   0.2875   0.1918   0.1780   0.1627   0.1799
    ## Balanced Accuracy      0.9889   0.9636   0.9710   0.9779   0.9810

    # Accuracy of the Generalized Boosted Model
    cMatixGbm$overall[1]

    ##  Accuracy 
    ## 0.9643194

Model selection and validation
------------------------------

We will be selecting the Random Forest over Generalized Boosted Model
because in this case Random Forest had better accuracy.  
<br />

We can get the estimate for out of sample error by using validation
dataset

    predictValidation <- predict(modelFitRandForest, newdata=validation)
    cMatrixValidation <- confusionMatrix(predictValidation, validation$classe)
    cMatrixValidation

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   A   B   C   D   E
    ##          A 558   3   0   0   0
    ##          B   0 376   5   0   0
    ##          C   0   0 337   2   2
    ##          D   0   0   0 319   0
    ##          E   0   0   0   0 358
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9939          
    ##                  95% CI : (0.9893, 0.9968)
    ##     No Information Rate : 0.2847          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9923          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9921   0.9854   0.9938   0.9944
    ## Specificity            0.9979   0.9968   0.9975   1.0000   1.0000
    ## Pos Pred Value         0.9947   0.9869   0.9883   1.0000   1.0000
    ## Neg Pred Value         1.0000   0.9981   0.9969   0.9988   0.9988
    ## Prevalence             0.2847   0.1934   0.1745   0.1638   0.1837
    ## Detection Rate         0.2847   0.1918   0.1719   0.1628   0.1827
    ## Detection Prevalence   0.2862   0.1944   0.1740   0.1628   0.1827
    ## Balanced Accuracy      0.9989   0.9945   0.9915   0.9969   0.9972

Use the selected model for course testing data
----------------------------------------------

    predict(modelFitRandForest, newdata=testing)

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
