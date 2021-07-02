# Import packages
library(caret)
library(dplyr)

# View a summary of the data
summary(trData)
# View structure of the data with a list.len that equals the number of columns
# in the dataset
str(trData, list.len = ncol(trData)) #dtypes are all either int or num

# create a working copy of the data
workData <- trData

# Remove features with zero variance
zeroVarData <- nearZeroVar(trData, saveMetrics = TRUE)
summary(zeroVarData)
# View features with zeroVar and near-zero-var
zeroVar <- zeroVarData[zeroVarData[,"zeroVar"] > 0, ]#55 obs with zeroVar -- remove
nzv <- zeroVarData[zeroVarData[,"nzv"] > 0, ] #520 obs with nzv -- keep these
# Remove all of the zeroVar columns - First take names of all columns in dataset, 
# then use rownames of zeroVar and create new dataframe with difference between 
# those all columns and zeroVar columns
all_cols <- names(trData)
rem_cols <- row.names(zeroVar)
newData <- trData[ , setdiff(all_cols, rem_cols)]

# EDA visualization
# Check out the distribution of building id, floor, and space id
hist(trData$BUILDINGID)
hist(trData$FLOOR)
hist(trData$SPACEID)

plot(trData$LONGITUDE, trData$LATITUDE) # get a visualization of the locations,
# results are very similar to the campus map that was shown

#############################################################################

# Subset a random sample (~20%) of the data for model testing
sampleData <- newData[sample(1:nrow(trData), 4000, replace=FALSE),]

# Remove unneccessary attributes
sampleData <- sampleData %>% select(-(RELATIVEPOSITION:TIMESTAMP)) 
sampleData <- sampleData %>% select(-(LONGITUDE:LATITUDE))


# Create a composite attribute for the building, floor, and specific location 
# attributes into a single unique identifier for each instance.
composite_df <- sampleData %>% mutate(LOCATION = paste0(FLOOR, "_", BUILDINGID, "_", SPACEID))
testData <- composite_df %>% select(-(FLOOR:SPACEID))
str(testData$LOCATION)
# Convert testData$LOCATION to factor dtype
testData$LOCATION <- as.factor(testData$LOCATION)
str(testData$LOCATION)

## sampleData contains a sample of ~20% of dataframe with zeroVar variables removed,
## and with FLOOR, BUILDINGID, and SPACEID as separate columns

## testData contains a sample of ~20% of dataframe with zeroVar variables removed,
## and with FLOOR, BUILDINGID, and SPACEID as one composite column called LOCATION

############################################################################

# I'm going to change my approach. Instead of taking a sample from the entire dataframe,
# I will separate the data by building ID - Building ID is either 0, 1, or 2

newData2 <- newData %>% select(-(LONGITUDE:LATITUDE))
newData2 <- newData %>% select(-(RELATIVEPOSITION:TIMESTAMP))

# Filter data so we only have data for Building 0
dataB0 <- filter(newData2, BUILDINGID == 0)
# Create composite ID for location
dataB0 <- dataB0 %>% mutate(LOCATION = paste0(BUILDINGID, "_", FLOOR, "_", SPACEID))
# Remove unneeded variables
dataB0 <- dataB0 %>% select(-(FLOOR:SPACEID))
dataB0 <- dataB0 %>% select(-(LONGITUDE:LATITUDE))
# Change to factor
dataB0$LOCATION <- as.factor(dataB0$LOCATION)
str(dataB0$LOCATION)

# Filter data for Building 1
dataB1 <- filter(newData2, BUILDINGID == 1)
# Create composite ID for location
dataB1 <- dataB1 %>% mutate(LOCATION = paste0(BUILDINGID, "_", FLOOR, "_", SPACEID))
# Remove unneeded variables
dataB1 <- dataB1 %>% select(-(FLOOR:SPACEID))
dataB1 <- dataB1 %>% select(-(LONGITUDE:LATITUDE))
# Change to factor
dataB1$LOCATION <- as.factor(dataB1$LOCATION)

# Filter data for Building 2
dataB2 <- filter(newData2, BUILDINGID == 2)
# Create composite ID for location
dataB2 <- dataB2 %>% mutate(LOCATION = paste0(BUILDINGID, "_", FLOOR, "_", SPACEID))
# Remove unneeded variables
dataB2 <- dataB2 %>% select(-(FLOOR:SPACEID))
dataB2 <- dataB2 %>% select(-(LONGITUDE:LATITUDE))
# Change to factor
dataB2$LOCATION <- as.factor(dataB2$LOCATION)


############################################################################
# Model Testing - testing models with sample data
############################################################################

# We will start by using testData
# Train/test split
set.seed(123)
inTraining <- createDataPartition(testData$LOCATION, p = .75, list = FALSE)
training <- testData[inTraining,]
testing <- testData[-inTraining,]

# 10 fold cv
fitControl <- trainControl(method = 'repeatedcv', number = 10, repeats =1)

# Train model - Let's start with SVM - 
# Least Squares Support Vector Machine with Radial Basis Function Kernel
system.time(svmFit1 <- train(LOCATION~. , data = training, method = "lssvmRadial", 
                 trControl = fitControl))
svmFit1

# Accuracy 0.4389165, Kappa 0.4375233
# The final values used for the model were sigma =
# 6.754978e-07 and tau = 0.0625.
# 3273 samples
# 465 predictor
# 728 classes

#############################################################################

# Start modeling with data for Building 0

# Train/test split
set.seed(123)
inTrainingB0 <- createDataPartition(dataB0$LOCATION, p = .75, list = FALSE)
trainingB0 <- dataB0[inTrainingB0,]
testingB0 <- dataB0[-inTrainingB0,]

#################
# Train KNN model
#################
system.time(knnFit1 <- train(LOCATION~. , data = trainingB0, method = "knn", 
                             trControl = fitControl))
knnFit1

# Results
# k  Accuracy   Kappa    
# 5  0.5499506  0.5480091
# 7  0.5329365  0.5309299
# 9  0.5051745  0.5030592

# Accuracy was used to select the optimal model using
# the largest value.
# The final value used for the model was k = 5.

#####################
## Now let's try SVM
#####################
system.time(svmFit2 <- train(LOCATION~. , data = trainingB0, method = "lssvmRadial",
                             trControl = fitControl))
svmFit2

# Accuracy   Kappa    
# 0.5843254  0.5825174
# The final values used for the model were sigma =
# 9.154607e-07 and tau = 0.0625.


################################
## Random Forest for Building 0
###############################
system.time(rfFit1 <- train(LOCATION~. , data = trainingB0, method = "rf",
                            trControl = fitControl))
rfFit1

# mtry  Accuracy     Kappa    
# 2   0.008006066  0.0000000
# 233   0.756296105  0.7552358
# 465   0.745558817  0.7444506

# Accuracy was used to select the optimal model using
# the largest value.
# The final value used for the model was mtry = 233.

# Make some predictions using this model
rfPred1 <- predict(rfFit1, testingB0)
rfPred1

confusionMatrix(rfPred1, testingB0$LOCATION)

# Accuracy : 0.767           
# 95% CI : (0.7425, 0.7901)
# No Information Rate : 0.008           
# P-Value [Acc > NIR] : < 2.2e-16       
# Kappa : 0.766

# Post Resample
postResample(rfPred1, testingB0$LOCATION)

#  Accuracy     Kappa 
# 0.7669593   0.7659957 

############################
## Use RF for Building 1
############################

# Train/test split

set.seed(123)
inTrainingB1 <- createDataPartition(dataB1$LOCATION, p = .75, list = FALSE)
trainingB1 <- dataB1[inTrainingB1,]
testingB1 <- dataB1[-inTrainingB1,]


system.time(rfFit2 <- train(LOCATION~. , data = trainingB1, method = "rf",
                            trControl = fitControl))
rfFit2

# mtry  Accuracy    Kappa      
# 2   0.03548567  0.009205643
# 233   0.85109230  0.849669574
# 465   0.83489006  0.833314150

# Accuracy was used to select the optimal model using
# the largest value.
# The final value used for the model was mtry = 233.

# Make some predictions using this model
rfPred2 <- predict(rfFit2, testingB1)
rfPred2

confusionMatrix(rfPred2, testingB1$LOCATION)

# Accuracy : 0.8624          
# 95% CI : (0.8421, 0.8809)
# No Information Rate : 0.027           
# P-Value [Acc > NIR] : < 2.2e-16       
# Kappa : 0.8611

# Post Resample
postResample(rfPred2, testingB1$LOCATION)

#  Accuracy     Kappa 
# 0.8623707   0.8610553 

#############################
## Run RF for building 2
#############################

# Train/test split

set.seed(123)
inTrainingB2 <- createDataPartition(dataB2$LOCATION, p = .75, list = FALSE)
trainingB2 <- dataB2[inTrainingB2,]
testingB2 <- dataB2[-inTrainingB2,]


system.time(rfFit3 <- train(LOCATION~. , data = trainingB2, method = "rf",
                            trControl = fitControl))
rfFit3

# Make Predictions
rfPred3 <- predict(rfFit3, testingB2)
rfPred3

confusionMatrix(rfPred3, testingB2$LOCATION)

# Post Resampling
postResample(rfPred3, testingB2$LOCATION)

# Accuracy     Kappa 
# 0.8269987  0.8262549

#####################
# Resamples
#####################

# Resampling of RF models
modelData <- resamples(list(B0 = rfFit1, B1 = rfFit2, B2 = rfFit3))
summary(modelData)

# Resampling of models for B0
modelData2 <- resamples(list(KNN = knnFit1, SVM = svmFit2, RF = rfFit1))
summary(modelData2)
