#Import h2o library for machine learning
library(h2o)

# initialize the thread  and connect to h2o Cluster
h2o.init(nthreads = -1)

# Training data sets
train_data<-"http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
train <-h2o.importFile(train_data)
summary(train)

# Testing data sets 
test_file<-"http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"
test<-h2o.importFile(test_file)
summary(test)

# Splitting the data into training and testing data sets
y <- "C65"
x<- setdiff(names(train), y)
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])
splits <- h2o.splitFrame(train, 0.80, seed=12345)

# Deep learning model # Mean Square Error

model <- h2o.deeplearning(
  x = x,
  y = y,
  training_frame = splits[[1]],
  validation_frame = splits[[2]],
  loss = "Quadratic",
  distribution = "multinomial",
  activation = "Tanh",
  hidden = c(310,310,310),
  adaptive_rate = FALSE,
  rate = 0.01,
  momentum_start = 0.2,
  input_dropout_ratio = 0.2,
  sparse = TRUE,
  l1 = 1e-5, 			#regularization
  epochs = 100)

model@parameters
h2o.confusionMatrix(model,train)
h2o.confusionMatrix(model,test)
prediction <- h2o.predict(model, newdata = test)
accuracy <- sum(test[,y] == prediction$predict)/nrow(test)
print(accuracy)
model@model$scoring_history


#To Print Class Accuracy
rownames = c("Class0", " Class1", " Class2"," Class3"," Class4"," Class5"," Class6"," Class7"," Class8"," Class9")
colnames = c("Accuracy")

# Training Confusion Matrix
CM<- h2o.confusionMatrix(model, train)
Error<-CM$Error
Accuracy<-1 - Error
result <- array(c(Accuracy),dim = c(1,10,1),dimnames = list(colnames,rownames))
print(result)

# Testing Confusion Matrix
CM<- h2o.confusionMatrix(model, test)
Error<-CM$Error
Accuracy<-1 - Error
result <- array(c(Accuracy),dim = c(1,10,1),dimnames = list(colnames,rownames))
print(result)

# Deep Learning Model with CrossEntropy
model <- h2o.deeplearning(
  x = x,
  y = y,
  training_frame = splits[[1]],
  validation_frame = splits[[2]],
  loss = "CrossEntropy",
  distribution = "multinomial",
  activation = "Tanh",
  hidden = c(100,100,100),
  adaptive_rate = FALSE,
  rate = 0.01,
  momentum_start = 0.2,
  input_dropout_ratio = 0.2,
  sparse = TRUE,
  l1 = 1e-5, 			#regularization
  epochs = 100)

model@parameters
h2o.confusionMatrix(model,train)
h2o.confusionMatrix(model,test)
prediction <- h2o.predict(model, newdata = test)
accuracy <- sum(test[,y] == prediction$predict)/nrow(test)
print(accuracy)
model@model$scoring_history

#To Print Class Accuracy
rownames = c("Class0", " Class1", " Class2"," Class3"," Class4"," Class5"," Class6"," Class7"," Class8"," Class9")
colnames = c("Accuracy")

# Training Confusion Matrix
CM<- h2o.confusionMatrix(model, train)
Error<-CM$Error
Accuracy<-1 - Error
result <- array(c(Accuracy),dim = c(1,10,1),dimnames = list(colnames,rownames))
print(result)

# Testing Confusion Matrix
CM<- h2o.confusionMatrix(model, test)
Error<-CM$Error
Accuracy<-1 - Error
result <- array(c(Accuracy),dim = c(1,10,1),dimnames = list(colnames,rownames))
print(result)


#CNN implementation Use the cross-entropy error function, and ReLU hidden units
install.packages("tensorflow")
library(tensorflow)

install.packages("keras")
library(keras)

# Training data sets
train1 <- read.csv(file = "/Users/kundankumar/Box/Machine_Learning_Project2/optdigits_train.csv")
dim(train1)
train1 <- train1/255
train_x  <- h2o.assign(splits[[1]], "train_x.csv") 
train_y  <- h2o.assign(splits[[2]], "train_y.csv") 
dim(train_x)
dim(train_y)
summary(train_x)

# Testing data sets 
test1 <- read.csv(file = "/Users/kundankumar/Box/Machine_Learning_Project2/optdigits.csv")
test1 <- test1/255
test_x  <- h2o.assign(splits[[1]], "test_x.csv") 
test_y  <- h2o.assign(splits[[2]], "test_y.csv") 

dim(test_x)
dim(test_y)
summary(test_x)

# Model with Keras
model <- keras_model_sequential()

model %>%
  # First layer on convolutional neural network
  layer_conv_2d(filters = 32, 
                kernel_size = c(3,3),
                activation = 'relu',
                input_shape = c(100, 100, 3)) %>%
  layer_conv_2d(filters = 32,
      kernel_size = c(3,3),
      activation = 'relu') %>%
  
  # Pooling layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Dropout layer
  layer_dropout(rate = 0.25) %>%
  
  # Adding one more layer
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  
  layer_flatten() %>%
  
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate=0.25) %>%
  layer_dense(units = 3, activation = 'softmax') %>%
  
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_sgd(lr = 0.01,
                                    decay = 1e-6,
                                    momentum = 0.8,
                                    nesterov = T),
          metrics = c('accuracy'))
summary(model)

# Fit model
history <- model %>%
  fit(train_x,
      train_y,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2,
      validation_data = list(test_x, test_y))

plot(history)

  
  
  