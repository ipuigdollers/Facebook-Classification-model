setwd("~/Documents/Facebook Crawler/Blocket/Data")
data <- read.csv("Blocket_ads.csv",header=T,sep=",",quote="")
data<-data[-1:-241,]#first 241 ads have different structure
colnames(data)<-c("list_id","category","price","subject","body")
data<-subset(data,body!="")#delete ads without body

#Rename categories in terms of verticals
temp<-as.numeric(as.character(data$category))
temp[temp>=1000 & temp<2000]<-1#1 means vertical "motor"
temp[temp>=3000 & temp<4000]<-2#2 means vertical "real estate"
temp[temp>=9000]<-3#3 means vertical "jobs"
temp[temp>=4]<-4#4 means vertical "mmarketplace"
data$category<-as.factor(temp)

library(caret)
libs <- c("tm", "plyr", "class", "RWeka")
lapply(libs, require, character.only = TRUE)
library("RWeka")
library(e1071)

Class <- as.factor(data$category)#Save category classes

body <- data$body
body.data <- as.data.frame(body)

#Text preparation (remove punctuation, stopwords, ...)
body.corpus <- Corpus(DataframeSource(body.data), readerControl = list(language = "swedish"))
body.corpus <- tm_map(body.corpus, removePunctuation)
body.corpus <- tm_map(body.corpus, stripWhitespace)
body.corpus <- tm_map(body.corpus, content_transformer(tolower))
body.corpus <- tm_map(body.corpus, removeWords,stopwords("swedish"))

#Build the Document-Term matrix
DocTermMatrix <- DocumentTermMatrix(body.corpus)

###Removing the sparse terms, i.e. terms appearing 0 times in documents

for(i in seq(from =0.9, to =0.999, by=0.001)){#We do not want more than 250 features
  temp<-removeSparseTerms(DocTermMatrix,i)
  if(temp$ncol>250)break()
  DocTermMatrix.final<-temp
}

#Write to matrices

matrix_bodymonogramdtm <- as.matrix(DocTermMatrix.final)
DocTermMatrix.data <- as.data.frame(matrix_bodymonogramdtm)

# #joining with original dataset
# 
# bodymonogramset.tmp <- cbind(body, DocTermMatrix.data)
# newset <- bodymonogramset.tmp
# newset2 <- newset[-1]#Remove body

#Make train/test split

inTrain <- createDataPartition(Class, p = 0.8, list = FALSE)

trainDescr <- DocTermMatrix.data[inTrain,]
testDescr <- DocTermMatrix.data[-inTrain,]
trainClass <- Class[inTrain]
testClass <- Class[-inTrain]

#Build the Naive Bayes Model

fit <- naiveBayes(as.factor(trainClass) ~ ., data= trainDescr,laplace = 1)#laplace, stands for Laplace Smoothing factor to avoid the reamining sparsity
pred <- predict(fit, testDescr)

a<-table(pred, testClass)
a#Show confusion matrix

# #Evaluation of the model
# 
# #Accuracy
# accuracy<-sum(diag(a))/sum(a)
# accuracy
# #Recall
# addition<-0
# for(i in 1:length(a[,1]))addition<-addition+(diag(a)[i]/sum(a[,i]))
# recall<-addition/ncol(a)
# recall
# #Precision
# addition<-0
# for(i in 1:ncol(a))addition<-addition+(diag(a)[i]/sum(a[i,]))
# precision<-addition/ncol(a)
# precision
# #F-meqasure
# fmeasure<-2 * precision * recall / (precision + recall)
# fmeasure
