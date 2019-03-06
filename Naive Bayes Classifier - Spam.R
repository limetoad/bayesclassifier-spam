
library(tm)
library(e1071)
library(naivebayes)
library(gmodels)
library(caret)
library(tidyverse)
library(wordcloud)

## Read in data
sms_raw <- read.csv("C:/Users/sroberts/Downloads/spam.csv", stringsAsFactors = FALSE)
sms_raw <- sms_raw[1:2]
names(sms_raw) = c("type", "text")
sms_raw$type <- factor(sms_raw$type)
str(sms_raw)

## Create Corpus using tm package
sms_corpus <- VCorpus(VectorSource(sms_raw$text))
typeof(sms_corpus)
sms_corpus

## Randomly select 2 results from Corpus
length(sms_corpus) %>%
  sample(replace = FALSE) %>%
  sort.list(decreasing = FALSE) %>%
  head(2) %>%
  sms_corpus[.] %>%
  inspect()

## Clean up data
sms_corpus_clean <- sms_corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords()) %>%
  tm_map(removePunctuation) %>%
  tm_map(stemDocument) %>%
  tm_map(stripWhitespace)

## Check text before and after cleaning
for(i in 1:4){
  print(as.character(sms_corpus[[i]]))
}
for(i in 1:4){
  print(as.character(sms_corpus_clean[[i]]))
}

## Create Document Term matrix
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm

## Wordcloud
wordcloud(sms_corpus_clean, min.freq = 50, random.order=FALSE)

## Reduce the number of features
sms_dtm_freq <- sms_dtm %>%
  findFreqTerms(lowfreq=5) %>%
  sms_dtm[ , .]

## Convert to yes/no if the text contains or doesn't the word
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

sms <- apply(sms_dtm_freq, 2, convert_count)
sms <- sms_dtm_freq %>%
  apply(MARGIN=2, FUN=convert_counts)
glimpse(sms)

## Partition data
# labels
lab <- data.frame(as.character(sms_raw$type))

# train and test datasets and labels
trainIndex <- createDataPartition(sms[,1], p=0.7)$Resample1
train = data.frame(sms)[trainIndex, ]
test = data.frame(sms)[-trainIndex, ]
train_labels = as.factor(lab[trainIndex, ])
test_labels = as.factor(lab[-trainIndex, ])

glimpse(train)
glimpse(test)
glimpse(train_labels)
glimpse(test_labels)

## Train Naive Bayes classifer
system.time( sms_classifier <- naiveBayes(x=train, y=train_labels) )
system.time( sms_pred <- predict(object=sms_classifier, newdata=test) )

## Evaluate Performance
CrossTable(sms_pred, test_labels, prop.chisq=FALSE, chisq=FALSE, prop.t=FALSE, dnn=c("Predicted", "Actual"))

## Confusion matrix
conf.mat <- confusionMatrix(data=sms_pred, reference=test_labels)
conf.mat
