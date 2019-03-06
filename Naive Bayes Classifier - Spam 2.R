
library(caret)
require(quanteda) #natural language processing package
require(RColorBrewer)
require(ggplot2)

## Read in Data
spam = read.csv("C:/Users/sroberts/Downloads/spam.csv", header=TRUE, sep=",", quote='\"\"', stringsAsFactors=FALSE)
table(spam$v1)
names(spam) <- c("type","message")
head(spam)

## Randomly shuffling the dataset
set.seed(2012)
spam <- spam[sample(nrow(spam)),]

msg.corpus <- corpus(spam$message)
#attaching the class labels to the corpus message text
docvars(msg.corpus) <- spam$type

## Wordcloud of Spam messages
#subsetting only the spam messages
spam.plot <- corpus_subset(msg.corpus, docvar1=="spam")
#now creating a document-feature matrix using dfm()
spam.plot <- dfm(spam.plot, tolower=TRUE, remove_punct=TRUE, remove_twitter=TRUE, remove_numbers=TRUE, remove=stopwords("SMART"))
spam.col <- brewer.pal(10, "BrBG")  
textplot_wordcloud(spam.plot, min_count=16, color=spam.col)  
title("Spam Wordcloud", col.main="grey14")

## Wordcloud of Spam messages
ham.plot <- corpus_subset(msg.corpus, docvar1=="ham")
ham.plot <- dfm(ham.plot, tolower=TRUE, remove_punct=TRUE, remove_twitter=TRUE, remove_numbers=TRUE, remove=c("gt", "lt")) #stopwords(source="smart")
ham.col = brewer.pal(10, "BrBG")
textplot_wordcloud(ham.plot, min_count=50, color=ham.col, fixed_aspect=TRUE)
title("Ham Wordcloud", col.main="grey14")


## Partition Data
spam.train<-spam[1:4458,]
spam.test<-spam[4458:nrow(spam),]
msg.dfm <- dfm(msg.corpus, tolower=TRUE)
msg.dfm <- dfm_trim(msg.dfm, min_termfreq=5, min_docfreq=3)
msg.dfm <- dfm_tfidf(msg.dfm)
msg.dfm
msg.dfm.train <- msg.dfm[1:4458,]
msg.dfm.test <- msg.dfm[4458:nrow(spam),]

## Naive Bayes model
nb.classifier <- textmodel_nb(msg.dfm.train, spam.train[,1])
nb.classifier
pred <- predict(nb.classifier, msg.dfm.test)

table(predicted=pred,actual=spam.test[,1])
conf.mat <- confusionMatrix(data=pred, reference=as.factor(spam.test[,1]))
conf.mat
