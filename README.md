# SentimentAnalysis
Preprocessing using NLTK. Implement Sklearn count vectorizer, MNB, TFIDF

Sklearn, NLTK, pandas, numpy, regex 

Read a CSV file of car reviews 

Preprocessing
Created functions using NLTK for negations (n’t not), shortenings (‘ve have), tokenisation, lemmatisation, lowercase, stop words

Labeled negatives with 0, positives with 1

Proportioned out training and test data. To have the same proportion of positive and negative reviews 

Used sklearn: count vectorizer, MNB, TFIDF vectoriser
compared accuracy with and without using it 

///////

A TFIDF vectorizer compares the number of times a word appears in the data with the number of data entries the word appears in.
It is used as it takes into account common words and the length of each data entry. 
Its implementation is similar to the count vectorizer. The simplicity of the count vectorizer just counts the frequency.
Whereas the TDIDF calculates probabilites using the word frequenct and the document frequency. 

Accuracy has gone up on the initital TFDIF. This is because using the TFIDF approach takes account of commonly used words and the length of the document.
Instead of just counting the words like previously, using TFIDF enables comparison between different lengths of documents.
The increase is only small and this is because the feature list created had a similar effect. 
The feature list could be improved further by some confidence interval criterion.
However TDIDF does have its limiations, and can sometime decrease accuracy. 
More improvements could be grouping together similar words and using ngrams. 
