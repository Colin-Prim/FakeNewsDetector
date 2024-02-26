import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("news.csv")

#  the .shape function (note that there are no parentheses) returns the dimensions of the relevant pandas dataframe
print(df.shape)
#  the .head() function returns the first five rows of the relevant pandas dataframe
print(df.head())

#  the .label is not a function, but the name of the label column (which shows news as either real or fake)
#  this will be used in classification
labels = df.label
print(labels.head())

#  notice that the train_test_split() fcn returns 4 different dataframes
#    the df["text"] is an input array from the column labeled "text"
#    labels is the complement array with the appropriate labels
#    the train/test is the default 80/20
#    the random_state is used for reproducibility
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#  the TfidfVectorizer() fcn sets defaults for future tfidf functions
#    this will measure the importance of the words and will help with the analysis somehow
#    stop_words = 'english' denotes standard stop words in the English language
#    the max_df = 0.7 removes words above a document frequency of 0.7
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

#  transforms the text of each article in the train and test sets into tf.idf matrices
#  fit and transform
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
#  transform
tfidf_test = tfidf_vectorizer.transform(x_test)

#  a passive-aggressive classifier is an algorithm frequently used for fake news detection and other
#    online-learning problems.
#  it works by changing its model in 'real-time' rather than using batch updates.
#  the algorithm is passive when its prediction is correct (the model does not change)
#    and aggressive when its prediction is incorrect (the model does change)
#  this is especially useful when a dataset is so large that training on the full training set is not feasible
#    this algorithm instead analyzes one point, then discards it until its next iteration

#  the max_iter = 50 argument tells the algorithm to iterate only 50 times.
pac = PassiveAggressiveClassifier(max_iter=50)
#  use the classifier on the training data
pac.fit(tfidf_train, y_train)

#  use the model to predict using the test data
y_pred = pac.predict(tfidf_test)
#  measure the accuracy by comparing predicted labels to real labels
score = accuracy_score(y_test, y_pred)
#  print accuracy as a percentage, rounded to two decimals
print(f'Accuracy: {round(score * 100, 2)}%')

#  build the confusion matrix
print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))
