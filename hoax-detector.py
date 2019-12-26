import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt

# CONFUSION MATRIX FUNCTION
# source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# Fungsi untuk menampilkan plot confusion matrix
# Note: Normalization can be applied by setting `normalize=True` 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# MOST INFORMATIVE FEATURE FUNCTION
# Mencetak fitur paling mengandung informasi
# source: https://stackoverflow.com/a/26980472
# Note: current implementation merely prints and does not return top classes.
def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):
    """
    Identify most important features if given a vectorizer and binary classifier. Set n to the number
    of weighted features you would like to show.
    """

    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)

    print()

    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)

# DATA EXPLORATION
# Membaca train data dari csv
df = pd.read_csv('fake_or_real_news.csv')
df.shape
df.head()
df = df.set_index('Unnamed: 0')
df.head()

# STOP WORDS
# referensi: OnnoCenterWiki
# source: https://github.com/masdevid/ID-Stopwords/blob/master/id.stopwords.02.01.2016.txt
# Mendefinisi variabel stopword bahasa indonesia

with open("stopwords_id.txt", "r") as f:
    stopwords = f.read().splitlines()
f.close()

# EXTRACTING DATA
y = df.label
df = df.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# BUILD VECTORIZER CLASSIFIER
# Menghapus stopwords bahasa indonesia, dan kata-kata yang muncul 70% semua artikel
count_vectorizer = CountVectorizer(stop_words=stopwords)
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

tfidf_vectorizer.get_feature_names()[-10:]
count_vectorizer.get_feature_names()[:10]

count_df = pd.DataFrame(count_vectorizer.get_feature_names())
tfidf_df = pd.DataFrame(tfidf_vectorizer.get_feature_names())
difference = set(count_df) - set(tfidf_df)
count_df.head()
tfidf_df.head()

clf = MultinomialNB()
clf.fit(count_train, y_train)
pred = clf.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
most_informative_feature_for_binary_classification(tfidf_vectorizer, clf, n=30)