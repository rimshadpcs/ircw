import pandas as pd
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import json

data = pd.read_csv('/Users/rimshad/Downloads/scholar-vertical-search-engine-rimshad/scholar-vertical-search-engine-rimshad/Classification/train.csv')
dataCopy = data
data.head()
data['Category'].value_counts()
data.title = data.Text.apply(simple_preprocess, min_len=3)
data.title.head()
stop_words = set(stopwords.words('english'))


def stemmingandstop(lis):
    lemmatizer = WordNetLemmatizer()
    filtered_lis = [lemmatizer.lemmatize(w) for w in lis if not w in stop_words and len(w) > 2]
    return filtered_lis

data.title = data.title.apply(stemmingandstop)
data.title.head()
type(data.title)
data.title = data.title.apply(' '.join)
data.head()
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
X = data.title
y = data.Category
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
text_clf.fit(X_train, y_train)
predictions = text_clf.predict(X_test)
print(accuracy_score(y_test, predictions))
svm_clftfidf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC()),
])
svm_clftfidf.fit(X_train, y_train)
tfsvmpred = svm_clftfidf.predict(X_test)
print(accuracy_score(y_test, tfsvmpred))
text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None)),
])
parameters = {
 'vect__ngram_range': [(1, 1), (1, 2)],
 'tfidf__use_idf': (True, False),
 'clf__alpha': (1e-2, 1e-3),
}
gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)
gs_value = gs_clf.predict(X_test)
print(accuracy_score(y_test, gs_value))
with open('/Users/rimshad/Downloads/scholar-vertical-search-engine-rimshad/scholar-vertical-search-engine-rimshad/Classification/final.json') as f:
    dataj = json.load(f)
dataj = [row for row in dataj if not (row['title'] is None)]
df = pd.DataFrame(dataj)
df.head()
classification = gs_clf.predict(df.title + df.description)
df['Tag'] = classification
df.to_json('classified.json', orient="records")
