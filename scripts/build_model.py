import re
import pandas as pd
import string
import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB

# Number of random trials
NUM_TRIALS = 3#30

# Read data
# --------------------------------------------------------------------------------------------------------------------
train_x = 'Data/train_X_languages_homework.json.txt'
train_y = 'Data/train_y_languages_homework.json.txt'

with open(train_x) as f:
    data = f.read()
train_x = []
[train_x.append(re.findall(':"(.*?)"}', value, re.S)[0]) for value in data.split('\n') if value]

with open(train_y) as f:
    data = f.read()
train_y = []
[train_y.append(re.findall(':"(.*?)"}', value, re.S)[0]) for value in data.split('\n') if value]

data = pd.DataFrame({'text': train_x, 'class': train_y})

# get number of labels and pick 100 random samples of a particular each label
# --------------------------------------------------------------------------------------------------------------------
data = shuffle(data)
label_count = {}
X = []
y = []
for i in range(0, len(data)):
    item = data['class'][i]
    if item not in label_count:
        label_count[item] = 1
        X.append(data['text'][i])
        y.append(item)
    elif label_count[item] < 100:
        temp = label_count[item]
        label_count[item] = temp + 1
        X.append(data['text'][i])
        y.append(item)

data = pd.DataFrame({'text': X, 'class': y})
No_labels = len(label_count)


# vectorize data
# --------------------------------------------------------------------------------------------------------------------

stop_words = []
for i in string.punctuation:
    stop_words.append(i)
    stop_words.append(' ' + i)
    stop_words.append(' ' + i + ' ')
    stop_words.append(i + ' ')
    stop_words.append(' ')

vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 3), analyzer='char')

X = vectorizer.fit_transform(data['text'])
y = data['class'].values

No_features = len(vectorizer.get_feature_names())

# Feature engineering
# Used TruncatedSVD for dimensionality reduction and generated k-means clusters on the projected data
# Added the k-means clusters as new features to the reduced data
# --------------------------------------------------------------------------------------------------------------------
n_comp = int(No_features/2.5)
svd = TruncatedSVD(n_components=n_comp)
data = svd.fit_transform(X)

kmeans = KMeans(n_clusters=No_labels, random_state=0)
kmeans = kmeans.fit(data)
labels = kmeans.labels_
data = pd.DataFrame(data)
data['clusters'] = labels
X = data

# Model and parameters initialization
# --------------------------------------------------------------------------------------------------------------------
models_and_parameters = {
    'LR': (LogisticRegression(), {'C': [0.1, 1, 10]}),
    'PT': (Perceptron(), {'tol': [1e-3]}),
    'svm': (SVC(), {'C': [0.1, 1, 10], 'gamma': [.01, .1]}),
}

# Experiment to find best model that maximizes out of sample performance by nested cross validation
# --------------------------------------------------------------------------------------------------------------------

final_models = []
average_score = dict()
for name, (model, param_grid) in models_and_parameters.items():
    average_nested_scores = np.zeros(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=i)
        outer_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=i)

        clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring='accuracy')
        average_nested_scores[i] = nested_score.mean()

    average_score[name] = np.mean(average_nested_scores)
    print("Finished training: " + str(name))

best_model_name, best_model_accuracy = max(average_score.items(),
    key=(lambda name_averagescore: name_averagescore[1]))

# Tune the best model to obtain optimized parameters
# --------------------------------------------------------------------------------------------------------------------
best_model, best_model_params = models_and_parameters[best_model_name]

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
final_model = GridSearchCV(best_model, best_model_params, cv=cv, scoring='accuracy', n_jobs=-1)
final_model.fit(X, y)

# creates a pipeline but the pipeline won't be used to predict because we are missing one step (svd + km) so that our
# out of sample data have equal dimensionality with the trained data
# Check make_predictions.py on how I used the pipeline
# --------------------------------------------------------------------------------------------------------------------
clf = Pipeline([
    ('vec', vectorizer),
    ('svd', svd),
    ('km', kmeans),
    ('clf', final_model)
])

# Write serialized version of best model to best_model.bin and
# Write expected performance of best model to performance.txt
# --------------------------------------------------------------------------------------------------------------------
pickle.dump(clf, open('./best_model.bin', 'wb'))

with open('Result/performance.txt', "w+") as f:
    f.write('Best Model: ' + str(final_model.best_estimator_) +
    '\n\nExpected performance/Accuracy on out of sample data: ' + str(round(best_model_accuracy*100, 2)) + " %")
f.close()