 # Natural Language Processing for Predict Network Anomaly

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib

# Importing the dataset
dataset = pd.read_csv('dataset.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 5550):
    log = re.sub('[^a-zA-Z0-9]', ' ', dataset['Info'][i])
    log = log.lower()
    log = log.split()
    ps = PorterStemmer()
    log = [ps.stem(word) for word in log if not word in set(stopwords.words('english'))]
    log = ' '.join(log)
    corpus.append(log)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)

#Accuracy
from sklearn.metrics import accuracy_score
# y_test is the actual values, y_pred is the predicted values
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % accuracy)

#Precision Score
from sklearn.metrics import precision_score
# y_test is the actual values, y_pred is the predicted values
precision = precision_score(y_test, y_pred, average='binary')
print('Precision: %.3f' % precision)

#Recall Score
from sklearn.metrics import recall_score
# y_test is the actual values, y_pred is the predicted values
recall = recall_score(y_test, y_pred)
print('Recall: %.3f' % recall)

#F1 Score
from sklearn.metrics import f1_score
# y_test is the actual values, y_pred is the predicted values
f1 = f1_score(y_test, y_pred, average='binary')
print('F1 Score: %.3f' % f1)

print("Predicted Outcomes of the Model")
print(y_pred)

print("Actual Outcomes of the Predicted Results")
print(y_test)

# Visualizing the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculating the number of true positives, true negatives, false positives, and false negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Creating a bar chart to show the comparison between the predicted and true outcomes
fig, ax = plt.subplots()
ax.bar(['True Negatives', 'False Positives', 'False Negatives', 'True Positives'], [tn, fp, fn, tp])
ax.set_xlabel('Outcome')
ax.set_ylabel('Count')
ax.set_title('Comparison of Predicted and True Outcomes')
plt.show()
########################################
# Dumping the trained model to a file
joblib.dump(classifier, 'naive_bayes_model.joblib')

# Dumping the CountVectorizer
joblib.dump(cv, 'count_vectorizer.joblib')