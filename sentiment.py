# Import libraries and methods
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem import WordNetLemmatizer
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn import ensemble

# Download corpus from NLTK library
nltk.download('stopwords')
nltk.download('punkt')

# Obtain 'English' stop words
stops = set(stopwords.words("english"))

# Initialize the "TfidfVectorizer" object, which is scikit-learn's bag of words and tfid tool.
vectorizer = TfidfVectorizer(max_features=None, ngram_range=(1, 2))


# Define word_lemmatize to perform lemmatization
def word_lemmatize(word_lemma):
    input_lemma = []
    for word in word_lemma:
        word = WordNetLemmatizer().lemmatize(word)
        input_lemma.append(word)
    return input_lemma


# Define plot_confusion_matrix to plot confusion matrix for each model
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="grey" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Format positive to 1 and negative to 0
def format(x):
    predicted_mapping = []
    for score in x:
        if score == 'negative':
            output = 0
        else:
            output = 1
        predicted_mapping.append(output)
    return predicted_mapping


# Import JSON file
with open('/yelp_academic_dataset_review.json', 'rb') as f:
    data = f.readlines()
data = map(lambda x: x.rstrip(), data)
data_json_str = "[" + ','.join(data) + "]"
data1 = pd.read_json(data_json_str)
names = ['text', 'stars']
dataset1 = data1.ix[:, names]

# Delete to free memory
lst = [data, data_json_str, data1]
del lst

# Define lists to hold predicted values for each model
predictionLR = list()
predictionML = list()
predictionETC = list()
predictionRF = list()

test = {}
results = []
rms = {}
y_test = []
count = 0

# Loop to compute the average performance of 10 samples of data
while count < 10:

    # Sample 100,000 rows of the original ~4 million rows
    dataset = dataset1.sample(n=100000)

    # Assign seperate variables to store specific data of the original dataset
    score_data = dataset['stars']
    # summary_data = dataset['Summary']
    text_data = dataset['text']

    # Remove puncuations from summary and text data. Keep only letters.
    text = text_data.str.replace('[^a-zA-Z]', " ")

    # Map the score rating to positive/negative
    result = []
    for score in score_data:
        if score < 3:
            output = 'negative'
        else:
            output = 'positive'
        result.append(output)

    # Delete to free up memory
    lst = [score_data, text_data]

    # Make every review lower case, and tokenize and lemmatize
    corpus = []
    for word in text:
        word = word.lower()
        word = nltk.word_tokenize(word)
        word = word_lemmatize(word)
        # tokens = [w for w in word if not w in stops]
        corpus.append(' '.join(word))

    # Split the features and class labels into training and test datasets
    x_train, x_test = train_test_split(corpus, test_size=0.3, random_state=43)
    y_train, y_t = train_test_split(result, test_size=0.3, random_state=43)

    # fit_transform() does two functions: First, it fits the model and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of strings.
    X_train_tfidf = vectorizer.fit_transform(x_train)
    X_test_tfidf = vectorizer.transform(x_test)

    # Fit the logistic regression model and predict using test data
    model = LogisticRegression(C=1e5).fit(X_train_tfidf, y_train)
    # Test the model using testing data
    pred = model.predict(X_test_tfidf)
    y_test.extend(y_t)
    predictionLR.extend(pred)

    # Fit the MultinomialNB model and predict using test data
    model = MultinomialNB().fit(X_train_tfidf, y_train)
    # Test the model using testing data
    pred = model.predict(X_test_tfidf)
    predictionML.extend(pred)

    # # Train the ExtraTreesClassifier model using training data
    model = ensemble.ExtraTreesClassifier().fit(X_train_tfidf, y_train)
    # # Test the model using testing data
    pred = model.predict(X_test_tfidf)
    predictionETC.extend(pred)

    # # Train the RandomForestClassifier model using training data
    model = ensemble.RandomForestClassifier().fit(X_train_tfidf, y_train)
    # # Test the model using testing data
    pred = model.predict(X_test_tfidf)
    predictionRF.extend(pred)

# Display performance metrics
print(metrics.classification_report(y_test, predictionLR, target_names=["positive", "negative"]))
nb_cnf_matrix = confusion_matrix(y_test, predictionLR)
np.set_printoptions(precision=10)
test['LR'] = test.get('LR', np.matrix("0 0;0 0")) + nb_cnf_matrix
rms['LR'] = rms.get('LR', 0) + mean_squared_error(format(list(y_test)), format(predictionLR))

# Display performance metrics
print(metrics.classification_report(y_test, predictionLR, target_names=["positive", "negative"]))
nb_cnf_matrix = confusion_matrix(y_test, predictionML)
np.set_printoptions(precision=2)
test['Multinomial'] = test.get('Multinomial', np.matrix("0 0;0 0")) + nb_cnf_matrix
rms['Multinomial'] = rms.get('Multinomial', 0) + accuracy_score(format(list(y_test)), format(predictionML), normalize=False)

# Display performance metrics
print(metrics.classification_report(y_test, predictionETC, target_names=["positive", "negative"]))
nb_cnf_matrix = confusion_matrix(y_test, predictionETC)
np.set_printoptions(precision=2)
test['Extra Trees'] = test.get('Extra Trees', np.matrix("0 0;0 0")) + nb_cnf_matrix
rms['Extra Trees'] = rms.get('Extra Trees', 0) + accuracy_score(format(list(y_test)), format(predictionETC), normalize=False)

# Display performance metrics
print(metrics.classification_report(y_test, predictionRF, target_names=["positive", "negative"]))
nb_cnf_matrix = confusion_matrix(y_test, predictionRF)
np.set_printoptions(precision=2)
test['Random Forest'] = test.get('Random Forest', np.matrix("0 0;0 0")) + nb_cnf_matrix
rms['Random Forest'] = rms.get('Random Forest', 0) + accuracy_score(format(list(y_test)), format(predictionRF), normalize=False)

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(test['LR'], classes=['positive', 'negative'], title='LR Confusion Matrix (Without Normalization)')

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(test['Multinomial'], classes=['positive', 'negative'], title='Multinomial Naive Bayes Confusion Matrix (Without Normalization)')

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(test['Extra Trees'], classes=['positive', 'negative'], title='Extra Trees Confusion Matrix (Without Normalization)')

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(test['Random Forest'], classes=['positive', 'negative'], title='Random Forest Confusion Matrix (Without Normalization)')

# Display accuracy
print("accuracy of LR", accuracy_score(y_test, predictionLR))
print("accuracy of Multinomial", accuracy_score(y_test, predictionML))
print("accuracy of Extra", accuracy_score(y_test, predictionETC))
print("accuracy of Random", accuracy_score(y_test, predictionRF))

plt.show()

# Plot ROC Curve
prediction = {}
prediction['LR'] = predictionLR
prediction['Multinomial'] = predictionML
prediction['Extra Trees'] = predictionETC
prediction['Random Forest'] = predictionRF
y_test_mapping = []
for score in y_test:
    if score == 'negative':
        output = 0
    else:
        output = 1
    y_test_mapping.append(output)
cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(np.array(y_test_mapping), np.array(format(predicted)))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f' % (model, roc_auc))
    cmp += 1
plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
