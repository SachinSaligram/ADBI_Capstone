Code Authors: 
Venkata Anudeep Varma Datla
Sachin Saligram
Vikas Rajendra Pandey


Description:
This code evaluates multiple models to predict the a users sentiment/review is based on their text reviews. Models used include Logistic Regression, Multinomial Naive Bayes, Random Forest Classifier and Extra Trees Classifier. The project was done taking a 100,000 subset of the original ~4 million instances of the dataset and taking the average performance over 10 iterations.


Execution:
The code can be executed either in IDE or terminal. The output will be a confusion matrix plot and performance metrics for each model, and a ROC curve (please close all confusion matrix plots for the ROC curve to show up).


Dependencies:
This code requires the following libraries: 
pandas, nltk, numpy, matplotlib, itertools and sklearn.