import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model

df = pd.read_csv('./data/pima-indians-diabetes.csv', index_col=0)
feature_names = df.columns[:-1]
print(df.head())

# Standardize the features
scaler = StandardScaler()
scaler.fit(df.drop('target', axis=1))
StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_features = scaler.transform(df.drop('target', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
#print(df_feat.head())

sns.pairplot(df, hue='target')
#plt.show()

# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3, stratify=df['target'], random_state=42)

# Apply Least Squares
# Ordinary least squares
clf = linear_model.LinearRegression()
# Ridge regression
'''clf = linear_model.Ridge(alpha=.5, # Regularization strength: Larger values specify stronger regularization.
                 random_state=0, # # The seed of the pseudo random number generator to use when shuffling the data.
                 )
# Lasso regression has a tendency to prefer solutions with fewer parameter values, effectively reducing the number of variables.'''
clf = linear_model.Lasso(alpha=0.1, # Regularization strength: Larger values specify stronger regularization. Default is 1.
                     random_state=0, # The seed of the pseudo random number generator that selects a random feature to update.
                     )

clf = clf.fit(x_train, y_train)

# Predictions
predictions_test = clf.predict(x_test)
class_names = [0, 1]
predictions_test[predictions_test <= 0.5] = 0
predictions_test[predictions_test > 0.5] = 1
predictions_test = predictions_test.astype(int)

# Display confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=class_names)
confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=class_names)
confusion_matrix_display.plot()
#plt.show()

# Report Overall Accuracy, precision, recall, F1-score
print(metrics.classification_report(
    y_true=y_test,
    y_pred=predictions_test,
    target_names=list(map(str, class_names)),
    zero_division=0 # Whenever number is divided by zero, instead of nan, return 0
))

# Optimize alpha
overall_accuracies = []
alpha_values = np.arange(0, 2, 0.2) # range(start, stop, step) will result in an error because range can only produce a list of integers not floats
for i in alpha_values:
    print('alpha is:', i)
    clf = linear_model.Lasso(alpha=i, random_state=0)
    clf = clf.fit(x_train, y_train)
    predictions_test = clf.predict(x_test)
    # Replace regression results by class labels
    predictions_test[predictions_test <= 0.5] = 0
    predictions_test[predictions_test > 0.5] = 1
    predictions_test = predictions_test.astype(int)

    overall_accuracies.append(metrics.accuracy_score(y_true=y_test, y_pred=predictions_test))

    # Find the confusion matrix and accuracy metrics using this value for the hyperparameter and the training and test data that you created before.
    # Display confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=class_names)
    confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=class_names)
    confusion_matrix_display.plot()
#    plt.show()

    # Report Overall Accuracy, precision, recall, F1-score
    print(metrics.classification_report(
        y_true=y_test,
        y_pred=predictions_test,
        target_names=list(map(str, class_names)),
        zero_division=0 # Whenever number is divided by zero, instead of nan, return 0
    ))


# Create a graph that shows the overall accuracy for different values of the hyperparameter.
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, overall_accuracies, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. alpha_value')
plt.xlabel('alpha-value')
plt.ylabel('Accuracy')
plt.show()

