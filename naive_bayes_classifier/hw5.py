import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, make_scorer

df = pd.read_csv('~/OneDrive/code/ait636/data/pima-indians-diabetes.csv', index_col=0)
feature_names = df.columns[:-1]
print(df.head())

# Standardize the features
scaler = StandardScaler()
scaler.fit(df.drop('target', axis=1))
StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_features = scaler.transform(df.drop('target', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
#print(df_feat.head())

# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3, stratify=df['target'], random_state=42)

# Apply Naive Bayes

clf = GaussianNB(priors=None) # An array whose size is equal to the number of classes. If not specified or None, the priors are adjusted based on relative class frequencies. Set it to [0.5 0.5] if you want equal priors.

clf = clf.fit(x_train, y_train)

print('Class priors are: ', clf.class_prior_)

# Predictions
predictions_test = clf.predict(x_test)

# Display confusion matrix
cm = metrics.confusion_matrix(y_test, predictions_test, labels=clf.classes_)
cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
cm_disp.plot()
plt.show()

# Report Overall Accuracy, precision, recall, F1-score
class_names = list(map(str, clf.classes_))
print(metrics.classification_report(
    y_true=y_test,
    y_pred=predictions_test,
    target_names=class_names,
    zero_division=0
))

# Hyperparameter Optimization
# Measuring the cross-validation accuracy for different values of hyperparameter and choosing the hyperparameter that results in the highest accuracy
cross_validation_accuracies = []
cross_validation_precisions = []
cross_validation_recalls = []
cross_validation_f1scores = []
cross_validation_roc_auc = []
priors = ([0.001,0.999],
          [0.1,0.9],
          [0.2,0.8],
          [0.3,0.7],
          [0.4,0.6],
          [0.5,0.5],
          [0.6,0.4],
          [0.7,0.3],
          [0.8,0.2],
          [0.9,0.1],
          [0.999,0.001]
          )

for p in priors:
    print('Priors are                :', p)
    clf = GaussianNB(priors=p)
    scores = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
    precision = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
    recall = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
    f1_score = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean()
    roc_auc = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='roc_auc').mean()

    cross_validation_accuracies.append(scores)
    cross_validation_precisions.append(precision)
    cross_validation_recalls.append(recall)
    cross_validation_f1scores.append(f1_score)
    cross_validation_roc_auc.append(roc_auc)

    print(f"Gaussian NB accuracy is  : {scores}")
    print(f"Gaussian NB precision is : {precision}")
    print(f"Gaussian NB recall is    : {recall}")
    print(f"Gaussian NB f1 is        : {f1_score}")
    print(f"Gaussian NB roc_auc is   : {roc_auc}")
    print("--------------------------------------")

plt.figure(figsize=(10, 6))
plt.plot(('0-100','10-90','20-80','30-70','40-60','50-50','60-40','70-30','80-20','90-10','100-0'),
         cross_validation_accuracies, color='blue', 
         linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. Prior')
plt.xlabel('Prior')
plt.ylabel('Accuracy')
plt.show()


