import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as cvs
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression


def scale(df):
    scaler = StandardScaler()
    scaler.fit(df.drop('target', axis=1))
    StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled_features = scaler.transform(df.drop('target', axis=1))
    df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

    return scaled_features, df_feat


def con_matrix_display(y_test, predictions_test, labels):
    cm = metrics.confusion_matrix(y_test, predictions_test, labels=labels)
    cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_disp.plot()
    plt.show()


def hyperparameter_optimization(c_vals, df, df_feat):
    cv_accuracies = []
    cv_precisions = []
    cv_recalls = []
    cv_f1s = []
    cv_roc_auc = []
    labels = ()
    for c in c_vals:
        clf = LogisticRegression(C=float(c))
        print(f"c is {c}")
        scores = cvs(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
        precision = cvs(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
        recall = cvs(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
        f1 = cvs(clf, df_feat, df['target'], cv=10, scoring='f1').mean()
        roc_auc = cvs(clf, df_feat, df['target'], cv=10, scoring='roc_auc').mean()
        cv_accuracies.append(scores)
        cv_precisions.append(precision)
        cv_recalls.append(recall)
        cv_f1s.append(f1)
        cv_roc_auc.append(roc_auc)

        print(f"LogReg Accuracy is     : {scores}")
        print(f"LogReg Precision is    : {precision}")
        print(f"LogReg Recall is       : {recall}")
        print(f"LogReg F1 Score is     : {f1}")
        print(f"LogReg Roc Auc is      : {roc_auc}")

    return cv_accuracies, labels

def main():
    df = pd.read_csv('../data/pima-indians-diabetes.csv', index_col=0)
    feature_names = df.columns[:-1]
    scaled_features, df_feat = scale(df)
    x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3,
                                                        stratify=df['target'], random_state=42)
    print("Confusion Matrix for C=2:")
    clf = LogisticRegression(C=2)
    clf = clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    labels = clf.classes_
    con_matrix_display(y_test, preds, labels)
    
    x_labels = [
            '0.01',
            '0.05',
            '0.1',
            '0.15',
            '0.2',
            '0.25',
            '0.3',
            '0.35',
            '0.4',
            '0.45',
            '0.5',
            '0.55',
            '0.6',
            '0.65',
            '0.7',
            '0.75',
            '0.8',
            '0.85',
            '0.9',
            '0.95',
            '1.0',
            ]

    acc = hyperparameter_optimization(x_labels, df, df_feat)
    acc = [float(x) for x in acc[0]]
    plt.figure(figsize=(10,6))
    plt.plot(x_labels, acc, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',
             markersize=10)
    plt.title('Accuracy vs C Value')
    plt.xlabel('C Values')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    main()
