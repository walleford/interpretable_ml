import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as py
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as cvs
from sklearn.svm import SVC

""" Scaler Function
    Used to scale the data by removing the mean and scaling to unit variance
    calculated as z = (x-u) / s where x is the standard score, u is the 
    mean of the training samples or zero if `with_mean` is false, and s is the 
    standard deviation of the training samples

    arguments : df = dataframe containing the data read in from the dataset.
    returns   : scaled_features, df_feat
"""
def scale(df):
    scaler = StandardScaler()
    scaler.fit(df.drop('target', axis=1))
    StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled_features = scaler.transform(df.drop('target', axis=1))
    df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

    return scaled_features, df_feat

""" Confusion Matrix Display
    displays the confusion matrix for the provided test labels and predictions
"""
def con_matrix_display(y_test, predictions_test, labels):
    cm = metrics.confusion_matrix(y_test, predictions_test, labels=labels)
    cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_disp.plot()
    plt.show()

""" Hyperparameter Optimization
    Optimizes hyperparameters using the provided c value, kernel, and degree to find the best
    for accuracy, precision, recall, f1, and roc_auc

    Requirements : 
        c_vals  = array of smoothing parameter. Smaller values = more regularization.
        kernel  = kernel function to use for finding best support vector classifier
        degree  = degree of the polynomial function (only for polynomial kernels)
        df      = dataframe of data 
        df_feat = scaled dataframe 

    Returns :
        cv_accuracies = array of accuracies used for plotting later.
"""
def hyperparameter_optimization(c_vals, kernel, df, df_feat, degree=None):
    cv_accuracies = []
    cv_precisions = []
    cv_recalls = []
    cv_f1s = []
    cv_roc_auc = []
    
    for c in c_vals:
        if kernel == 'poly':
            clf = SVC(C=float(c), kernel=kernel, degree=degree)
            print(f"C  = {c}")
            print(f"Kernel = {kernel}")
            print(f"Degree = {degree}")
        else:
            clf = SVC(C=float(c), kernel=kernel)
            print(f"C = {c}")
            print(f"Kernel = {kernel}")
            print(f"Degree = {degree}")

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

    return [float(x) for x in cv_accuracies]

def main():
    df = pd.read_csv('../data/pima-indians-diabetes.csv', index_col=0)
    feature_names = df.columns[:-1]
    scaled_features, df_feat = scale(df)
    x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3,
                                                        stratify=df['target'], random_state=42)
    print("Confusion Matrix for C=1.0, poly kernel, and degree of 3")
    clf = SVC(C=1.0, kernel='poly', degree=3)
    clf = clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    labels = clf.classes_
    con_matrix_display(y_test, preds, labels)
    c_vals = ['0.5', '1.0', '1.5', '2.0'] 
    accuracies = []
    acc1 = hyperparameter_optimization(c_vals, 'poly', df, df_feat,2)
    acc2 = hyperparameter_optimization(c_vals, 'poly', df, df_feat,3)
    acc3 = hyperparameter_optimization(c_vals, 'poly', df, df_feat,4)
    acc4 = hyperparameter_optimization(c_vals, 'linear', df, df_feat)
    acc5 = hyperparameter_optimization(c_vals, 'sigmoid', df, df_feat)
    acc6 = hyperparameter_optimization(c_vals, 'rbf', df, df_feat)

    plt.figure(figsize=(10,6))
    print(acc1)
    plt.plot(c_vals, acc1, label='poly 2', linestyle='dashed', marker='o', markersize=10)
    plt.plot(c_vals, acc2, linestyle='dashed', marker='o', markersize=10, label = 'poly 3')
    plt.plot(c_vals, acc3, linestyle='dashed', marker='o', markersize=10, label = 'poly 4')
    plt.plot(c_vals, acc4, linestyle='dashed', marker='o', markersize=10, label = 'linear')
    plt.plot(c_vals, acc5, linestyle='dashed', marker='o', markersize=10, label = 'sigmoid')
    plt.plot(c_vals, acc6, linestyle='dashed', marker='o', markersize=10, label = 'rbf')
    plt.title('Accuracy vs C Value per Kernel')
    plt.legend(loc='best')
    plt.xlabel('C Values')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    main()
