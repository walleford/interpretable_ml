import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as py
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as cvs
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

def scale(df):
    """ Scaler Function
        Used to scale the data by removing the mean and scaling to unit variance
        calculated as z = (x-u) / s where x is the standard score, u is the 
        mean of the training samples or zero if `with_mean` is false, and s is the 
        standard deviation of the training samples

        arguments : df = dataframe containing the data read in from the dataset.
        returns   : scaled_features, df_feat
    """
    scaler = StandardScaler()
    scaler.fit(df.drop('target', axis=1))
    StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled_features = scaler.transform(df.drop('target', axis=1))
    df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

    return scaled_features, df_feat

def con_matrix_display(y_test, predictions_test, labels):
    """ Confusion Matrix Display
        displays the confusion matrix for the provided test labels and predictions
    """
    cm = metrics.confusion_matrix(y_test, predictions_test, labels=labels)
    cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_disp.plot()
    plt.show()

def hyperparameter_optimization(c_vals, c_2, classifier_type, df, df_feat, feature_names):
    """ Hyperparameter Optimization
        Optimizes hyperparameters using the provided c value, kernel, and degree to find the best
        for accuracy, precision, recall, f1, and roc_auc

        Requirements : 
            c_vals  = array of smoothing parameter. Smaller values = more regularization.
            degree  = degree of the polynomial function (only for polynomial kernels)
            df      = dataframe of data 
            df_feat = scaled dataframe 

        Returns :
            cv_accuracies = array of accuracies used for plotting later.
    """
    cv_accuracies = []
    cv_precisions = []
    cv_recalls = []
    cv_f1s = []
    cv_roc_auc = []
    
    for c in c_vals:
        leaf_acc = []
        print(f"NEW C VALUE: {c}")
        for c2 in c_2:
            print(f"NEW C_2: {c2}")
            if classifier_type == 'tree':
                clf = tree.DecisionTreeClassifier(max_depth=None, min_samples_split=int(c),
                                                  min_samples_leaf=int(c2))
                if c2 == 3:
                    class_names = list(map(str, clf.classes_))
                    plt.figure(figsize=(16,8))
                    tree.plot_tree(
                            decision_tree=clf,
                            max_depth=3,
                            feature_names=feature_names,
                            class_names=class_names, 
                            filled=True
                            )
            elif classifier_type == 'forest':
                clf = RandomForestClassifier(max_depth=None, 
                                             min_samples_split=int(c),
                                             min_samples_leaf=int(c2),
                                             n_estimators=100)
            
            scores = cvs(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
            precision = cvs(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
            recall = cvs(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
            f1 = cvs(clf, df_feat, df['target'], cv=10, scoring='f1').mean()
            roc_auc = cvs(clf, df_feat, df['target'], cv=10, scoring='roc_auc').mean()
            leaf_acc.append(scores)
            print(f"Accuracy is     : {scores}")
            print(f"Precision is    : {precision}")
            print(f"Recall is       : {recall}")
            print(f"F1 Score is     : {f1}")
            print(f"Roc Auc is      : {roc_auc} \n")
        cv_accuracies.append(leaf_acc)
    for x in cv_accuracies:
        x = [float(y) for y in x]
    return cv_accuracies

def main():
    # loop through, pass hyperparameters into the optimizer and do each of these for the report

    df = pd.read_csv('../data/pima-indians-diabetes.csv', index_col=0)
    feature_names = df.columns[:-1]
    scaled_features, df_feat = scale(df)
    x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3,
                                                        stratify=df['target'], random_state=42)
    clf = tree.DecisionTreeClassifier(max_depth=None, min_samples_split=7,min_samples_leaf=2)
    clf = clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    labels = clf.classes_
    con_matrix_display(y_test, preds, labels)
    c_vals = ['5', '10', '15', '20'] 
    c_2 = ['3','7','11','15']
    accuracies = []
    accuracies.append(hyperparameter_optimization(c_vals, c_2, 'tree', df, df_feat, feature_names))
    forest_acc = []
    forest_acc.append(hyperparameter_optimization(c_vals,c_2, 'forest', df, df_feat, feature_names))

    plt.figure(figsize=(10,6))
    plt.plot(c_2, accuracies[0], linestyle='dashed', marker='o', markersize=10, label = '3')
    plt.plot(c_2, accuracies[1], linestyle='dashed', marker='o', markersize=10, label = '7')
    plt.plot(c_2, accuracies[2], linestyle='dashed', marker='o', markersize=10, label = '11')
    plt.plot(c_2, accuracies[3], linestyle='dashed', marker='o', markersize=10, label = '15')
    plt.title('Accuracy vs Min Samples Leaf')
    plt.legend(loc='best')
    plt.xlabel('Accuracies')
    plt.ylabel('Min Samples Leaf')
    plt.show()

if __name__ == '__main__':
    main()
