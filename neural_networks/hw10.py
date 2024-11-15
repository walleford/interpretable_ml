from sklearn.neural_network import MLPClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as py
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as cvs

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
def hyperparameter_optimization(activators, hidden_layers, df, df_feat):
    cv_accuracies = []
    cv_precisions = []
    cv_recalls = []
    cv_f1s = []
    cv_roc_auc = []
    
    for af in activators:
        af_accuracy = []
        for layers in hidden_layers:

            clf = MLPClassifier(
                    random_state=1,
                    hidden_layer_sizes=layers,
                    activation=af,
                    solver='adam',
                    alpha=0.00001,
                    batch_size='auto',
                    learning_rate='adaptive',
                    learning_rate_init=0.001,
                    max_iter=1000,
                    shuffle=True,
                    tol=0.001,
                    early_stopping=False,
                    n_iter_no_change=10
                    )

            scores = cvs(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
            precision = cvs(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
            recall = cvs(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
            f1 = cvs(clf, df_feat, df['target'], cv=10, scoring='f1').mean()
            roc_auc = cvs(clf, df_feat, df['target'], cv=10, scoring='roc_auc').mean()
            af_accuracy.append(float(scores))
            cv_precisions.append(precision)
            cv_recalls.append(recall)
            cv_f1s.append(f1)
            cv_roc_auc.append(roc_auc)
            print(f"{af} with {layers} Accuracy is     : {scores}")
            print(f"{af} with {layers} Precision is    : {precision}")
            print(f"{af} with {layers} Recall is       : {recall}")
            print(f"{af} with {layers} F1 Score is     : {f1}")
            print(f"{af} with {layers} Roc Auc is      : {roc_auc}")
        cv_accuracies.append({af: af_accuracy})
    return cv_accuracies

def main():
    df = pd.read_csv('../data/pima-indians-diabetes.csv', index_col=0)
    feature_names = df.columns[:-1]
    scaled_features, df_feat = scale(df)
    x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3,
                                                        stratify=df['target'], random_state=42)
    activators = [
            'relu',
            'identity',
            'logistic',
            'tanh'
            ]
    hidden_layers = [
            (10),
            (20),
            (10,10),
            (20,20)
            ]
    acc = hyperparameter_optimization(activators, hidden_layers, df, df_feat)
    
    plt.figure(figsize=(10,6))
    plt.plot(activators, acc[0]['relu'], label='relu', linestyle='dashed', marker='o', markersize=10)
    plt.plot(activators, acc[1]['identity'], linestyle='dashed', marker='o', markersize=10, label = 'identity')
    plt.plot(activators, acc[2]['logistic'], linestyle='dashed', marker='o', markersize=10, label = 'logistic')
    plt.plot(activators, acc[3]['tanh'], linestyle='dashed', marker='o', markersize=10, label = 'tanh')
    plt.title('Accuracy vs Hidden Layer Size per Activation Function')
    plt.legend(loc='best')
    plt.ylabel('Accuracy')
    plt.show()
if __name__ == "__main__":
    main()
